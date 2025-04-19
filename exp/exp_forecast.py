import os
import time
import warnings

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.metrics import metric
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map
from utils.augmentation import run_augmentation_single

from utils.adaptation import adaptivity

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):

    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args)
            model = DDP(model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model = self.model_dict[self.args.model].Model(self.args)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.use_weight_decay:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate,
                                     weight_decay=self.args.weight_decay)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion, epoch=0, flag='vali'):
        total_loss = []
        total_count = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float()
                if self.args.output_attention:
                    # output used to calculate loss misaligned patch_len compared to input
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    # only use the forecast window to calculate loss
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_ims:
                    pred = outputs[:, -self.args.seq_len:, :]
                    true = batch_y
                    if flag == 'vali':
                        loss = criterion(pred, true)
                    elif flag == 'test':  # in this case, only pred_len is used to calculate loss
                        pred = pred[:, -self.args.pred_len:, :]
                        true = true[:, -self.args.pred_len:, :]
                        loss = criterion(pred, true)
                else:
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                loss = loss.detach().cpu()
                total_loss.append(loss)
                total_count.append(batch_x.shape[0])
                torch.cuda.empty_cache()

        if self.args.use_multi_gpu:
            total_loss = torch.tensor(np.average(total_loss, weights=total_count)).to(self.device)
            dist.barrier()
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            total_loss = total_loss.item() / dist.get_world_size()
        else:
            total_loss = np.average(total_loss, weights=total_count)
        self.model.train()
        return total_loss

    def finetune(self, setting):
        finetune_data, finetune_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        test_data, test_loader = data_provider(self.args, flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(finetune_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        scheduler = LargeScheduler(self.args, model_optim)

        for epoch in range(self.args.finetune_epochs):
            iter_count = 0

            loss_val = torch.tensor(0., device="cuda")
            count = torch.tensor(0., device="cuda")

            self.model.train()
            epoch_time = time.time()

            print("Step number per epoch: ", len(finetune_loader))
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(finetune_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if self.args.use_ims:
                    # output used to calculate loss misaligned patch_len compared to input
                    loss = criterion(outputs[:, -self.args.seq_len:, :], batch_y)
                else:
                    # only use the forecast window to calculate loss
                    
                    loss = criterion(outputs[:, -self.args.pred_len:, :], batch_y[:, -self.args.pred_len:, :])

                loss_val += loss
                count += 1

                if i % 50 == 0:
                    cost_time = time.time() - time_now
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f} | cost_time: {3:.0f} | memory: allocated {4:.0f}MB, reserved {5:.0f}MB, cached {6:.0f}MB "
                        .format(i, epoch + 1, loss.item(), cost_time,
                                torch.cuda.memory_allocated() / 1024 / 1024,
                                torch.cuda.memory_reserved() / 1024 / 1024,
                                torch.cuda.memory_cached() / 1024 / 1024))
                    time_now = time.time()

                loss.backward()
                model_optim.step()
                torch.cuda.empty_cache()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            if self.args.use_multi_gpu:
                dist.barrier()
                dist.all_reduce(loss_val, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            train_loss = loss_val.item() / count.item()

            vali_loss = self.vali(vali_data, vali_loader, criterion)
            if self.args.train_test:
                test_loss = self.vali(test_data, test_loader, criterion, flag='test')
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss))


            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            scheduler.schedule_epoch(epoch)

        best_model_path = path + '/' + 'checkpoint.pth'
        if self.args.use_multi_gpu:
            dist.barrier()
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):

        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        attns = []
        folder_path = './test_results/' + setting + '/' + self.args.data_path + '/' + f'{self.args.output_len}/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        self.model.eval()
        if self.args.output_len_list is None:
            self.args.output_len_list = [self.args.output_len]

        preds_list = [[] for _ in range(len(self.args.output_len_list))]
        trues_list = [[] for _ in range(len(self.args.output_len_list))]
        self.args.output_len_list.sort()

        with torch.no_grad():
            for output_ptr in range(len(self.args.output_len_list)):
                self.args.output_len = self.args.output_len_list[output_ptr]
                test_data, test_loader = data_provider(self.args, flag='test')
                
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    if self.args.augmentation_ratio > 0:
                        true_batch_x, true_batch_y = batch_x.clone(), batch_y.clone()
                        batch_x, batch_y, augmentation_tags = run_augmentation_single(batch_x, batch_y, self.args)
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                    inference_steps = self.args.output_len // self.args.pred_len
                    dis = self.args.output_len - inference_steps * self.args.pred_len
                    if dis != 0:
                        inference_steps += 1
                    pred_y = []
                    for j in range(inference_steps):
                        if len(pred_y) != 0:
                            batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                            tmp = batch_y_mark[:, j - 1:j, :]
                            batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

                        if self.args.output_attention:
                            outputs, attns = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        pred_y.append(outputs[:, -self.args.pred_len:, :])
                    pred_y = torch.cat(pred_y, dim=1)

                    if dis != 0:
                        pred_y = pred_y[:, :-dis, :]

                    if self.args.use_ims:
                        batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(
                            self.device)
                    else:
                        batch_y = batch_y[:, :self.args.output_len, :].to(self.device)

                    outputs = pred_y.detach().cpu()
                    batch_y = batch_y.detach().cpu()
                    if self.args.augmentation_ratio > 0:
                        true_batch_y = batch_y.clone()
                        true_batch_y, batch_x, augmentation_tags = run_augmentation_single(true_batch_y, batch_x, self.args)
                        
                    if test_data.scale and self.args.inverse:
                        shape = outputs.shape
                        outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                        batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
                        if self.args.augmentation_ratio > 0:
                            true_batch_y = test_data.inverse_transform(true_batch_y.squeeze(0)).reshape(shape)

                    outputs = outputs[:, :, f_dim:]
                    batch_y = batch_y[:, :, f_dim:]
                    if self.args.augmentation_ratio > 0:
                        true_batch_y = true_batch_y[:, :, f_dim:]

                    pred = outputs
                    true = batch_y

                    preds_list[output_ptr].append(pred)
                    if self.args.augmentation_ratio == 0:
                        trues_list[output_ptr].append(true)
                    else:
                        aug_true = true_batch_y
                        trues_list[output_ptr].append(true)
                    
                    if i % 10 == 0:
                        if self.args.augmentation_ratio == 0:
                            input = batch_x.detach().cpu().numpy()
                            gt = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
                            pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)
                        else:
                            # if augmentation:
                            # pred: prediction output
                            # aug_true: aug output
                            # true: no aug output
                            # input: aug input
                            # true_input: no aug input
                            input = batch_x.detach().cpu().numpy()
                            gt = np.concatenate((input[0, -self.args.pred_len:, -1], aug_true[0, :, -1]), axis=0)
                            pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)
                            true_input = true_batch_x.detach().cpu().numpy()
                            tgt = np.concatenate((true_input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)

                        if self.args.local_rank == 0:
                            if self.args.output_attention:
                                attn = attns[0].cpu().numpy()[0, 0, :, :]
                                attn_map(attn, os.path.join(folder_path, f'attn_{i}_{self.args.local_rank}.pdf'))
                            if self.args.augmentation_ratio == 0:
                                visual(gt, pd, name=os.path.join(folder_path, f'{i}_{self.args.local_rank}.pdf'))
                            else:
                                aug_gt = gt
                                visual(tgt, pd, aug_gt, os.path.join(folder_path, f'{i}_{self.args.local_rank}.pdf'))

        if self.args.output_len_list is not None:
            for i in range(len(preds_list)):
                preds = preds_list[i]
                trues = trues_list[i]
                preds = torch.cat(preds, dim=0).numpy()
                trues = torch.cat(trues, dim=0).numpy()
                
                mae, mse, rmse, mape, mspe = metric(preds, trues)
                print(f"output_len: {self.args.output_len_list[i]}")

                print('mse:{}, mae:{}'.format(mse, mae))
                f = open("result_long_term_forecast.txt", 'a')
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}'.format(mse, mae))
                f.write('\n')
                f.write('\n')
                f.close()

        return

    def any_inference(self, batch_x, pred_len):
        if self.args.output_attention:
            outputs, attns = self.model(batch_x)
        else:
            outputs = self.model(batch_x)
        return outputs
    
    # 适应性测试，计算适应性指标
    def adaptation_test(self, setting):
        print('Model parameters: ', sum(param.numel() for param in self.model.parameters()))
        attns = []
        folder_path = './test_results/' + setting + '/' + self.args.data_path + '/' + f'{self.args.output_len}/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        self.model.eval()
        if self.args.output_len_list is None:
            self.args.output_len_list = [self.args.output_len]

        trues_list = [[] for _ in range(len(self.args.output_len_list))]
        trans_list = [[] for _ in range(len(self.args.output_len_list))]
        self.args.output_len_list.sort()
        adp_type = ""
        
        # pic_path = './adapt/' + setting + '/' + self.args.data_path + '/'

        with torch.no_grad():
            for output_ptr in range(len(self.args.output_len_list)):
                self.args.output_len = self.args.output_len_list[output_ptr]
                test_data, test_loader = data_provider(self.args, flag='test')
                
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    # 适应性数据替换
                    if i % 2 == 0:
                        trans_batch_x, adp_type = adaptivity(batch_x, i, self.args, rd=1)
                    else:
                        trans_batch_x, adp_type = adaptivity(batch_x, i, self.args, rd=1)
                    # 推理
                    trans_pred, true = self.adaptation_forecast(test_data, trans_batch_x, batch_y, batch_x_mark, batch_y_mark)

                    trues_list[output_ptr].append(true)
                    trans_list[output_ptr].append(trans_pred)

        if self.args.output_len_list is not None:
            for i in range(len(trues_list)):
                true = trues_list[i]
                trans_preds = trans_list[i]
                true = torch.cat(true, dim=0).numpy()
                trans_preds = torch.cat(trans_preds, dim=0).numpy()
                
                mae, mse, rmse, mape, mspe = metric(true, trans_preds)
                print(f"output_len: {self.args.output_len_list[i]}")
                print('{} adaptation:{}'.format(adp_type, mae))
        
        return

    def adaptation_forecast(self, test_data, batch_x, batch_y, batch_x_mark, batch_y_mark):
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        inference_steps = self.args.output_len // self.args.pred_len
        dis = self.args.output_len - inference_steps * self.args.pred_len
        if dis != 0:
            inference_steps += 1
        pred_y = []
        for j in range(inference_steps):
            if len(pred_y) != 0:
                batch_x = torch.cat([batch_x[:, self.args.pred_len:, :], pred_y[-1]], dim=1)
                tmp = batch_y_mark[:, j - 1:j, :]
                batch_x_mark = torch.cat([batch_x_mark[:, 1:, :], tmp], dim=1)

            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            pred_y.append(outputs[:, -self.args.pred_len:, :])
        pred_y = torch.cat(pred_y, dim=1)

        if dis != 0:
            pred_y = pred_y[:, :-dis, :]

        if self.args.use_ims:
            batch_y = batch_y[:, self.args.label_len:self.args.label_len + self.args.output_len, :].to(
                self.device)
        else:
            batch_y = batch_y[:, :self.args.output_len, :].to(self.device)

        outputs = pred_y.detach().cpu()
        batch_y = batch_y.detach().cpu()
            
        if test_data.scale and self.args.inverse:
            shape = outputs.shape
            outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
            batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)

        outputs = outputs[:, :, f_dim:]
        batch_y = batch_y[:, :, f_dim:]

        pred = outputs
        true = batch_y
        
        return pred, true

    # 验证适应性指标
    def adaptation_varify(self, setting):
        attns = []
        folder_path = './test_results/' + setting + '/' + self.args.data_path + '/' + f'{self.args.output_len}/'
        if not os.path.exists(folder_path) and int(os.environ.get("LOCAL_RANK", "0")) == 0:
            os.makedirs(folder_path)
        self.model.eval()
        if self.args.output_len_list is None:
            self.args.output_len_list = [self.args.output_len]

        trues_list = [[] for _ in range(len(self.args.output_len_list))]
        trans_list = [[] for _ in range(len(self.args.output_len_list))]
        self.args.output_len_list.sort()
        adp_type = ""
        
        # pic_path = './adapt/' + setting + '/' + self.args.data_path + '/'

        with torch.no_grad():
            for output_ptr in range(len(self.args.output_len_list)):
                self.args.output_len = self.args.output_len_list[output_ptr]
                test_data, test_loader = data_provider(self.args, flag='test')
                
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                    if i % 50 == 0 or i % 51 == 0:
                        # 生成系统随机数
                        random_bytes = os.urandom(4)
                        random_int = int.from_bytes(random_bytes, 'big')
                        random_float = (random_int >> 11) / (1 << 23) + 0.5
                        # 适应性数据替换
                        trans_batch_x, adp_type = adaptivity(batch_x, i, self.args, rd=random_float)
                    # 推理
                    trans_pred, true = self.adaptation_forecast(test_data, trans_batch_x, batch_y, batch_x_mark, batch_y_mark)
                    
                    trues_list[output_ptr].append(true)
                    trans_list[output_ptr].append(trans_pred)

        if self.args.output_len_list is not None:
            for i in range(len(trues_list)):
                true = trues_list[i]
                trans_preds = trans_list[i]
                true = torch.cat(true, dim=0).numpy()
                trans_preds = torch.cat(trans_preds, dim=0).numpy()
                
                mae, mse, rmse, mape, mspe = metric(true, trans_preds)
                print('{} adaptation verify:{}'.format(adp_type, mae))
        
        return


    def fgsm_attack(self, data, epsilon, data_grad):
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        perturbed_data = data + epsilon*sign_data_grad
        # Adding clipping to maintain [0,1] range
        # perturbed_data = torch.clamp(perturbed_data, 0, 1)
        # Return the perturbed image
        return perturbed_data

    def adversarial_attack(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            if self.args.ckpt_path != '':
                if self.args.ckpt_path == 'random':
                    print('loading model randomly')
                else:
                    print('loading model: ', self.args.ckpt_path)
                    if self.args.ckpt_path.endswith('.pth'):
                        self.model.load_state_dict(torch.load(self.args.ckpt_path))
                    else:
                        raise NotImplementedError
            else:
                print('loading model with settings: {}'.format(setting))
                self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            
        preds = []
        trues = []
        adv_preds = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
            model_optim.zero_grad()
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x.requires_grad= True

            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)

            # decoder input
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            outputs = outputs[:, -self.args.pred_len:, :]
            batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
            loss = criterion(outputs,batch_y)
            loss.backward()
            data_grad = batch_x.grad.data
            if i % 2 == 0:
                adv_batch_x = self.fgsm_attack(batch_x, 0.1, data_grad)
            else:
                adv_batch_x = self.fgsm_attack(batch_x, 0.07, data_grad)

            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        adv_outputs = self.model(adv_batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        adv_outputs = self.model(adv_batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    adv_outputs = self.model(adv_batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    adv_outputs = self.model(adv_batch_x, batch_x_mark, dec_inp, batch_y_mark)

            f_dim = -1 if self.args.features == 'MS' else 0
            adv_outputs = adv_outputs[:, -self.args.pred_len:, :]

            outputs = outputs.detach().cpu().numpy()
            adv_outputs = adv_outputs.detach().cpu().numpy()
            batch_y = batch_y.detach().cpu().numpy()

            if ',' in self.args.data:
                if test_data.datasets[0].scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.squeeze(0)).reshape(shape)
            else:
                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    adv_outputs = test_data.inverse_transform(adv_outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)

            outputs = outputs[:, :, f_dim:]
            batch_y = batch_y[:, :, f_dim:]
            adv_outputs = adv_outputs[:, :, f_dim:]

            pred = outputs
            true = batch_y
            adv_pred = adv_outputs

            preds.append(pred)
            trues.append(true)
            adv_preds.append(adv_pred)

            if i % 5 == 0:
                input = batch_x.detach().cpu().numpy()
                adv_input = adv_batch_x.detach().cpu().numpy()
                if ',' in self.args.data:
                    if test_data.datasets[0].scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                else:
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                        adv_input = test_data.inverse_transform(adv_input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                # gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                # pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                # adv = np.concatenate((adv_input[0, :, -1], adv_pred[0, :, -1]), axis=0)
                # visual(gt, pd, adv, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        adv_preds = np.array(adv_preds)
        # print('test shape:', preds.shape, trues.shape, adv_preds.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        adv_preds = adv_preds.reshape(-1, adv_preds.shape[-2], adv_preds.shape[-1])
        # print('test shape:', preds.shape, trues.shape, adv_preds.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        adv_mae, adv_mse, adv_rmse, adv_mape, adv_mspe = metric(adv_preds, trues)
        print(f"output_len: {self.args.output_len}")
        print('Before attacking: mse:{}, mae:{}'.format(mse, mae))
        print('After attacking: mse:{}, mae:{}'.format(adv_mse, adv_mae))
        print("Adversary adaptation:{}".format(adv_mae))

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        f.write('Before attacking: mse:{}, mae:{}'.format(mse, mae))
        f.write('After attacking: mse:{}, mae:{}'.format(adv_mse, adv_mae))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return