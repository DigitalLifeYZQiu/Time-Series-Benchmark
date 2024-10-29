import os
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP

from data_provider.data_factory import data_provider
from utils.metrics import metric
from utils.tools import EarlyStopping, visual, LargeScheduler, attn_map
from exp.exp_benchmark_basic import Exp_Benchmark_Basic


class Exp_Benchmark(Exp_Benchmark_Basic):
    def _build_model(self):
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = self.model_dict[self.args.model].Model(self.args)
            model = DDP(model.cuda(), device_ids=[self.args.local_rank], find_unused_parameters=True)
        else:
            self.args.device = self.device
            model=self.model_dict[self.args.model]
        return model(self.args, self.args.model, self.args.ckpt_path, self.device.type)

    
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
    
    def test(self, setting, test=0):
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
                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

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
                            outputs, attns = self.model.forecast(batch_x, self.args.pred_len)
                        else:
                            outputs = self.model.forecast(batch_x, self.args.pred_len)

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

                    preds_list[output_ptr].append(pred)
                    trues_list[output_ptr].append(true)
                    if i % 10 == 0:
                        input = batch_x.detach().cpu().numpy()
                        gt = np.concatenate((input[0, -self.args.pred_len:, -1], true[0, :, -1]), axis=0)
                        pd = np.concatenate((input[0, -self.args.pred_len:, -1], pred[0, :, -1]), axis=0)

                        if self.args.local_rank == 0:
                            if self.args.output_attention:
                                attn = attns[0].cpu().numpy()[0, 0, :, :]
                                attn_map(attn, os.path.join(folder_path, f'attn_{i}_{self.args.local_rank}.pdf'))

                            visual(gt, pd, os.path.join(folder_path, f'{i}_{self.args.local_rank}.pdf'))

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