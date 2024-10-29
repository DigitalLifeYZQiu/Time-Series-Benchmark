import logging
import os
import time
import torch
import pandas as pd
from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.split import split
from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
import argparse
import numpy as np
import torch.cuda.amp as amp
from chronos import ChronosPipeline
from exp.exp_forecast import Exp_Forecast as Exp_Timer

class Timer:
    def __init__(self, args, model_name, ckpt_path, device):
        # fix_seed = args.seed
        # random.seed(fix_seed)
        # torch.manual_seed(fix_seed)
        # np.random.seed(fix_seed)

        args.ckpt_path = ckpt_path
        # assert 'cpu' == device or 'cuda' in device
        # args.use_gpu = True if 'cuda' in device else False
        # args.gpu = device.split(':')[-1] if 'cuda' in device else 0
        print(f'args.use_gpu={args.use_gpu}, args.gpu={args.gpu}')

        self.model_name = model_name
        self.args = args
        self.exp = Exp_Timer(args)
        self.patch_len = self.args.patch_len  # 96

    def forecast(self, batch_x, pred_len):
        with torch.no_grad():
            _pred_total = self.exp.any_inference(batch_x, pred_len)  # batch,time,feature
        pred = _pred_total[:, -pred_len:, :]
        return pred


class MOIRAI:  # 速度对batch敏感，几乎成倍增加时间
    def __init__(self, args, model_name, ckpt_path, device):
        self.model_name = model_name
        self.dtype = torch.float32  # 16节省内存 32最快？
        self.device = self.choose_device(device)
        self.num_samples = 20
        self.patch_size = 64
        self.patch_len = self.patch_size  # FIXME:
        self.module = MoiraiModule.from_pretrained(ckpt_path, local_files_only=True)
        logging.info(f'num_samples={self.num_samples}, patch_size={self.patch_size}')

    def choose_device(self, device):
        if 'cpu' == device:
            return 'cpu'
        elif 'cuda' in device:
            # idx = int(device.split(':')[-1])
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            return 'cuda:0'
        else:
            raise ValueError(f'Unknown device: {device}')

    def forecast(self, data, pred_len):
        batch_size, seq_len, feature = data.shape
        assert feature == 1, f'feature={feature}'

        # real_seq_l = len(data)
        real_seq_l = seq_len
        real_pred_l = pred_len

        # 重新拼接了同一个batch内的data！！！ 由此test_data的生成方式也会变！！！
        _data = data.reshape(batch_size * real_seq_l).to('cpu')
        # seq_with_zero_pred = np.concatenate([_data, np.zeros(real_pred_l)])
        seq_with_zero_pred = torch.cat((_data, torch.zeros(real_pred_l)), 0)
        date_range = pd.date_range(start='1900-01-01', periods=len(seq_with_zero_pred), freq='s')
        data_pd = pd.DataFrame(seq_with_zero_pred, index=date_range, columns=['target'])
        ds = PandasDataset(dict(data=data_pd))
        train, test_template = split(ds, offset=real_seq_l)
        test_data = test_template.generate_instances(
            prediction_length=real_pred_l,
            windows=batch_size,
            distance=real_seq_l,
        )
        with torch.no_grad(), amp.autocast(dtype=self.dtype):  # FIXME
            # with torch.no_grad():
            predictor = MoiraiForecast(
                module=self.module,
                prediction_length=real_pred_l,
                context_length=real_seq_l,
                patch_size=self.patch_size,  # FIXME: auto
                num_samples=self.num_samples,
                target_dim=1,
                feat_dynamic_real_dim=0,
                past_feat_dynamic_real_dim=0,
            ).create_predictor(batch_size=batch_size, device=self.device)  # FIXME:batch_size=batch_size!!!
            forecasts = predictor.predict(test_data.input)
            forecast_list = list(forecasts)
        assert len(forecast_list) == batch_size, f'len(forcast_list)={len(forecast_list)}'
        preds = np.array([forecast.quantile(0.5) for forecast in forecast_list])
        preds = preds.reshape((batch_size, real_pred_l, 1))
        preds = torch.Tensor(preds)
        return preds


class Chronos:  # pred较长时时间巨长...
    def __init__(self, args, model_name, ckpt_path, device):
        if not os.path.exists(ckpt_path):
            os.makedirs(ckpt_path)
        self.model_name = model_name
        self.device = self.choose_device(device)
        self.org_device = self.device
        self.ckpt_path = ckpt_path
        self.dtype = torch.float16  # 16节省内存 32最快？
        self.pipeline = ChronosPipeline.from_pretrained(
            ckpt_path,
            device_map=self.device,
            torch_dtype=self.dtype,
        )
        # self.pipeline.model = self.pipeline.model.to(self.device)  # Ensure the model is on the correct device
        self.pipeline.model.eval()
        self.num_samples = 3
        self.patch_len = 512

    def reinit(self, device, dtype):
        self.device = self.choose_device(device)
        self.pipeline = None
        self.pipeline = ChronosPipeline.from_pretrained(
            self.ckpt_path,
            device_map=device,
            torch_dtype=dtype
        )
        self.pipeline.model.eval()

    def choose_device(self, device):
        if 'cpu' == device:
            return 'cpu'
        elif 'cuda' in device:
            # import pdb
            # pdb.set_trace()
            # idx = int(device.split(':')[-1])
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(idx)
            return 'cuda:0'
        else:
            raise ValueError(f'Unknown device: {device}')

    def forecast(self, data, pred_len):
        batch_size, seq_len, feature = data.shape
        assert feature == 1, f'feature={feature}'
        with torch.no_grad(), amp.autocast(dtype=self.dtype):  # FIXME:
            # with torch.no_grad():
            max_repeat = 5
            while max_repeat > 0:
                try:
                    if self.device != self.org_device:
                        logging.info(f'Chronos device changed, reinit...')
                        self.reinit(self.org_device, self.dtype)
                    # FIXME：既不能to dtype 也不能to device 都会报错
                    # data = torch.Tensor(data.reshape(batch_size, seq_len)).to(self.device)
                    # data = torch.Tensor(data.reshape(batch_size, seq_len)).to(self.dtype)
                    data = torch.Tensor(data.reshape(batch_size, seq_len)).to('cpu')
                    forecast = self.pipeline.predict(
                        context=data,
                        prediction_length=pred_len,
                        num_samples=self.num_samples,
                        limit_prediction_length=False,
                    )
                    
                    print(max_repeat)
                    break
                except Exception as e:
                    logging.error(e)
                    logging.info(f'Chronos predict failed, max_repeat={max_repeat}, reinit...')
                    time.sleep(3)
                    # device = 'cuda:0' if max_repeat != 1 else 'cpu'
                    # dtype = random.choice([torch.float16, torch.float32, torch.float64])
                    device, dtype = self.device, self.dtype
                    logging.info(f'device={device}, dtype={dtype}')
                    try:
                        self.reinit(device, dtype)  # 也会失败
                    except Exception as e:
                        logging.error(e)
                        logging.info(f'Chronos reinit failed, max_repeat={max_repeat}, reinit...')
                    max_repeat -= 1
                    if max_repeat == 0:
                        raise ValueError(f'Chronos predict failed, with error: {e}')
            assert forecast.shape == (batch_size, self.num_samples, pred_len), f'forecast.shape={forecast.shape}'
            preds = np.median(forecast.numpy(), axis=1).reshape((batch_size, pred_len, 1))
            preds = torch.Tensor(preds)
            return preds
        
