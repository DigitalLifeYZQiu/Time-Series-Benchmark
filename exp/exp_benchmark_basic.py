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

from import_model import Chronos, MOIRAI, Timer

class Exp_Benchmark_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'Chronos': Chronos,
            'Moirai': MOIRAI,
            'Timer': Timer,
        }
        if self.args.use_multi_gpu:
            self.model = self._build_model()
            self.device = torch.device('cuda:{}'.format(self.args.local_rank))
        else:
            self.device = self._acquire_device()
            self.model = self._build_model()
    
    def _build_model(self):
        raise NotImplementedError
        return None

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _get_data(self, flag):
        pass
    
    def vali(self, setting):
        pass

    def finetune(self, setting):
        pass

    def test(self, setting, test=0):
        pass
    