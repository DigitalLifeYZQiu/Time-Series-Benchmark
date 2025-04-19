import argparse
import os
import random
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist

from exp.exp_forecast import Exp_Forecast
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_imputation import Exp_Imputation
from exp.exp_benchmark import Exp_Benchmark
from utils.tools import HiddenPrints

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Large Time Series Model')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',
                        help='task name, options:[forecast, imputation, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=0, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Timer',
                        help='model name, options: [Timer TrmEncoder]')
    parser.add_argument('--seed', type=int, default=0, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    parser.add_argument('--stride', type=int, default=1, help='stride')
    parser.add_argument('--ckpt_path', type=str, default='', help='ckpt file')
    parser.add_argument('--finetune_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--finetune_rate', type=float, default=0.1, help='finetune ratio')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')

    parser.add_argument('--patch_len', type=int, default=24, help='input sequence length')
    parser.add_argument('--subset_rand_ratio', type=float, default=1, help='mask ratio')
    parser.add_argument('--data_type', type=str, default='custom', help='data_type')

    parser.add_argument('--decay_fac', type=float, default=0.75)

    # cosin decay
    parser.add_argument('--cos_warm_up_steps', type=int, default=100)
    parser.add_argument('--cos_max_decay_steps', type=int, default=60000)
    parser.add_argument('--cos_max_decay_epoch', type=int, default=10)
    parser.add_argument('--cos_max', type=float, default=1e-4)
    parser.add_argument('--cos_min', type=float, default=2e-6)

    # weight decay
    parser.add_argument('--use_weight_decay', type=int, default=0, help='use_post_data')
    parser.add_argument('--weight_decay', type=float, default=0.01)

    # autoregressive configs
    parser.add_argument('--use_ims', action='store_true', help='Iterated multi-step', default=False)
    parser.add_argument('--output_len', type=int, help='output len')
    parser.add_argument('--output_len_list', type=int, nargs="+", help="output_len_list")

    # train_test
    parser.add_argument('--train_test', type=int, default=1, help='train_test')
    parser.add_argument('--is_finetuning', type=int, default=0, help='status')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    

    # imputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    # parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true", help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true", help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true", help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true", help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")
    
    augmentations = [
        'jitter', 'scaling', 'permutation', 'randompermutation', 'magwarp',
        'timewarp', 'windowslice', 'windowwarp', 'rotation', 'spawner',
        'dtwwarp', 'shapedtwwarp', 'wdba', 'discdtw', 'discsdtw'
    ]
    
    parser.add_argument('--adaptation', default=False, action="store_true", help="Add adaptation test")
    parser.add_argument('--noiseness', default=False, action="store_true", help="Add noiseness")
    parser.add_argument('--periodicity', default=False, action="store_true", help="Add periodicity")
    parser.add_argument('--distribution', default=False, action="store_true", help="Change distribution")
    parser.add_argument('--anomaly', default=False, action="store_true", help="Add anomaly")
    parser.add_argument('--adversary', default=False, action="store_true", help="Add adversary")
    
    args = parser.parse_args()
    
    if args.output_len is None:
        args.output_len = args.pred_len
    fix_seed = args.seed
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_multi_gpu:
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "64209")
        hosts = int(os.environ.get("WORLD_SIZE", "8"))  # number of nodes
        rank = int(os.environ.get("RANK", "0"))  # node id
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpus = torch.cuda.device_count()  # gpus per node
        args.local_rank = local_rank
        print(
            'ip: {}, port: {}, hosts: {}, rank: {}, local_rank: {}, gpus: {}'.format(ip, port, hosts, rank, local_rank,
                                                                                     gpus))
        dist.init_process_group(backend="nccl", init_method=f"tcp://{ip}:{port}", world_size=hosts, rank=rank)
        print('init_process_group finished')
        torch.cuda.set_device(local_rank)

    if args.task_name == 'imputation':
        Exp = Exp_Imputation
    elif args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'forecast' or args.task_name == 'long_term_forecast':
        Exp = Exp_Forecast
    elif args.task_name == 'benchmark':
        Exp = Exp_Benchmark
    else:
        raise ValueError('task name not found')

    with HiddenPrints(int(os.environ.get("LOCAL_RANK", "0"))):
        print('Args in experiment:')
        print(args)
        if args.is_finetuning or args.is_training:
            for ii in range(args.itr):
                # setting record of experiments
                setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_aug{}_'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.data,
                    args.features,
                    args.seq_len,
                    args.label_len,
                    args.pred_len,
                    args.patch_len,
                    args.d_model,
                    args.n_heads,
                    args.e_layers,
                    args.d_layers,
                    args.d_ff,
                    args.factor,
                    args.embed,
                    args.distil,
                    args.des,
                    args.augmentation_ratio,
                    ii)
                for aug in augmentations:
                    if getattr(args, aug):
                        setting += '{}_'.format(aug)
                setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")

                exp = Exp(args)  # set experiments
                print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
                exp.finetune(setting)

                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                if args.adaptation and not args.adversary:
                    exp.adaptation_test(setting)
                elif args.adaptation and args.adversary:
                    exp.adversarial_attack(setting)
                else:
                    exp.test(setting)
                torch.cuda.empty_cache()
        else:
            ii = 0
            setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}'.format(
                args.task_name,
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers,
                args.d_ff,
                args.factor,
                args.embed,
                args.distil,
                args.des,
                ii)

            setting += datetime.now().strftime("%y-%m-%d_%H-%M-%S")
            exp = Exp(args)  # set experiments
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            if args.adaptation and not args.adversary:
                exp.adaptation_test(setting)
            elif args.adaptation and args.adversary:
                exp.adversarial_attack(setting)
            else:
                exp.test(setting)
            torch.cuda.empty_cache()
