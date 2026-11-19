import argparse
import os
import random
import sys

import numpy as np
import torch

from exp.exp_main import Exp_Main


class Tee:
    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        return len(message)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()

    def fileno(self):
        return self.terminal.fileno()

    @property
    def encoding(self):
        return self.terminal.encoding


DEFAULT_FLINEAR_CONFIG = {
    'alpha': 0.90,
    'c': 1.00,
    'groups': 8,
}


def resolve_flinear_config(args):
    if args.flinear_alpha is None:
        args.flinear_alpha = DEFAULT_FLINEAR_CONFIG['alpha']
    if args.flinear_c is None:
        args.flinear_c = DEFAULT_FLINEAR_CONFIG['c']

    if not 0.0 < args.flinear_alpha <= 1.0:
        raise ValueError('flinear_alpha must be in the interval (0, 1].')
    if args.flinear_c <= 0.0:
        raise ValueError('flinear_c must be greater than 0.')
    if args.flinear_groups <= 0:
        raise ValueError('flinear_groups must be a positive integer.')

    return args



parser = argparse.ArgumentParser(
    description='FTQNet for time series forecasting')

# Reproducibility
parser.add_argument('--random_seed', type=int, default=2024,
                    help='random seed')

# Basic configuration
parser.add_argument('--is_training', type=int, required=True,
                    help='1: train and test; 0: test only')
parser.add_argument('--model_id', type=str, required=True,
                    help='experiment id')
parser.add_argument('--model', type=str, default='FTQNet', choices=['FTQNet'],
                    help='only FTQNet is retained')
parser.add_argument('--des', type=str, default='test',
                    help='kept for compatibility with existing FTQNet scripts')

# Data loader
parser.add_argument('--data', type=str, required=True, default='ETTh1',
                    choices=['ETTh1', 'ETTh2', 'ETTm1', 'ETTm2',
                             'custom', 'Solar', 'PEMS', 'ILI', 'Covid-19', 'Wearable',
                             'Chickenpox-Hungary', 'Dengue-Colombia',
                             'Influenza-US-2', 'Dengue-Argentina', 'Influenza-US',
                             'M3-stool', 'A-stool', 'A-saliva', 'F4-stool'],
                    help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset/',
                    help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv',
                    help='data filename')
parser.add_argument('--features', type=str, default='M',
                    choices=['M', 'S', 'MS'],
                    help='M: multivariate-to-multivariate; '
                         'S: univariate-to-univariate; '
                         'MS: multivariate-to-univariate')
parser.add_argument('--target', type=str, default='OT',
                    help='target column for S or MS tasks')
parser.add_argument('--freq', type=str, default='h',
                    help='time-feature frequency')
parser.add_argument('--embed', type=str, default='timeF',
                    choices=['timeF', 'fixed', 'learned'],
                    help='time encoding used by the data loader')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                    help='checkpoint directory')

# Forecasting task
parser.add_argument('--seq_len', type=int, default=96,
                    help='input sequence length')
parser.add_argument('--label_len', type=int, default=0,
                    help='label length used by the dataset')
parser.add_argument('--pred_len', type=int, default=96,
                    help='prediction length')

# FTQNet
parser.add_argument('--cycle', type=int, default=24,
                    help='cycle length of the temporal query')
parser.add_argument('--use_revin', type=int, default=1, choices=[0, 1],
                    help='use instance normalization')
parser.add_argument('--enc_in', type=int, default=7,
                    help='number of input channels')
parser.add_argument('--d_model', type=int, default=512,
                    help='hidden dimension')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='output dropout')
parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp'],
                    help='kept for compatibility with existing FTQNet scripts')
parser.add_argument('--flinear_alpha', type=float, default=None,
                    help='FLinear fractional order')
parser.add_argument('--flinear_c', type=float, default=None,
                    help='FLinear RMS clipping threshold')
parser.add_argument('--flinear_groups', type=int,
                    default=DEFAULT_FLINEAR_CONFIG['groups'],
                    help='number of FLinear output groups')

# Optimization
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--itr', type=int, default=1,
                    help='number of experiment repetitions')
parser.add_argument('--train_epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'mae'],
                    help='mse for the original 12 and Wearable, mae for ILI/COVID-19')
parser.add_argument('--lradj', type=str, default='type3')
parser.add_argument('--pct_start', type=float, default=0.3)
parser.add_argument('--use_amp', action='store_true', default=False)

# Hardware and diagnostics
parser.add_argument('--use_gpu', type=bool, default=True)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--use_multi_gpu', action='store_true', default=False)
parser.add_argument('--devices', type=str, default='0,1')
parser.add_argument('--test_flop', action='store_true', default=False)

# The original 12 datasets retain every original argument and training path.
# These options are only used by the three added benchmark adapters.
parser.add_argument('--health_strict_schema', type=int, choices=[0, 1], default=1)
parser.add_argument('--health_train_drop_last', type=int, choices=[0, 1], default=1)
parser.add_argument('--health_deterministic', type=int, choices=[0, 1], default=1)
parser.add_argument('--covid_calibrated', action='store_true',
                    help='validation-only COVID-19 calibration for H=24,36,48,60')

args = resolve_flinear_config(parser.parse_args())

if args.covid_calibrated and (args.data != 'Covid-19' or args.pred_len not in (24, 36, 48, 60)):
    parser.error('--covid_calibrated requires Covid-19 with H=24,36,48,60')
if args.data == 'Covid-19' and not args.covid_calibrated:
    parser.error('COVID-19 uses the scripted calibrated flow; include --covid_calibrated')
if args.data in ('ILI', 'Covid-19'):
    if args.loss != 'mae':
        parser.error('ILI and Covid-19 expect --loss mae')
    from exp.exp_health import Exp_Health as Experiment
    if args.health_deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
elif args.data in ('Chickenpox-Hungary', 'Dengue-Colombia', 'Dengue-Argentina',
                   'Influenza-US-2', 'Influenza-US'):
    from exp.exp_epicast import get_epicast_experiment
    Experiment = get_epicast_experiment(args.data)
    if not torch.cuda.is_available():
        torch.set_num_threads(1)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
elif args.data in ('M3-stool', 'A-stool', 'A-saliva', 'F4-stool'):
    if args.loss != 'mae':
        parser.error('Paper-aligned microbiome tasks require --loss mae')
    from data_provider.data_microbiome import microbiome_feature_count
    from exp.exp_microbiome import Exp_Microbiome as Experiment
    args.enc_in = microbiome_feature_count(args.root_path, args.data_path, args.data)
elif args.data == 'Wearable':
    if args.loss != 'mse':
        parser.error('Wearable uses --loss mse')
    from data_provider.data_wearable import wearable_feature_names
    args.enc_in = len(wearable_feature_names(os.path.join(args.root_path,args.data_path), args.target))
    from exp.exp_wearable import Exp_Wearable as Experiment
else:
    if args.loss != 'mse':
        parser.error('The original 12 datasets use --loss mse')
    Experiment = Exp_Main


os.makedirs('./results', exist_ok=True)
dataset_name = args.model_id.split('_', 1)[0]

log_file = open(
    os.path.join('./results', dataset_name + '.txt'),
    'a',
    buffering=1,
)

sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ', '')
    args.device_ids = [int(device) for device in args.devices.split(',')]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)
print('FLinear config | alpha: {}, c: {}, groups: {}'.format(
    args.flinear_alpha, args.flinear_c, args.flinear_groups))


def make_setting(config):
    return '{}_{}_{}_ft{}_sl{}_pl{}_cycle{}_fa{}_fc{}_fg{}_seed{}'.format(
        config.model_id,
        config.model,
        config.data,
        config.features,
        config.seq_len,
        config.pred_len,
        config.cycle,
        config.flinear_alpha,
        config.flinear_c,
        config.flinear_groups,
        config.random_seed,
    )


setting = make_setting(args)

if args.is_training:
    for _ in range(args.itr):
        exp = Experiment(args)
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(
            setting))
        exp.train(setting)
        if args.data == 'Covid-19':
            print('>>>>>>>fitting COVID calibration on validation: {}<<<<<<<<'.format(setting))
            exp.fit_covid_calibration(setting)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(
            setting))
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    exp = Experiment(args)
    if args.data == 'Covid-19':
        # Load this horizon's best weights first, then fit rules on validation.
        exp.fit_covid_calibration(setting, load_best_checkpoint=True)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
    else:
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
    torch.cuda.empty_cache()
