import argparse
import logging

def str2bool(v):
    return v.lower() in ("true", "1")

arg_lists = []
parser = argparse.ArgumentParser(
    description="Sub-Pixel Accurate Keypoint Refinement.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("experiment", type=str)

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

common_arg = add_argument_group("Common")
common_arg.add_argument('--fmat', '-fmat', action='store_true', 
    help='estimate the fundamental matrix, instead of the essential matrix')

common_arg.add_argument('--datasets', '-ds', 
    default='megadepth',
    help='which datasets to use, separate multiple datasets by comma')

common_arg.add_argument('--nfeatures', '-nf', type=int, default=2048, 
    help='fixes number of features by clamping/replicating, set to -1 for dynamic feature count but then batchsize (-bs) has to be set to 1')

common_arg.add_argument('--sideinfo', action='store_true', 
    help='Do not provide side information (matching ratios) to the network. The network should be trained and tested consistently.')

common_arg.add_argument('--detect_anomaly', '-da', action='store_true', 
    help='Anomaly detection of PyTorch')

common_arg.add_argument('--ransac_thr', '-rt', type=float, default=1.0, 
    help='GCRANSAC inlier threshold. Recommended value is 1.0px')

common_arg.add_argument('--train_thr', '-tt', type=float, default=1.5, 
    help='Train inlier threshold. Recommended value is 1.5px')

common_arg.add_argument('--model', '-m', default='', 
    help='load a model to contuinue training or leave empty to create a new model')

common_arg.add_argument('--detector', '-detc', choices=['spnn', 'splg', 'aliked', 'dedode', 'xfeat'], default='sp', 
    help='Type of detector; "sp" means SuperPoint, "aliked" means ALIKED')

# Training parameters
train_arg = add_argument_group("Training")
train_arg.add_argument('--resume', action='store_true', 
    help='Resume from existing experiment')

train_arg.add_argument('--batchsize', '-bs', type=int, default=8,
    help='batch size')

train_arg.add_argument('--learning_rate', '-lr', type=float, default=0.0001, 
    help='learning rate')

train_arg.add_argument('--weight_decay', '-wd', type=float, default=0.0, 
    help='weight decay')

train_arg.add_argument('--train_iter', '-it', type=int, default=200000,
    help='number of iterations to train')

train_arg.add_argument('--save_intv', type=int, default=500,
    help='model saving interval')

train_arg.add_argument('--val_intv', type=int, default=5000,
    help='evaluation interval')

# Testing parameters
test_arg = add_argument_group("Testing")
test_arg.add_argument('--total_run', '-tr', type=int, default=10,
    help='Total number of runs on validation set')

test_arg.add_argument('--test', action='store_true', 
    help='Testing mode')

test_arg.add_argument('--total_split', '-ts', type=int, default=1,
    help='total split size')

test_arg.add_argument('--current_split', '-cs', type=int, default=0,
    help='current split')

test_arg.add_argument('--vanilla', action='store_true',
    help='Run vanilla pose estimation pipeline (without K2S model)')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def print_usage():
    parser.print_usage()

def get_logger(args):
    formatter = logging.Formatter(
        fmt="[%(asctime)s %(name)s %(levelname)s] %(message)s", datefmt="%m/%d/%Y %H:%M:%S"
    )
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)

    logger = logging.getLogger(f"K2S")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False

    __module_name__ = f"K2S"
    return logger
