from __future__ import print_function
import argparse
import torch
#from solver_noise import Solver_Noise
from solver_noise_mdd import Solver_Noise
#from solver import Solver_Noise
import os
import numpy as np

# Training settings
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser(description='PyTorch MCD-resnet Implementation')
parser.add_argument('--all_use', type=str, default='no', metavar='N',
                    help='use all training data? in usps adaptation')
parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoint', metavar='N',
                    help='source only or not')
parser.add_argument('--eval_only', action='store_true', default=False,
                    help='evaluation only option')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--max_epoch', type=int, default=110, metavar='N',
                    help='how many epochs')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num_k', type=int, default=4, metavar='N',
                    help='hyper paremeter for generator update')
parser.add_argument('--one_step', action='store_true', default=False,
                    help='one step training with gradient reversal layer')
parser.add_argument('--optimizer', type=str, default='adam', metavar='N', help='which optimizer')
parser.add_argument('--resume_epoch', type=int, default=100, metavar='N',
                    help='epoch to resume')
parser.add_argument('--save_epoch', type=int, default=10, metavar='N',
                    help='when to restore the model')
parser.add_argument('--save_model', action='store_true', default=False,
                    help='save_model or not')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--source', type=str, default='huashan', metavar='N',
                    help='source dataset')
parser.add_argument('--target', type=str, default='renji', metavar='N', help='target dataset')
parser.add_argument('--use_abs_diff', action='store_true', default=False,
                    help='use absolute difference value as a measurement')
parser.add_argument('--dataset_folder', type=str, default='/storage1/21721505/data/', help='where your dataset stored')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
print(args)


def main():
    # if not args.one_step:

    if not os.path.exists(args.dataset_folder):
        os.makedirs(args.dataset_folder)
    if not os.path.exists(args.dataset_folder+'record'):
        os.makedirs(args.dataset_folder+'record')

    ys_file = args.dataset_folder + "ys.npy"
    yt_file = args.dataset_folder + "yt.npy"
    if os.path.isfile(ys_file):
            ys = np.load(ys_file)
    else:
            ys = []
    if os.path.isfile(yt_file):
            yt = np.load(yt_file)
    else:
            yt = []

    solver = Solver_Noise(args, source=args.source, target=args.target, learning_rate=args.lr, batch_size=args.batch_size,
                    optimizer=args.optimizer, num_k=args.num_k, all_use=args.all_use,
                    checkpoint_dir=args.checkpoint_dir,
                    save_epoch=args.save_epoch, datafolder = args.dataset_folder, ys=ys,yt=yt)
    record_num = 0
    record_train = 'record/%s_%s_k_%s_%s_%s.txt' % (
        args.source, args.target, args.num_k, args.one_step, record_num)
    record_test = 'record/%s_%s_k_%s_%s_%s_test.txt' % (
        args.source, args.target, args.num_k, args.one_step, record_num)
    while os.path.exists(record_train):
        record_num += 1
        record_train = 'record/%s_%s_k_%s_%s_%s.txt' % (
            args.source, args.target, args.num_k, args.one_step, record_num)
        record_test = 'record/%s_%s_k_%s_%s_%s_test.txt' % (
            args.source, args.target, args.num_k, args.one_step, record_num)

    if not os.path.exists(args.checkpoint_dir):
        os.mkdir(args.checkpoint_dir)
    if not os.path.exists('record'):
        os.mkdir('record')
    if args.eval_only:
        solver.test(0)
    else:
        count = 0
        for t in range(args.max_epoch):
            num = solver.train(t, record_file=record_train)
            count += num
            if t % 1 == 0:
                solver.test(t, record_file=record_test, save_model=args.save_model)
            if count >= 20000:
                break


if __name__ == '__main__':
    main()
