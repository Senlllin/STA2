import os
import sys 

# cuda_index = '1'
cuda_index = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_index

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.append('../util')
from my_dataset import RealComGANDataset
from train_util_gan import TrainFramework

import argparse

from sta_classifier import STA
from types import SimpleNamespace


def train(args):
    train_dataset = RealComGANDataset(args.lmdb_train, args.lmdb_sn, args.input_pn, args.gt_pn, args.class_name)
    valid_dataset = RealComGANDataset(args.lmdb_valid, args.lmdb_sn, args.input_pn, args.gt_pn, args.class_name)

    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace(use_sta=bool(args.use_sta), use_sta_bias=bool(args.use_sta_bias))
    cfg.sta = SimpleNamespace(k_mirror=args.sta_k_mirror, beta=args.sta_beta, temperature=args.sta_temperature)
    cfg.loss_weights = SimpleNamespace(w_symp=args.w_symp, w_symg=args.w_symg)
    cfg.train = SimpleNamespace(sta_ramp_epochs=args.sta_ramp_epochs)

    net = STA(cfg=cfg)

    tf = TrainFramework(args.batch_size, args.log_dir, args.restore, cuda_index)
    tf._set_dataset(args.lmdb_train, args.lmdb_valid, train_dataset, valid_dataset)
    tf._set_net(net, 'STA_CLASSIFIER')
    tf._set_optimzer(args.opt, lr=args.lr, weight_decay=args.weight_decay)
    tf.train(args.max_epoch, G_opt_step=1, D_opt_step=1, save_pre_epoch=10, print_pre_step=100)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--lmdb_train', default='/home/STA2/data/RealComData/realcom_data_train.lmdb')
    parser.add_argument('--lmdb_valid', default='/home/STA2/data/RealComData/realcom_data_test.lmdb')
    parser.add_argument('--lmdb_sn', default='/home/STA2/data/RealComShapeNetData/shapenet_data.lmdb')

    parser.add_argument('--class_name', default='all', choices=['all'])

    parser.add_argument('--restore', action='store_true')
    
    parser.add_argument('--log_dir', default='log_') 

    parser.add_argument('--opt', default='Adam')
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)

    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--input_pn', type=int, default=2048)
    parser.add_argument('--gt_pn', type=int, default=2048)
    parser.add_argument('--max_epoch', type=int, default=480)

    parser.add_argument('--use_sta', type=int, default=0)
    parser.add_argument('--use_sta_bias', type=int, default=1)
    parser.add_argument('--sta_k_mirror', type=int, default=8)
    parser.add_argument('--sta_beta', type=float, default=1.5)
    parser.add_argument('--sta_temperature', type=float, default=0.07)
    parser.add_argument('--w_symp', type=float, default=0.1)
    parser.add_argument('--w_symg', type=float, default=0.1)
    parser.add_argument('--sta_ramp_epochs', type=int, default=10)
    
    args = parser.parse_args()
    args.log_dir = args.log_dir + args.class_name
    train(args)