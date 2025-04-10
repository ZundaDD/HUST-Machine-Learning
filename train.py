import numpy as np
import torch
import random
import argparse
from learner import Learner

# python train.py --seed 4 --model resnet50 --num_steps 100
# nohup python -u train.py --seed 4 --model resnet50 > exp4.log 2>&1 &
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # editable
    parser.add_argument("--model", default='resnet18', type=str)
    parser.add_argument("--augment", action='store_true')
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)

    # default
    parser.add_argument("--num_workers", default=16, type=int)
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--num_steps", default= 25000, type=int)

    # environment
    parser.add_argument("--log_dir", default='./result', type=str)
    parser.add_argument("--log_freq", default=5, type=int)
    parser.add_argument("--data_dir", default='./dataset', type=str)
    parser.add_argument("--calculate_freq", default=100, type=int)

    args = parser.parse_args()

    random.seed(args.seed)
    print('Training starts ...')
    learner = Learner(args)
    learner.train(args)
