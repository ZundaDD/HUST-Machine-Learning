import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import itertools


exps = {'model':[], 'batch':[], 'lr':[], 'pretrain':[], 'augment':[], 'best':[]}
seed = [0]
model = ["CNN", "resnet18", "resnet34", "resnet50", "vgg11_bn"]
batch_size = [64, 256]
lr = [1e-3, 1e-4]
resnet_pretrain = [False, True]
data_aug = [False, True]


all_experiments = list(itertools.product(seed, model, batch_size, lr, resnet_pretrain, data_aug))
for idx, exp in enumerate(all_experiments):
    df = pd.read_csv(f'result/{idx}/0_result.csv')
    exps['model'].append(exp[1])
    exps['batch'].append(exp[2])
    exps['lr'].append(exp[3])
    exps['pretrain'].append(exp[4])
    exps['augment'].append(exp[5])
    exps['best'].append(df['test'].max())

df = pd.DataFrame(exps)
df.to_csv(f'analysis.csv', index=False)