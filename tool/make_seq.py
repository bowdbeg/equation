import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.optim as optim
import matplotlib.pyplot as plt
import tensorboardX as tbx
import time
import os
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from statistics import mean
import pickle as pkl
import dataloader as DL
from define import Mode

save_flag = False

mode = 'valid'

if mode == 'train':
    name = 'data/seq/train.pickle'
    iname = 'data/seq/train_in.pickle'
    oname = 'data/seq/train_out.pickle'
    m = Mode.train
elif mode == 'test':
    name = 'data/seq/test.pickle'
    iname = 'data/seq/test_in.pickle'
    oname = 'data/seq/test_out.pickle'
    m = Mode.test
elif mode == 'valid':
    name = 'data/seq/valid.pickle'
    iname = 'data/seq/valid_in.pickle'
    oname = 'data/seq/valid_out.pickle'
    m = Mode.train
else:
    exit(1)


# load data

train = DL.DataLoader(m)
# test = DL.DataLoader(Mode.test)

# for validation data
if mode == 'train':
    for k in train.keys():
        train[k] = train[k][76:]
elif mode == 'valid':
    for k in train.keys():
        train[k] = train[k][:76]


# get time data
output_train = train.pop('output')
# output_test = test.pop('output')

input_train = train.pop('input')

# input_test = test.pop('input')

# delete 0 after finesh #######
# search end of simulation
# ends = []
# for i, al in enumerate(train['SWA']):
#     for j, dat in enumerate(reversed(al)):
#         if dat != 0:
#             if j == 0:
#                 j = len(al)-1
#             ends.append(j)
#             break

# minimum zero and nan
ends = []
for i in range(len(train['SWA'])):
    m = 0
    for k in train.keys():
        for j, d in enumerate(reversed(train[k][i])):
            if not np.isnan(d):
                if d != 0:
                    if j == 0:
                        m = len(train[k])-1
                    else:
                        m = max(m, j)
                    break
    ends.append(m)

# slice to delete zero and nan
for i, k in enumerate(train.keys()):
    for j, e in enumerate(ends):
        train[k][j] = train[k][j][:-e]

##############################

seq = []
for i in range(len(input_train)):
    d = []
    for k in train.keys():
        d.append(train[k][i])
    seq.append(d)

if save_flag:
    with open(name, mode='wb') as f:
        pkl.dump(seq, f)
    with open(iname, mode='wb') as f:
        pkl.dump(input_train, f)
    with open(oname, mode='wb') as f:
        pkl.dump(output_train, f)
