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

save_flag = True

mode = 'train'

if mode == 'train':
    name = 'data/train_time/'
    m = Mode.train
elif mode == 'test':
    name = 'data/test_time/'
    m = Mode.test
elif mode == 'valid':
    name = 'data/valid_time/'
    m=Mode.train
else:
    exit(1)

iname = name+'input.pickle'
oname = name+'output.pickle'

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

# minimum zero
ends = []
for i in range(len(train['SWA'])):
    m = 0
    for k in train.keys():
        for j, d in enumerate(reversed(train[k][i])):
            if not np.isnan(d):
                if d != 0:
                    if j==0:
                        m=len(train[k])-1
                    else:
                        m = max(m, j)
                    break
    ends.append(m)

# slice to delete zero
for i, k in enumerate(train.keys()):
    for j, e in enumerate(ends):
        train[k][j] = train[k][j][:-e]

##############################


ts = []
for i in range(len(input_train)):
    for j, k in enumerate(train.keys()):
        tar = torch.tensor(train[k][i])
        if j == 0:
            ts.append(tar.reshape(1, len(tar)))
        else:
            ts[i] = torch.cat((ts[i], tar.reshape(1, len(tar))), dim=0)
for i, t in enumerate(ts):
    if len(t.size()) != 2:
        print(t)
    for j in range(t.size(1)-1):
        x = torch.tensor(input_train[i])
        x = torch.cat((x, t[:, j]))
        y = t[:, j+1]
        if i == 0 and j == 0:
            X_train = x.reshape(1, len(x))
            y_train = y.reshape(1, len(y))
        else:
            X_train = torch.cat((X_train, x.reshape(1, len(x))))
            y_train = torch.cat((y_train, y.reshape(1, len(y))))
    print(i)

if save_flag:
    with open(iname, mode='wb') as f:
        pkl.dump(X_train, f)

    with open(oname, mode='wb') as f:
        pkl.dump(y_train, f)
