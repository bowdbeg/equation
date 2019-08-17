# %%
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


mode = sys.argv[1]

series = 'rnn'
prefix = 'data/{}/'.format(series)
name = '{}/{}/seq.pickle'.format(prefix,mode)
iname = '{}/{}/in.pickle'.format(prefix, mode)
oname = '{}/{}/out.pickle'.format(prefix, mode)
mname = '{}/{}/mask.pickle'.format(prefix, mode)
sname = '{}/{}/swa.pickle'.format(prefix, mode)

if mode == 'train':
    m = Mode.train
elif mode == 'test':
    m = Mode.test
elif mode == 'valid':
    m = Mode.train
else:
    exit(1)

# load data
data = DL.DataLoader(m)
print(data.keys())
# for validation data
if mode == 'train':
    for k in data.keys():
        data[k] = data[k][76:]
elif mode == 'valid':
    for k in data.keys():
        data[k] = data[k][:76]


# get time data
output = data.pop('output')
input = data.pop('input')
swa = data.pop('SWA')
print(np.array(swa).shape)

datasize = len(swa)
seq_len=len(swa[0])
# input_test = test.pop('input')

# delete 0 after finesh #######
# search end of simulation
# ends = []
# for i, al in enumerate(data['SWA']):
#     for j, dat in enumerate(reversed(al)):
#         if dat != 0:
#             if j == 0:
#                 j = len(al)-1
#             ends.append(j)
#             break

# minimum zero and nan
ends = []
for i in range(datasize):
    m = 0
    for k in data.keys():
        for j, d in enumerate(reversed(data[k][i])):
            if not np.isnan(d):
                if d != 0:
                    if j == 0:
                        m = len(data[k])-1
                    else:
                        m = max(m, j)
                    break
    ends.append(m)

# print(ends)

# %%
# make mask
mask = np.zeros((len(ends), len(data.keys()), len(swa[0])))

# print(mask)
# print(mask.shape)

for i, e in enumerate(ends):
    mask[i, :, :e] = 1

# print(mask)
# print(mask.shape)

# %%
# fill to zero from nan
for i,key in enumerate(data.keys()):
    for j in range(datasize):
        for k in range(seq_len):
            if np.isnan(data[key][j][k]):
                data[key][j][k]=0

##############################

#%%
seq = []
for i in range(len(input)):
    d = []
    for k in data.keys():
        d.append(data[k][i])
    seq.append(d)

#%%
if save_flag:
    with open(name, mode='wb') as f:
        pkl.dump(seq, f)
    with open(iname, mode='wb') as f:
        pkl.dump(input, f)
    with open(oname, mode='wb') as f:
        pkl.dump(output, f)
    with open(mname, mode='wb') as f:
        pkl.dump(mask, f)
    with open(sname, mode='wb') as f:
        pkl.dump(swa, f)
