import pandas as pd
import numpy as np
import torch
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from define import Mode

train_dir='./data/train/'
test_dir='./data/test/'

class DataScaler():
    def __init__(self,train=None):
        if train is not None:
            self.train=train
            self.fit(train)

    def fit(self,train):
        pass

    def norm(self,x):
        pass
    
    def denorm(self,x):
        pass




def DataLoader(mode=Mode.train):

    if not mode in Mode:
        exit(1)

    if mode == Mode.train:
        dir = train_dir
    elif mode == Mode.test:
        dir = test_dir

    data = {}

    dirs = os.listdir(dir)
    for d in dirs:
        with open(dir+d, 'rb') as f:
            dat = pickle.load(f)

        name = d.split('.')[0]
        data.update({name: dat})

    return data


# not recommend
def DataLoaderFromFile(is_tensor=True):
    path = './data/raw/'

    fileName = [
        'input', 'output',
        'SWA',
        'x', 'y', 'z',
        'rx', 'ry', 'rz',
        'Vx', 'Vy', 'Vz',
        'Vrx', 'Vry', 'Vrz',
        'Ax', 'Ay', 'Az',
        'Arx', 'Ary', 'Arz',
        'beta',
        'FyFL', 'FyFR',
        'FyRL', 'FyRR',
        'FzFL', 'FzFR',
        'FzRL', 'FzRR',
        's_FL', 's_FR',
        's_RL', 's_RR'
    ]

    data = {}

    for n in fileName:
        fname = path + n + '.csv'
        d = np.delete(np.double(pd.read_csv(
            fname, header=None).values[1:]), 0, axis=1)
        data.update({n: d})
    if is_tensor:
        Data2Tensor(data)
    return data


# not recomend
def Data2Tensor(data):
    for key in data.keys():
        data[key] = torch.from_numpy(data[key]).float()


# not recomend
def DataLoaderSplit(testsize=0.3, random=0, is_tensor=True):
    data = DataLoaderFromFile(is_tensor=False)

    train = {}
    test = {}

    for key in data.keys():
        tr, te = train_test_split(
            data[key], test_size=testsize, random_state=random)
        train.update({key: tr})
        test.update({key: te})

    if is_tensor:
        Data2Tensor(train)
        Data2Tensor(test)

    return train, test
