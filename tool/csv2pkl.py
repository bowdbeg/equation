import os
import pickle
import pandas as pd
import numpy as np

spath = '../pkl/'
dir = os.listdir()

for d in dir:
    a = np.double(pd.read_csv(d))
    a = np.delete(a, 0, axis=1)
    name = d.split('.')[0]
    with open(spath+name+'.pickle', 'wb') as f:
        pickle.dump(a.tolist(), f)
