import random
import pickle
import os
import sys
# exe in data

NUM_SAMPLE = 500


args = sys.argv

idir = args[1]  # pkl
odir = args[2]  # train
edir = args[3]  # test

dirs = os.listdir(idir)


for dir in dirs:
    with open(idir+dir, 'rb') as f:
        d = pickle.load(f)
        
    # defime sample index
    sample = list(range(0,len(d),10))

    exsample = list(range(len(d)))
    for s in sample:
        exsample.remove(s)

    name = dir.split('.')[0]

    l=[]
    el=[]

    for s in sample:
        l.append(d[s])
    
    for s in exsample:
        el.append(d[s])

    with open(odir+name+'.pickle', 'wb') as f:
        pickle.dump(l,f)
    
    with open(edir+name+'.pickle', 'wb') as f:
        pickle.dump(el,f)

