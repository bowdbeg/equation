import pandas as pd
import matplotlib.pyplot as plt
import numpy as numpy
import sys


def csv2plot(filename, savedir, show=False,save=True):
    data = pd.read_csv(filename)
    f = filename.split('/')[-1]

    plt.plot(data['Step'],data['Value'])
    plt.xlabel('epoch')
    plt.ylabel('average error rate')
    plt.title(f)

    f = f.split('.')[0]

    if savedir[-1] != '/':
        savedir += '/'

    if save:
        plt.savefig(savedir+f+'.png')
    
    if show:
        plt.show()


if __name__ == '__main__':
    args = sys.argv
    
    if len(args) < 3:
        print('few args. min:3')
    
    filename=args[1]
    savedir=args[2]

    csv2plot(filename,savedir,show=False)
