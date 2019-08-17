import sys
import subprocess as sp

f=sys.argv[1]

names=['train','test','valid']

for n in names:
    sp.call('python {} {}'.format(f,n).split())
    