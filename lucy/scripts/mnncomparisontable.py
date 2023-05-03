import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
from lmz import *
import ubergauss.tools as ut
import wrappers as nu
from lucy import load, adatas

# datasets = load.load_scib() + load.load_timeseries(path= '/home/ubuntu/repos/cellsaw/notebooks/')
# ssdata = [[adatas.subsample(i,1000,31442)  for i in series[:10]]  for series in datasets]
# ssdata = Map(adatas.preprocess, ssdata)


# TODO  decide if i want to remove these unnecassary wrappers
def runmnn(data):
    data = nu.domnn(data)
    return data
def runlucy(data):
    data = nu.dolucy(data)
    return data




def ff(x):
    task, f = x
    ds = ssdata[task]
    data = [z.copy() for z in ds]
    return f(ds)

# tasks = [( task, alg) for task in Range(ssdata) for alg in [runmnn, runlucy]]
tasks = [( task, alg) for task in Range(10) for alg in [runmnn, runlucy]]


# adata with label batch and lsa annotation
#ppdata = ut.xmap( ff, tasks) ; ut.dumpfile(ppdata, f'datadelme.dmp')

# ppdata = ut.loadfile(f'datadelme.dmp')
print("loaded")

# scores = Map(nu.scores, ppdata)
print("scored")

# while scores:
#     a = scores.pop()
#     b = scores.pop()

scores = ut.loadfile('scores.delme')


result = []
snames = 'label shilouette batchmix'.split(' ')
for (ds,algo),score in zip(tasks,scores):
    if algo == runmnn:
        aname = 'mnn'
    else:
        aname = 'lucy'

    for value,scorename in zip(score,snames):
            result.append({"dataset":ds , "algo":aname, 'test':scorename, 'score': value})




import pandas as pd
print(pd.DataFrame(result))
print (result)

def maketable(mix, shape):
for score_mix in [.4,.6,.8]:
    for score_shape in [2,4,6,8]:
        make_table (score_mix, score_shape )

