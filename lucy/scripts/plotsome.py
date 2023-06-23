from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import numpy as np
from lucy.scripts import wrappers
from lucy import adatas, load
import ubergauss.tools as ut


import matplotlib
# matplotlib.use('module://matplotlib-sixel')
import matplotlib.pyplot as plt


# get optimized values
def loadparams():
    results, tasks  = wrappers.loadresults('/home/ubuntu/data/yoda/8outts/')
    print(f"{ results=}")
    params = results[0]
    return params

def loaddata():
    datasets = load.load_timeseries(path = ut.fixpath(f"~/repos/cellsaw/notebooks/"))
    ssdata = [[adatas.subsample(i,1000,31443) for i in series[:10]] for series in datasets]
    ssdata = Map(adatas.preprocess, ssdata)
    return ssdata


def project2d(params, adata):
    params[f'embed_components'] = 2
    data = wrappers.dolucy(adata, **params)
    return data


def plot(data):
    datas = adatas.split_by_obs(data)
    #  plt.figure(figsize=(10,10), dpi = 300)
    adatas.plot(datas, projection = f'lsa', size = 3)

# p = loadparams()
# ssd = loaddata()
# data = project2d(p,ssd[0])



