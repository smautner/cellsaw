from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import numpy as np
from lucy.scripts import wrappers
from lucy import adatas, load
import ubergauss.tools as ut


import matplotlib
# matplotlib.use('module://matplotlib-sixel')
import matplotlib.pyplot as plt


# get optimized values
def loadparams_timeseries_optimization():
    results, tasks  = wrappers.loadresults('/home/ubuntu/data/yoda/8outts/')
    for r,taskitem in zip(results, tasks):
        for label, thing in zip('optigoal train test'.split(), taskitem):
            r[label] = thing
    return results

def loadparams_scib_optimization():
    results, tasks  = wrappers.loadresults('/home/ubuntu/data/yoda/8outscib/')
    for r,taskitem in zip(results, tasks):
        for label, thing in zip('optigoal train test'.split(), taskitem):
            r[label] = thing
    return results


def loaddata_timeseries():
    datasets = load.load_timeseries(path = ut.fixpath(f"~/repos/cellsaw/notebooks/"))
    ssdata = [[adatas.subsample(i,1000,31443) for i in series[:10]] for series in datasets]
    ssdata = Map(adatas.preprocess, ssdata)
    return ssdata

def loaddata_scib():
    datasets = load.load_scib()
    ssdata = [[adatas.subsample(i,1000,31443) for i in series[:10]] for series in datasets]
    ssdata = Map(adatas.preprocess, ssdata)
    return ssdata

def project2d( adatas, params):
    params[f'embed_components'] = 2
    data = wrappers.dolucy(adatas, **params)
    return data


def plot(data):
    datas = adatas.split_by_obs(data)
    #  plt.figure(figsize=(10,10), dpi = 300)
    adatas.plot(datas, projection = f'lsa', size = 3)


def test_prep_diffusion():
    p = loadparams_scib_optimization()
    ssd = loaddata_scib()
    data = wrappers.dolucy(ssd[0][:2], **p[0])
    return data


def test_embeddin(data):
    m = data
    return adatas.graph_embed2(m[1][0].todense(),m[0])

def test_embedding():
    data = test_prep_diffusion()





