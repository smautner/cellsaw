# from lucy import load, adatas

from lmz import Map
from scalp.data.hvg import preprocess
from scalp.data.subsample import subsample, subsample_iflarger

from scalp.data import load

# def loaddata_timeseries(path,maxcells= 1000, maxdatasets = -1, **ppargs):
#     datasets = load.load_timeseries(path )
#     ssdata = subsample_preprocess(datasets, maxcells= maxcells, maxdatasets = maxdatasets, **ppargs)
#     return ssdata
# def loaddata_scib(path,maxcells= 1000, maxdatasets = -1, datasets = False, **ppargs):
#     datasets = load.load_scib(path, datasets = datasets)
#     ssdata = subsample_preprocess(datasets, maxcells= maxcells, maxdatasets = maxdatasets, **ppargs)
#     return ssdata

def loaddata_timeseries(path,datasets=False,maxcells= 1000, maxdatasets = -1, **ppargs):
    datasets = load.load_timeseries(path , datasets= datasets)
    ssdata = subsample_preprocess(datasets, maxcells= maxcells, maxdatasets = maxdatasets, **ppargs)
    return ssdata


def loaddata_scib(path, datasets = False, maxcells= 1000, maxdatasets = -1, **ppargs):
    datasets = load.load_scib(path, datasets= datasets)
    ssdata = subsample_preprocess(datasets, maxcells= maxcells, maxdatasets = maxdatasets, **ppargs)
    return ssdata



def subsample_preprocess(datasets, maxcells = 1000, maxdatasets = 10, **preprocessing_args):
    ssdata = [[subsample(i,maxcells,31443) for i in series[:maxdatasets]] for series in datasets]
    return Map(preprocess, ssdata, **preprocessing_args)


def test_load():
    from scalp import test_config
    a = loaddata_scib(test_config.scib_datapath)
    b = loaddata_timeseries(test_config.timeseries_datapath)

