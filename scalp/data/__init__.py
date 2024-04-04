# from lucy import load, adatas

from lmz import Map,Range
from scalp.data.hvg import preprocess
from scalp.data.subsample import subsample, subsample_iflarger
import scanpy as sc
from scalp.data import load
import numpy as np

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

from sklearn.datasets import make_blobs
def create_anndata(matrix, labels, batch_string):
    """
    Create an AnnData object with specified observations and batch information.

    Parameters:
    - matrix (numpy.ndarray): The data matrix.
    - labels (list): List of labels corresponding to the observations.
    - batch_string (str): The batch string to be repeated for all observations.

    Returns:
    - anndata.AnnData: An AnnData object.
    """
    adata = sc.AnnData(X=matrix)
    adata.obs['label'] = labels
    adata.obs['batch'] = np.repeat(batch_string, len(labels))
    adata.obsm['umap'] = matrix
    return adata

def mkblobs(sizes = [[20,25],[25,20],[23,22]], clusterspread = .2 , batchspread = 1):
    res = []
    for batchid, slist in enumerate(sizes):
        centers = np.array([[sid, batchid * batchspread] for sid in Range(slist)] )
        X,y = make_blobs(n_samples= slist, centers = centers, cluster_std = clusterspread)
        res.append(create_anndata(X,y, batchid))

    return res

