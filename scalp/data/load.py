from lmz import *
import scanpy as sc
import numpy as np
import anndata as ann

def rename_obs(datasets, batch, typ):
    for ds, bat, lab in zip(datasets, batch, typ):
        if batch != 'batch':
            ds.obs['batch'] = ds.obs[bat]
        if typ != 'label':
            ds.obs['label'] = ds.obs[lab]

def load_scib(path, datasets = False):
    if not datasets:
        datasets = "Immune_ALL_hum_mou Immune_ALL_human Lung_atlas_public human_pancreas_norm_complexBatch".split()

    #datasets = [sc.read_h5ad(path+data+".h5ad") for data in datasets]
    datasets = [ann.read_h5ad(path+data+".h5ad") for data in datasets]
    batch,typ = Transpose (Map(lambda x:x.split(), 'batch final_annotation#batch final_annotation#batch cell_type#tech celltype'.split("#")))
    # rename fields
    rename_obs(datasets,batch, typ)
    # split by batch, added the copy to avoid the warning message for initialozing view
    datasets =  [[z[z.obs['batch']==i].copy() for i in z.obs['batch'].unique()] for z
                 in datasets]

    for batchlist in datasets:
        for batch in batchlist:
            batch.uns['timeseries'] = False
    return datasets

def load_timeseries(path,datasets=False,remove_unassigned = True):

    if not datasets:
        datasets = "s5 509 1290 mousecortex water pancreatic cerebellum".split()
    datasets = [sc.read_h5ad(path+data+".h5ad") for data in datasets]

    # datasets = [sc.read(path+data+".h5ad") for data in "s5 509 1290 mousecortex water pancreatic cerebellum".split()]

    if remove_unassigned:
        okcells =  lambda ds: [ l not in ['-1','Unassigned','nan'] for l in ds.obs['label']]
        datasets = [ds[okcells(ds)] for ds in datasets]

    datasets =  [[z[z.obs['batch']==i] for i in z.obs['batch'].unique()]
                 for z in datasets]
    for batchlist in datasets:
        for batch in batchlist:
            batch.uns['timeseries'] = True
    return datasets


def test_ts():
    ds = load_timeseries()
    for i,adata in enumerate(ds):
        print(i)
        print(adata)
        for a in adata:
            print(np.unique(a.obs['label']))


