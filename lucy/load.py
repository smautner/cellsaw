from lmz import *
import scanpy as sc
import numpy as np




def rename_obs(datasets, batch, typ):
    for ds, bat, lab in zip(datasets, batch, typ):
        if batch != 'batch':
            ds.obs['batch'] = ds.obs[bat]
        if typ != 'label':
            ds.obs['label'] = ds.obs[lab]

def load_scib(path = '/home/ubuntu/benchdata/'):
    datasets = [sc.read(path+data) for data in "Immune_ALL_hum_mou.h5ad  Immune_ALL_human.h5ad  Lung_atlas_public.h5ad  human_pancreas_norm_complexBatch.h5ad".split()]
    batch,typ = Transpose (Map(lambda x:x.split(), 'batch final_annotation#batch final_annotation#batch cell_type#tech celltype'.split("#")))
    # rename fields
    rename_obs(datasets,batch, typ)
    # split by batch
    datasets =  [[z[z.obs['batch']==i] for i in z.obs['batch'].unique()] for z
                 in datasets]
    return datasets



def load_timeseries(path = './',remove_unassigned = True):
    datasets = [sc.read(path+data+".h5ad") for data in
                "s5 509 1290 mousecortex water pancreatic cerebellum".split()]
    if remove_unassigned:
        okcells =  lambda ds: [ l not in ['-1','Unassigned','nan'] for l in ds.obs['label']]
        datasets = [ds[okcells(ds)] for ds in datasets]

    datasets =  [[z[z.obs['batch']==i] for i in z.obs['batch'].unique()]
                 for z in datasets]
    return datasets


def test_ts():
    ds = load_timeseries()
    for i,adata in enumerate(ds):
        print(i)
        print(adata)
        for a in adata:
            print(np.unique(a.obs['label']))


