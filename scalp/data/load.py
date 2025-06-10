from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import scanpy as sc
import numpy as np
import anndata as ann


def rename_obs(datasets, batch, typ):
    '''
    rename the fields in the datasets to batch and label
    '''
    for ds, bat, lab in zip(datasets, batch, typ):
        if batch != 'batch':
            ds.obs['batch'] = ds.obs[bat]
        if typ != 'label':
            ds.obs['label'] = ds.obs[lab]

def load_scib(path, datasets = False):
    alldatasets = "Immune_ALL_hum_mou Immune_ALL_human Lung_atlas_public human_pancreas_norm_complexBatch".split()

    if not datasets:
        dataset_names = alldatasets
    else:
        dataset_names = datasets

    #datasets = [sc.read_h5ad(path+data+".h5ad") for data in datasets]
    datasets = [ann.read_h5ad(path+data+".h5ad") for data in dataset_names]


    # labels and batchlabels are wrong in the original data, so we fix it here
    # select batch and label for the selected datasets
    labels_per_dataset = dict(zip(alldatasets, 'batch final_annotation#batch final_annotation#batch cell_type#tech celltype'.split('#')))
    new_names = [labels_per_dataset[data] for data in dataset_names]
    batch,typ = Transpose (Map(lambda x:x.split(), new_names))
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
        #datasets = "s5 509 1290 mousecortex water pancreatic cerebellum done_bone_marrow done_lung done_pancreas done_reprogramming_morris done_reprogramming_schiebinger".split()
        datasets = "s5 509 1290 mousecortex water pancreatic cerebellum done_bone_marrow done_lung done_pancreas done_reprogramming_morris".split()

    datasetsd = [sc.read_h5ad(path+data+".h5ad") for data in datasets]

    # datasets = [sc.read(path+data+".h5ad") for data in "s5 509 1290 mousecortex water pancreatic cerebellum".split()]

    if remove_unassigned:
        okcells =  lambda ds: [ l not in ['-1','Unassigned','nan'] for l in ds.obs['label']]
        datasetsd = [ds[okcells(ds)] for ds in datasetsd]

    datasetsd =  [[z[z.obs['batch']==i] for i in z.obs['batch'].unique()]
                 for z in datasetsd]
    for batchlist in datasetsd:
        for batch in batchlist:
            batch.uns['timeseries'] = True
    datasetsd = fix(datasetsd,datasets)
    return datasetsd

def fix(ds,names):
    tgt = 'done_reprogramming_schiebinger'

    if tgt in names:
        adataa= ds[names.index(tgt)]
        adataa = adataa[2:]
        return ds
        for adata in adataa:
            candidate_keys = [
                'MEF.identity', 'Neural.identity', 'Placental.identity', 'XEN',
                'Trophoblast', 'Trophoblast progenitors', 'Spiral Artery Trophpblast Giant Cells',
                'Spongiotrophoblasts', 'Oligodendrocyte precursor cells (OPC)', 'Astrocytes',
                'Cortical Neurons', 'RadialGlia-Id3', 'RadialGlia-Gdf10', 'RadialGlia-Neurog2',
                'Long-term MEFs', 'Embryonic mesenchyme', 'Cxcl12 co-expressed',
                'Ifitm1 co-expressed', 'Matn4 co-expressed'
            ]
            # Subset the relevant columns from .obs
            score_df = adata.obs[candidate_keys]
            # For each cell, get the name of the column (i.e., cell type) with the highest score
            adata.obs['label'] = score_df.idxmax(axis=1)

    return ds

def test_ts():
    ds = load_timeseries()
    for i,adata in enumerate(ds):
        print(i)
        print(adata)
        for a in adata:
            print(np.unique(a.obs['label']))


