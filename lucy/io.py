from lmz import *
import scanpy as sc

def samplecopy(data,num, seed):
    np.random.seed(seed)
    obs_indices = np.random.choice(data.n_obs, size=num, replace=True)
    r=  data[obs_indices].copy()
    r.obs_names_make_unique()
    return r


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



def load_timeseries(path = './'):
    datasets = [sc.read(path+data+".h5ad") for data in
                "s5 509 1290 mousecortex water pancreatic cerebellum".split()]
    datasets =  [[z[z.obs['batch']==i] for i in z.obs['batch'].unique()]
                 for z in datasets]
    return datasets

