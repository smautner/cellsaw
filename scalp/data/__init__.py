# from lucy import load, adatas
import random

from lmz import Map,Range
from scalp.data.hvg import preprocess
from scalp.data.subsample import subsample, subsample_iflarger
import scanpy as sc
from scalp.data import load
import numpy as np

from scalp.data import transform

# def loaddata_timeseries(path,maxcells= 1000, maxdatasets = -1, **ppargs):
#     datasets = load.load_timeseries(path )
#     ssdata = subsample_preprocess(datasets, maxcells= maxcells, maxdatasets = maxdatasets, **ppargs)
#     return ssdata
# def loaddata_scib(path,maxcells= 1000, maxdatasets = -1, datasets = False, **ppargs):
#     datasets = load.load_scib(path, datasets = datasets)
#     ssdata = subsample_preprocess(datasets, maxcells= maxcells, maxdatasets = maxdatasets, **ppargs)
#     return ssdata





datasets_scmark = ['external_azizi_cell_2018_29961579.h5ad',
 'external_bassez_natmed_2021_33958794.h5ad',
 'external_bi_cancercell_2021_33711272.h5ad',
 'external_elyada_cancerdiscov_2019_31197017.h5ad',
 'external_karlsson_sciadv_2021_34321199.h5ad',
 'external_lee_natgenet_2020_32451460.h5ad',
 'external_nath_natcommun_2021_34031395.h5ad',
 'external_peng_cellres_2019_31273297.h5ad',
 'external_qian_cellres_2020_32561858.h5ad',
 'external_slyper_natmed_2020_32405060.h5ad',
 'external_zhang_procnatlacadsciusa_2021_34099557.h5ad']


datasets_scib = "Immune_ALL_hum_mou Immune_ALL_human Lung_atlas_public human_pancreas_norm_complexBatch".split()
# datasets_ts = "s5 509 1290 mousecortex water pancreatic cerebellum".split()
# datasets_ts = "s5 509 1290 mousecortex water pancreatic cerebellum done_bone_marrow done_lung done_pancreas done_reprogramming_morris done_reprogramming_schiebinger".split()
datasets_ts = "s5 509 1290 mousecortex water pancreatic cerebellum done_bone_marrow done_lung done_pancreas done_reprogramming_morris".split()



# ####################
# # dont use theese, too much memory is required and they return the non-stacked datasets
# ###############
# def loaddata_timeseries(path,datasets=False,maxcells= 1000, maxdatasets = -1, **ppargs):
#     datasets = load.load_timeseries(path , datasets= datasets)
#     ssdata = subsample_preprocess(datasets, maxcells= maxcells, maxdatasets = maxdatasets, **ppargs)
#     return ssdata
# def loaddata_scib(path, datasets = False, maxcells= 1000, maxdatasets = -1, **ppargs):
#     datasets = load.load_scib(path, datasets= datasets)
#     ssdata = subsample_preprocess(datasets, maxcells= maxcells, maxdatasets = maxdatasets,pretransformed = True, **ppargs)
#     return ssdata


################################
# use these instead
##########################
def scib(path,datasets=False,maxcells=1000,slow=0,maxdatasets=-1,**other):
    '''
    this function is a generator that yields the preprocessed datasets
    '''
    print('loading scib datasets..')
    if not datasets:
        datasets = datasets_scib

    def loader(dataset):
        data = load.load_scib(path,  datasets = [dataset])
        data = subsample_preprocess(data,maxcells=maxcells,maxdatasets=maxdatasets, **other)[0]
        data = transform.stack(data)
        data.uns['timeseries'] = False
        data.uns['name'] = dataset
        if find_duplicate_csr(csr_matrix(data.X)): print (':(')
        return data
    if slow:
        return Map(loader, datasets)
        # for dataset in datasets: yield(loader(dataset))
    else:
        return ut.xxmap(loader,datasets)


def timeseries(path,datasets=False,maxcells=1000,slow=0,maxdatasets=-1,**other):
    '''
    this function is a generator that yields the preprocessed datasets
    '''
    print('loading timeseries datasets..')
    if not datasets:
        datasets = datasets_ts
    def loader(dataset):
        data = load.load_timeseries(path,  datasets = [dataset])
        if False:
            counts = {id:e.X.shape[0]for id,e in enumerate(data[0])}
            print(f"{dataset=} {counts=}")

        data = subsample_preprocess(data,maxcells=maxcells,maxdatasets=maxdatasets, **other)[0]
        data = transform.stack(data)
        data.uns['timeseries'] = True
        data.uns['name'] = dataset
        if find_duplicate_csr(csr_matrix(data.X)): print (':(')
        return data
    if slow:
        return Map(loader, datasets)
        # for dataset in datasets: yield(loader(dataset))
    else:
        return ut.xxmap(loader,datasets)



def scmark(path,datasets=False,maxcells=1000,slow=0,maxdatasets=-1,**other):
    '''
    this function is a generator that yields the preprocessed datasets
    '''
    print('loading scmark datasets..')
    if not datasets:
        datasets = datasets_scmark

    def l(dataset):
        data = sc.read_h5ad('/home/ubuntu/data/scmark/scmark_v2/' + dataset)
        data.X.data = data.X.data.astype(int) - 1
        data.X.eliminate_zeros()
        data.obs['batch'] = data.obs['sample_name']
        data.obs['label'] = data.obs['standard_true_celltype']

        data = transform.split_by_obs(data)

        data = [d for d in data if d.X.shape[0] > 99]

        # data = load.load_timeseries(path,  datasets = [dataset])
        data = subsample_preprocess([data],maxcells=maxcells,maxdatasets=maxdatasets, **other)

        data = data[0]
        data = transform.stack(data)
        data.uns['timeseries'] = False
        data.uns['name'] = dataset
        if find_duplicate_csr(csr_matrix(data.X)): print (':(')
        return data

    if slow:
        return Map(l, datasets)
        # for dataset in datasets: yield l(dataset)
    else:
        return ut.xxmap(l,datasets)

# for e in ds: print(demo.find_duplicate_rows(e.X))
from scipy.sparse import csr_matrix
def find_duplicate_rows(adl):
    for x in adl:
        mat=csr_matrix(x.X)
        if find_duplicate_csr(mat):
            return True
    return False

def find_duplicate_csr(mat):
    di = {}
    for i, row in enumerate(mat):
        h = hash(tuple(row.data) + tuple(row.indices))
        if h in di:
            # print(i, di[h])
            return True
        di[h] = i
    return False




from scalp import pca
from ubergauss import tools as ut




def subsample_preprocess(datasets, maxcells = 1000, maxdatasets = 10,random_datasets = False,
                          **preprocessing_args):


    filter_clusters = preprocessing_args.pop('filter_clusters', 0)
    def select_slices(series):
        if random_datasets:
            random.shuffle(series)
        ret = series[:maxdatasets]

        if filter_clusters > 0:
            # remove rare clusters... but we need to make sure that no dataset is affected disproportionally
            # so each item in ret is an anndata object. we extract obs['labels'] and rank each label for each dataset
            # if a label is not in all datasets, we remove it
            # for the rest we calculate the average rank and select the filter_cluster top ones :)

            all_labels = set()
            for s in ret:all_labels.update(s.obs['label'].unique())
            labels = [dict(s.obs['label'].value_counts(dropna=False)) for s in ret]

            # make a numpy array all_labels x all_labels
            array = np.zeros((len(ret), len(all_labels)))
            # populate with the labels
            sm = ut.spacemap(all_labels)
            for i, d in enumerate(labels):
                array[i,sm.encode(d.keys())] = list(d.values())
            # remove columns that contain 0 from array
            array = array[:, np.any(array != 0, axis=0)]
            # turn the numbers into ranks
            ranks = array.argsort(axis=0).argsort(axis=0)
            # calculate the average rank for each label across datasets
            avg_ranks = np.mean(ranks, axis=0)
            # get the labels corresponding to the smallest average ranks
            top_labels_indices = np.argsort(avg_ranks)[:filter_clusters]
            # map back to original label names
            top_labels = sm.decode(top_labels_indices)
            # filter each series in ret
            ret = [  s[s.obs['label'].isin(top_labels)] for s in ret]
        return ret

    print(find_duplicate_rows(datasets[0]),end=' ->')
    ssdata = [[subsample(i,maxcells,3143) for i in select_slices(series)] for series in datasets]
    print(find_duplicate_rows(ssdata[0]),end=' ->')
    ssdata = Map( pca.pca, ssdata, dim = 40, label = 'pca40')
    print(find_duplicate_rows(ssdata[0]),end=' ->')
    ret =  Map(preprocess, ssdata, **preprocessing_args)
    print(find_duplicate_rows(ret[0]))
    return ret


####################
# a test
#####################

def test_load():
    from scalp import test_config
    a = list(scib(test_config.scib_datapath, filter_clusters = 10))
    # b = timeseries(test_config.timeseries_datapath, filter_clusters=10)












##################
# BLOB stuff
########################


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


def rotate(anndata, degree=90):
    center = anndata.X.mean(axis=0)
    anndata.X -= center
    angle = degree/180*np.pi
    rot = np.array([[np.cos(angle),-np.sin(angle)],
                    [np.sin(angle),np.cos(angle)]])
    anndata.X = anndata.X @ rot
    anndata.X+= center
    anndata.obsm['umap']= anndata.X
    return anndata




