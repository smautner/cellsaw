from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import scalp.data.subsample
import scanpy as sc
import numpy as np

def check_adatas(adatas):
        assert isinstance(adatas, list), f'merge wants a list, not {type(adatas)}'
        assert all([a.X.shape[1] == adatas[0].X.shape[1] for a in adatas])




def preprocess(adatas,cut_ngenes = 2000, cut_old = False, hvg = 'cell_ranger', make_even = True, pretransformed = False):


    check_adatas(adatas)
    if hvg == 'cell_ranger':
        adatas = cell_ranger(adatas, pretransformed = pretransformed)
    else:
        assert False

    selector = hvg_ids_from_union if cut_old else hvg_ids_from_union_limit_binsearch

    for a in adatas: # saving this info to be able to calculate similarity
        a.uns[hvg] = a.var[hvg]

    adatas = hvg_cut(adatas, selector(adatas,cut_ngenes,hvg_name=hvg))
    if make_even:
        adatas = subsample_to_min_cellcount(adatas)

    return adatas


def hvg_cut(adatas,hvg_ids):
    [d._inplace_subset_var(hvg_ids) for d in adatas]
    return adatas


def hvg_ids_from_union(adatas, numGenes, hvg_name= 'cell_ranger'):
    scores = [a.var[hvg_name] for a in adatas]
    hvg_ids_per_adata = np.argpartition(scores, -numGenes)[:,-numGenes:]
    hvg_ids = np.unique(hvg_ids_per_adata.flatten())
    return hvg_ids


def hvg_ids_from_union_limit_binsearch(adatas,numgenes,hvg_name = 'cell_ranger'):

    scores = [a.var[hvg_name] for a in adatas]
    ar = np.array(scores)
    ind = np.argsort(ar)

    def top_n_union(array,n):
        indices = array[:,-n:]
        indices = np.unique(indices.flatten())
        return indices

    def findcutoff(low,high, lp = -1):
        probe = int((low+high)/2)
        y = top_n_union(ind,probe) # hvg_ids_from_union(adatas,probe)
        if probe == lp:
            return y
        if len(y) > numgenes:
            return findcutoff(low,probe,probe)
        else:
            return findcutoff(probe,high,probe)

    indices = findcutoff(0,numgenes)
    return indices


import pandas as pd
def hvg_ids_from_union_limit_additive(
    adatas,
    numgenes: int,
    hvg_name: str = 'cell_ranger'
) -> np.ndarray:
    """

    """
    if not adatas:
        return np.array([], dtype=int)

    # 1. Get indices sorted by score for each dataset.
    #    The `[:, ::-1]` reverses each row so columns rank from most to least variable.
    sorted_indices = np.argsort(
        np.array([a.var[hvg_name].values for a in adatas])
    )[:, ::-1]

    # 2. Flatten the indices rank-by-rank.
    #    .T transposes so that rows represent ranks (rank 1, rank 2, ...).
    #    .flatten() creates a single array: [d1_rank1, d2_rank1, ..., d1_rank2, d2_rank2, ...]
    #    This ensures we process all top-ranked genes before any second-ranked genes.
    ordered_indices = sorted_indices.T.flatten()

    # 3. Get the first `numgenes` unique indices from this ordered list.
    #    `pd.Series.unique()` is highly optimized and preserves order of appearance.
    #    We then slice to get exactly the number of genes we need.
    unique_indices = pd.Series(ordered_indices).unique()[:numgenes]

    # 4. Sort for a deterministic output, as the order from `unique` depends
    #    on the order of datasets in the input list.
    return np.sort(unique_indices)



def subsample_to_min_cellcount(adatas):
        smallest = min([e.X.shape[0] for e in adatas])
        for a in adatas:
            if a.X.shape[0] > smallest:
                sc.pp.subsample(a, n_obs=smallest, random_state=0, copy=False)
                # adatas = scalp.data.subsample(a, num = smallest, seed=0, copy=False)
        return adatas


def cell_ranger(adatas, mingenes = 200,
                        normrow= True,
                        pretransformed = False,
                        log = True):
    if 'cell_ranger' in adatas[0].var:
        return adatas

    return Map( lambda x:cell_ranger_single(x, mingenes=mingenes, normrow= normrow,  log= log, pretransformed = pretransformed), adatas)


def cell_ranger_single(adata,
                        mingenes = 200,
                        normrow= True,
                        pretransformed = False,
                        log = True):

    if pretransformed:
        adata.X = np.expm1(adata.X)
    okgenes = sc.pp.filter_genes(adata, min_counts=3, inplace=False)[0]

    if not pretransformed:
        sc.pp.normalize_total(adata, 1e4)
    sc.pp.log1p(adata)
    adata2 = adata[:,okgenes].copy()

    sc.pp.highly_variable_genes(adata2, n_top_genes=5000,
                                         flavor='cell_ranger',
                                        inplace=True)

    fullscores = np.full(adata.X.shape[1],np.NINF,np.cfloat)
    fullscores[okgenes]  = adata2.var['dispersions_norm']
    adata.var['cell_ranger']=  fullscores
    return adata
