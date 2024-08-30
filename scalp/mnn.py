from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from scalp.data import transform
import ubergauss.tools as ut
import scanpy.external.pp as sep
import scanpy as sc

def scanorama(adatas, base = 'pca40', batchindicator = 'batch', label =  'scanorama'):
    adata = transform.stack(adatas)
    # sep.scanorama_integrate(adata, batchindicator, basis = base, adjusted_basis = label)
    sep.scanorama_integrate(adata, batchindicator, basis = base, adjusted_basis = label)
    return transform.split_by_obs(adata)

def combat(adatas, base = 'pca40', batchindicator = 'batch', label =  'combat'):
    adata = transform.stack(adatas)
    r = sc.pp.combat(adata,batchindicator, inplace=False)
    adata.obsm[label] = r
    return transform.split_by_obs(adata)

import bbknn
def bbknnwrap(adatas, base = 'pca40',
          batchindicator = 'batch', dim = 2):
    adata = transform.stack(adatas)
    # sc.external.pp.bbknn(adata, batchindicator, use_rep=base)
    bbknn.bbknn(adata, use_rep = base)
    sc.tl.umap(adata,n_components=dim)
    return transform.split_by_obs(adata)

    # use this to do umap to a speciffic dim:
    # https://scanpy.readthedocs.io/en/latest/generated/scanpy.tl.umap.html#scanpy-tl-umap

def mnn(adata, label = 'mnn'):

    # this used to work... grrr
    #mnn = sc.external.pp.mnn_correct(adata, n_jobs = 30)
    #mnnstack = adatas.stack(mnn[0][0])

    # needs to be dense...
    import mnnpy
    mnnpy.settings.normalization = "single"

    data = [ut.zehidense(a.X) for a in adata]
    matrixes = mnnpy.mnn_correct(*data,
                                 n_jobs = 1, do_concatenate = False, var_index = Range(data[0].shape[1]) )[0]
    # data = transform.stack(adata)
    # data.obsm[target] = mnn_matrix[0].X
    # adata = transform.attach_stack(adata,mnn_matrix,label)


    for a,m in zip(adata,matrixes):
        a.obsm[label] = m

    return adata




def umap_last_experiment(adata,adjacencymatrix, base = 'pca40', batchindicator = 'batch', n_components = 2):
    '''
    the idea is to use bbknn stuff to do out umap emebdding
    '''
    # adata = transform.stack(adatas)
    knn_indices, knn_distances= graphumap.make_knn(adjacencymatrix)
    dist, cnts = bbknn.matrix.compute_connectivities_umap(knn_indices, knn_distances,
                                             knn_indices.shape[0],
                                             knn_indices.shape[1],
                                             set_op_mix_ratio=1,
                                             local_connectivity=1)


    p_dict = {'n_neighbors': knn_distances.shape[1], 'method': 'umap'}
                          # 'metric': params['metric'], 'n_pcs': params['n_pcs'],
                          # 'bbknn': {'trim': params['trim'], 'computation': params['computation']}}
    key_added = 'neighbors'
    conns_key = 'connectivities'
    dists_key = 'distances'
    adata.uns[key_added] = {}

    adata.uns[key_added]['params'] = p_dict
    adata.uns[key_added]['params']['use_rep'] = base
    #adata.uns[key_added]['params']['bbknn']['batch_key'] = batchindicator

    adata.obsp[dists_key] = dist
    adata.obsp[conns_key] = cnts
    adata.uns[key_added]['distances_key'] = dists_key
    adata.uns[key_added]['connectivities_key'] = conns_key
    sc.tl.umap(adata,n_components=n_components)
    return adata #transform.split_by_obs(adata)
