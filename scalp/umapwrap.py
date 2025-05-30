from scalp.data.transform import stack_single_attribute, attach_stack
import trimap
from sklearn.manifold import spectral_embedding
import pacmap
import umap
import numpy as np
from ubergauss import graphumap
from ubergauss import csrjax as cj
from scipy.sparse import csr_matrix






def adatas_umap(adatas, dim = 10, label = 'umap10', from_obsm = 'pca40', **umapargs):

    if label in adatas[0].obsm:
        print('redundant umap :) ')

    #attr = data.obsm.get(start,'')
    X = stack_single_attribute(adatas, attr = from_obsm)
    res = umap.UMAP(n_components = dim, **umapargs).fit_transform(X)
    return attach_stack(adatas, res, label)


from sklearn.manifold import MDS
def graph_umap(adatas=False, distance_adjacency_matrix=None,
               label=f'lsa', n_neighbors = 10,
               n_components = 2, **kwargs):
    # res =  umap.UMAP(n_components=n_components,
    #                  metric='precomputed',n_neighbors = n_neighbors,
    #                  **kwargs).fit_transform(distance_adjacency_matrix)
    distance_adjacency_matrix = csr_matrix(distance_adjacency_matrix)
    res = graphumap.graphumap(distance_adjacency_matrix, n_dim=n_components)
    if not adatas:
        return res
    return attach_stack(adatas, res, label)

def graph_mds(adatas, distance_adjacency_matrix,
               label='spring', n_components = 2, **kwargs):
    d = np.asarray(distance_adjacency_matrix.todense())
    d[d==0] = np.inf
    res = MDS(n_components=n_components,dissimilarity='precomputed').fit_transform(d)
    return attach_stack(adatas, res, label)

def graph_NX(adatas, distance_adjacency_matrix,
               label='spring',algo='spring', n_components = 2, **kwargs):
    res = graphumap.embed_via_nx(distance_adjacency_matrix,algo = algo, dim=n_components)
    return attach_stack(adatas, res, label)

from sknetwork.embedding import PCA
def graph_PCA(adatas, distance_adjacency_matrix,
               label='gpca', n_components = 2, **kwargs):
    res = PCA(n_components = n_components).fit_transform(distance_adjacency_matrix)
    return attach_stack(adatas, res, label)

from scalp.graph import negstuff
def graph_jax(adatas, distance_adjacency_matrix,
               label='jax', n_components = 2, **kwargs):
    data = csr_matrix(distance_adjacency_matrix), negstuff(adatas,**kwargs)
    res = cj.embed(data, n_components = n_components)
    return attach_stack(adatas, res, label)

def graph_xumap(adatas, distance_adjacency_matrix,
               label=f'lsa', n_neighbors = 10,
               n_components = 2, **kwargs):
    distance_adjacency_matrix = csr_matrix(distance_adjacency_matrix)
    res = umap.UMAP().fit_transform(distance_adjacency_matrix)
    return attach_stack(adatas, res, label)



def graph_pacmap(adatas = False, distance_adjacency_matrix=None,
               label=f'pacmap',
               n_components = 2, neighbors = None, MN= .5, FP=2, **kwargs):
    # distance_adjacency_matrix = csr_matrix(distance_adjacency_matrix)
    distance_adjacency_matrix = distance_adjacency_matrix.todense()
    distance_adjacency_matrix[distance_adjacency_matrix ==0 ] = 99999999
    res = pacmap.PaCMAP(n_components=n_components, n_neighbors=neighbors,
                        MN_ratio=MN, FP_ratio=FP).fit_transform(distance_adjacency_matrix)

    if not adatas:
        return res
    return attach_stack(adatas, res, label)

def graph_pacmap2(adatas=False, distance_adjacency_matrix=None,label='pacmap2',n_components =2):

    X , di = graphumap.make_knn(csr_matrix(distance_adjacency_matrix))
    nbrs=X
    n,n_neighbors = X.shape

    scaled_dist = np.ones((n, n_neighbors)) # No scaling is needed
    # Type casting is needed for numba acceleration
    X = X.astype(np.float32)
    scaled_dist = scaled_dist.astype(np.float32)
    # make sure n_neighbors is the same number you want when fitting the data
    pair_neighbors = pacmap.sample_neighbors_pair(X.astype(np.float32),
                    scaled_dist.astype(np.float32), nbrs.astype(np.int32), np.int32(n_neighbors))
    # initializing the pacmap instance
    # feed the pair_neighbors into the instance
    embedding = pacmap.PaCMAP(n_components=n_components, n_neighbors=n_neighbors,
                        MN_ratio=0.5, FP_ratio=2.0, pair_neighbors=pair_neighbors)

    # fit the data (The index of transformed data corresponds to the index of the original data)
    X_transformed = embedding.fit_transform(X, init="pca")
    if not adatas:
        return X_transformed
    return attach_stack(adatas, X_transformed, label)



def graph_trimap(adatas, distance_adjacency_matrix,
               label=f'trimap',
               n_components = 2):
    # distance_adjacency_matrix = csr_matrix(distance_adjacency_matrix)
    distance_adjacency_matrix = np.asarray(distance_adjacency_matrix.todense())
    # res = pacmap.PaCMAP(n_components=n_components, n_neighbors=neighbors, MN_ratio=MN, FP_ratio=FP).fit_transform(distance_adjacency_matrix)
    res = trimap.TRIMAP(use_dist_matrix=True).fit_transform(distance_adjacency_matrix)
    return attach_stack(adatas, res, label)

def graph_spectral(adatas, distance_adjacency_matrix,
               label=f'lsa', n_neighbors = 10,
               n_components = 2, **kwargs):
    distance_adjacency_matrix = csr_matrix(distance_adjacency_matrix)
    res = spectral_embedding(distance_adjacency_matrix, n_components=2)
    return attach_stack(adatas, res, label)

from sklearn.decomposition import TruncatedSVD
def graph_tsvd(adatas, distance_adjacency_matrix,
               label=f'tsvd',
               n_components = 2):
    # distance_adjacency_matrix = csr_matrix(distance_adjacency_matrix)
    #distance_adjacency_matrix = np.asarray(distance_adjacency_matrix.todense())
    # res = pacmap.PaCMAP(n_components=n_components, n_neighbors=neighbors, MN_ratio=MN, FP_ratio=FP).fit_transform(distance_adjacency_matrix)
    res = TruncatedSVD().fit_transform(distance_adjacency_matrix)
    return attach_stack(adatas, res, label)


import bbknn
import scanpy as sc
from scalp.data import transform
def umap_last_experiment(adata,adjacencymatrix, base = 'pca40', label = 'lastexpo',
                         batchindicator = 'batch', n_components = 2):
    '''
    the idea is to use bbknn stuff to do out umap emebdding
    '''
    adata.obsm.pop('umap',None)
    adata.obsm.pop('X_umap',None)
    knn_indices, knn_distances= graphumap.make_knn( csr_matrix(adjacencymatrix))
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
    adata.uns.pop('umap', None)
    adata.obsm.pop('X_umap', None)

    sc.tl.umap(adata,n_components=n_components)
    adata.uns.pop('umap')
    return adata.obsm.pop('X_umap')
