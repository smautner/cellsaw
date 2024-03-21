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
def graph_umap(adatas, distance_adjacency_matrix,
               label=f'lsa', n_neighbors = 10,
               n_components = 2, **kwargs):
    # res =  umap.UMAP(n_components=n_components,
    #                  metric='precomputed',n_neighbors = n_neighbors,
    #                  **kwargs).fit_transform(distance_adjacency_matrix)
    distance_adjacency_matrix = csr_matrix(distance_adjacency_matrix)
    res = graphumap.graphumap(distance_adjacency_matrix, n_dim=n_components)
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

def graph_pacmap(adatas, distance_adjacency_matrix,
               label=f'pacmap',
               n_components = 2,neighbors = None, MN= .5, FP=2, **kwargs):
    # distance_adjacency_matrix = csr_matrix(distance_adjacency_matrix)
    # distance_adjacency_matrix = distance_adjacency_matrix.todense()
    res = pacmap.PaCMAP(n_components=n_components, n_neighbors=neighbors, MN_ratio=MN, FP_ratio=FP).fit_transform(distance_adjacency_matrix)
    return attach_stack(adatas, res, label)

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


