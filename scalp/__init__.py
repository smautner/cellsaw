from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from scalp.data import transform
from scalp import data, pca, umapwrap, mnn, graph, test_config
from scalp import diffuse
from scalp.output import score
from scalp.data.align import align
from scalp.output.draw import snsplot
import scanpy as sc




def construct_sparse_adjacency_matrix_multiple(matrices, k, h):
    """
    Constructs a sparse adjacency matrix based on k-NN within each matrix,
    adds cross-edges using linear assignment pairwise between matrices,
    and filters edges based on horizon h.

    Parameters:
    - matrices: list of numpy arrays, each of shape (n_i, d)
    - k: int, number of nearest neighbors within each matrix
    - h: int, horizon parameter for filtering edges

    Returns:
    - adjacency: scipy.sparse.csr_matrix of shape (total_instances, total_instances)
    """
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    # Construct k-NN within each matrix
    knn = []
    for matrix in matrices:
        tree = cKDTree(matrix)
        dists, neighs = tree.query(matrix, k=k+1)
        knn.append(neighs[:,1:])
    # Construct cross-edges using linear assignment pairwise between matrices
    n = sum([len(neighs) for neighs in knn])
    row, col, data = [], [], []
    for i, neighs1 in enumerate(knn):
        for j, neighs2 in enumerate(knn):
            if i == j:
                continue
            cost = cdist(matrices[i], matrices[j][neighs2.flatten()])
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                row.append(neighs1[r] + i)
                col.append(neighs2[c] + j)
                data.append(1)
    row = np.concatenate(row)
    col = np.concatenate(col)
    data = np.concatenate(data)
    # Construct adjacency matrix
    adjacency = csr_matrix((data, (row, col)), shape=(n, n))
    # Filter edges based on horizon h
    for i in range(n):
        row = adjacency[i].indices
        col = adjacency[:,i].indices
        row = row[np.abs(row - i) <= h]
        col = col[np.abs(col - i) <= h]
        adjacency[i, row] = 1
        adjacency[col, i] = 1
    return adjacency





# horizonCutoff 4 10 1 # the idea is flawed
mkgraphParameters = '''
neighbors_total 15 45 1
neighbors_intra_fraction .2 .5
intra_neighbors_mutual 0 1 1
add_tree 0 1 1
horizonCutoff 50 100 1
standardize 0 1 1
'''
# copy_lsa_neighbors 0 1 1
# distance_metric ['euclidean', 'sqeuclidean' ]
# inter_outlier_threshold .60 .97
# inter_outlier_probabilistic_removal 0 1 1

from sklearn.preprocessing import StandardScaler

def mkgraph( adata ,pre_pca = 40,
            horizonCutoff = 0,
            neighbors_total = 20, neighbors_intra_fraction = .5,
              scaling_num_neighbors = 2, inter_outlier_threshold = -1,
            distance_metric = 'euclidean',
                inter_outlier_probabilistic_removal= False,
            epsilon = 1e-6,standardize=0,
                intra_neighbors_mutual = False, copy_lsa_neighbors = False,
              add_tree= False, dataset_adjacency = None, **kwargs ):
    '''
    this does our embedding,
    '''
    # adatas = pca.pca(adatas,dim = pre_pca, label = 'pca40')
    assert 'pca40' in adata.obsm

    if horizonCutoff:
        inter_outlier_threshold = 0
        inter_outlier_probabilistic_removal = False

    if standardize == 0: # no standardization
        adatas = data.transform.split_by_obs(adata)
    elif standardize == 1: # joint
        sc.pp.scale(adata)
        adatas = data.transform.split_by_obs(adata)
    elif standardize == 2: # separate
        adatas = data.transform.split_by_obs(adata)
        [sc.pp.scale(a) for a in adatas]
    else:
        assert False, f"unknown standardize value {standardize=}"


    if False:#aislop
        matrix = graph.aiSlopSolution(adatas, 20, 240)
    else:
        matrix = graph.linear_assignment_integrate(adatas,base = 'pca40',
                                                    neighbors_total=neighbors_total,
                                                distance_metric=distance_metric,
                                horizonCutoff = horizonCutoff,
                                                    neighbors_intra_fraction=neighbors_intra_fraction,
                                                      intra_neighbors_mutual=intra_neighbors_mutual,
                                                      outlier_probabilistic_removal= inter_outlier_probabilistic_removal,
                                                      scaling_num_neighbors = scaling_num_neighbors,
                                                      outlier_threshold = inter_outlier_threshold,
                                                      dataset_adjacency =  dataset_adjacency,
                                                      copy_lsa_neighbors=copy_lsa_neighbors,
                                                   epsilon=epsilon,
                                                  add_tree=add_tree)
    #data = umapwrap.graph_umap(data, matrix, label = 'graphumap')
    if False: # debug
        from scipy.sparse import csr_matrix
        import structout as so
        matrix2 = csr_matrix(matrix)
        vals = [ len(x.data) for x in matrix2]
        print(f"will plot the number of neighbors for each item... {min(vals)=},{max(vals)=}")
        so.lprint(vals)
    return  matrix



# diffuse.diffuse_label  -> diffuses the label

def graph_embed_plot(dataset,matrix, embed_label= 'embedding', snskwargs={}):
    dataset = umapwrap.graph_umap(dataset,matrix,label = embed_label)
    snsplot(dataset,coordinate_label=embed_label,**snskwargs)
    return dataset

import umap
def plot(adata,embedding,**plotargs):
    # adata.obsm['X_umap']=adata.obsm[embedding]
    # sc.pl.umap(adata,basis= embedding, **plotargs)

    if adata.obsm[embedding].shape[1] > 2:
        adata.obsm['newlayer'] =  umap.UMAP(n_components = 2).fit_transform(adata.obsm[embedding])
    else:
        adata.obsm['newlayer'] =  adata.obsm[embedding]
    sc.pl.embedding(adata, basis= 'newlayer', **plotargs)


def test_scalp():
    n_cells = 100
    a = data.scib(test_config.scib_datapath, maxdatasets=3,
                           maxcells = n_cells, datasets = ["Immune_ALL_hum_mou"]).__next__()
    # print("=============== mnn ===============")
    # mnn and scanvi are no longer maintained, scanoram is second on the nature method ranking
    # a = mnn.mnn(a)
    # print(f"{ a[0].obsm[f'mnn'].shape= }")

    print("=============== PCA ===============")
    a = pca.pca(a)
    print(f"{a[0].obsm['pca40'].shape = }")
    assert a[0].obsm['pca40'].shape == (n_cells,40)
    align(a,'pca40')

    print("=============== scanorama ===============")
    a = mnn.scanorama(a)
    print(f"{ a[0].obsm[f'scanorama'].shape= }")

    print("=============== umap ===============")
    a = umapwrap.adatas_umap(a,label= 'umap10')
    print(f"{ a[0].obsm['umap10'].shape= }")
    assert a[0].obsm['umap10'].shape == (n_cells,10)

    print("=============== make lina-graph ===============")
    # matrix = graph.linear_assignment_integrate(a, base ='pca40')
    matrix = graph.integrate(a, base ='pca40')
    print(f"{matrix.shape=}")
    assert matrix.shape== (n_cells*3,n_cells*3)

    print("=============== diffuse label ===============")
    a = diffuse.diffuse_label(a, matrix, use_labels_from_dataset_ids=[2, 1], new_label ='difflabel')
    #print(f"{type(a[0].obs['difflabel'])=}")
    print(f"{a[0].obs['difflabel'].shape=}")
    assert a[0].obs['difflabel'].shape== (n_cells,)
    print(f"{Map(score.anndata_ari, a, predicted_label='difflabel')=}")

    print("=============== sklearn diffusion ===============")
    a = diffuse.diffuse_label_sklearn(a, use_labels_from_dataset_ids=[2, 1], new_label ='skdiff')
    print(f"{a[0].obs['skdiff'].shape=}")
    assert a[0].obs['skdiff'].shape== (n_cells,)

    print("=============== lina-graph umap ===============")
    a = umapwrap.graph_umap(a,matrix, label = 'graphumap')
    print(f"{ a[0].obsm['graphumap'].shape= }")
    assert a[0].obsm['graphumap'].shape== (n_cells,2)



