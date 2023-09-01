from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from scalp import data, pca, diffuse, umapwrap, graph
from scalp.data import transform
from scalp import data, pca, diffuse, umapwrap, mnn, graph, test_config
from scalp.output import score
from scalp.data.align import align
from sklearn.manifold import Isomap

def embed( data ,pre_pca = 40,isodim = 10, intra_neigh = 15, inter_neigh = 1,
              scaling_num_neighbors = 2, outlier_threshold = .8,
              scaling_threshold = .9, dataset_adjacency = None):
    '''
    this does our embedding,
    written such that the optimizer can do its thing
    '''
    data = pca.pca(data,dim = pre_pca, label = 'pca')

    matrix = graph.to_linear_assignment_graph(data,base = 'pca',
                                                  intra_neigh = intra_neigh,
                                                  inter_neigh = inter_neigh,
                                                  scaling_num_neighbors = scaling_num_neighbors,
                                                  outlier_threshold = outlier_threshold,
                                                  scaling_threshold = scaling_threshold,
                                                  dataset_adjacency =  dataset_adjacency)
    #data = umapwrap.graph_umap(data, matrix, label = 'graphumap')
    emb = Isomap(n_components=isodim,metric = 'precomputed').fit_transform(matrix)
    data = transform.attach_stack(data, emb, 'emb')
    data = transform.stack(data)
    return data


def make_matrix_diffuse_label(a, **kwargs):
    a = pca.pca(a)
    matrix = graph.to_linear_assignment_graph(a, base ='pca40')
    return diffuse.diffuse_label(a, matrix, **kwargs) # use_labels_from_datasets=[2, 1], new_label ='difflabel')

def test_scalp():
    a = data.loaddata_scib(test_config.scib_datapath, maxdatasets=3, maxcells = 600, datasets = ["Immune_ALL_hum_mou.h5ad"])[0]
    # print("=============== mnn ===============")
    # mnn and scanvi are no longer maintained, scanoram is second on the nature method ranking
    # a = mnn.mnn(a)
    # print(f"{ a[0].obsm[f'mnn'].shape= }")

    print("=============== PCA ===============")
    a = pca.pca(a)
    print(f"{a[0].obsm['pca40'].shape = }")
    assert a[0].obsm['pca40'].shape == (600,40)
    align(a,'pca40')

    print("=============== scanorama ===============")
    a = mnn.scanorama(a)
    print(f"{ a[0].obsm[f'scanorama'].shape= }")

    print("=============== umap ===============")
    a = umapwrap.adatas_umap(a,label= 'umap10')
    print(f"{ a[0].obsm['umap10'].shape= }")
    assert a[0].obsm['umap10'].shape == (600,10)

    print("=============== make lina-graph ===============")
    matrix = graph.to_linear_assignment_graph(a, base ='pca40')
    print(f"{matrix.shape=}")
    assert matrix.shape== (1800,1800)

    print("=============== diffuse label ===============")
    a = diffuse.diffuse_label(a, matrix, use_labels_from_datasets=[2, 1], new_label ='difflabel')
    #print(f"{type(a[0].obs['difflabel'])=}")
    print(f"{a[0].obs['difflabel'].shape=}")
    assert a[0].obs['difflabel'].shape== (600,)
    print(f"{Map(score.anndata_ari, a, predicted_label='difflabel')=}")

    print("=============== sklearn diffusion ===============")
    a = diffuse.diffuse_label_sklearn(a, ids_to_mask=[2, 1], new_label ='skdiff')
    print(f"{a[0].obs['skdiff'].shape=}")
    assert a[0].obs['skdiff'].shape== (600,)

    print("=============== lina-graph umap ===============")
    a = umapwrap.graph_umap(a,matrix, label = 'graphumap')
    print(f"{ a[0].obsm['graphumap'].shape= }")
    assert a[0].obsm['graphumap'].shape== (600,2)



