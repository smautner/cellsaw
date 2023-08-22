from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten

from scalp import data, pca, diffuse, umapwrap, graph
from scalp.data import transform


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



def test_scalp():
    from scalp import data, pca, diffuse, umapwrap, mnn, graph, test_config

    a = data.loaddata_scib(test_config.scib_datapath, maxdatasets=3, maxcells = 600, datasets = ["Immune_ALL_hum_mou.h5ad"])[0]

    print("=============== PCA ===============")
    a = pca.pca(a)
    print(f"{a[0].obsm['pca40'].shape = }")

    # print("=============== mnn ===============")
    # a = mnn.mnn(a)
    # print(f"{ a[0].obsm[f'mnn'].shape= }")

    print("=============== umap ===============")
    a = umapwrap.adatas_umap(a,label= 'umap10')
    print(f"{ a[0].obsm[f'umap10'].shape= }")

    print("=============== make lina-graph ===============")
    matrix = graph.to_linear_assignment_graph(a, base ='pca40')
    print(f"{matrix.shape=}")

    print("=============== diffuse label ===============")
    import copy
    b = diffuse.diffuse_label(copy.deepcopy(a), matrix, use_labels_from_datasets=[2, 1], new_label ='difflabel')
    print(f"{type(b[0].obs['difflabel'])=}")
    print(f"{b[0].obs['difflabel'].shape=}")

    print("=============== sklearn diffusion ===============")
    b = diffuse.diffuse_label_sklearn(b, ids_to_mask=[2, 1], new_label ='skdiff')
    print(f"{b[0].obs['skdiff'].shape=}")


    print("=============== lina-graph umap ===============")
    a = umapwrap.graph_umap(a,matrix, label = 'graphumap')
    print(f"{ a[0].obsm[f'graphumap'].shape= }")
