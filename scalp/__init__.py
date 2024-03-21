from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from scalp.data import transform
from scalp import data, pca, umapwrap, mnn, graph, test_config
from scalp import diffuse
from scalp.output import score
from scalp.data.align import align
from scalp.output.draw import snsplot


def mkgraph( adatas ,pre_pca = 40,
            neighbors_total = 20, neighbors_intra_fraction = .5,
              scaling_num_neighbors = 2, inter_outlier_threshold = .9,
                inter_outlier_probabilistic_removal= False,
            epsilon = 1e-6,
                intra_neighbors_mutual = False, copy_lsa_neighbors = False,
              add_tree= False, dataset_adjacency = None ):
    '''
    this does our embedding,
    written such that the optimizer can do its thing
    '''

    adatas = pca.pca(adatas,dim = pre_pca, label = 'pca')
    matrix = graph.linear_assignment_integrate(adatas,base = 'pca',
                                                neighbors_total=neighbors_total,
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
    return adatas, matrix


# diffuse.diffuse_label  -> diffuses the label

def graph_embed_plot(dataset,matrix, embed_label= 'embedding', snskwargs={}):
    dataset = umapwrap.graph_umap(dataset,matrix,label = embed_label)
    snsplot(dataset,coordinate_label=embed_label,**snskwargs)
    return dataset



def test_scalp():
    n_cells = 100
    a = data.loaddata_scib(test_config.scib_datapath, maxdatasets=3,
                           maxcells = n_cells, datasets = ["Immune_ALL_hum_mou"])[0]
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
    matrix = graph.linear_assignment_integrate(a, base ='pca40')
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



