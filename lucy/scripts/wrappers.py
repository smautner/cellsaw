


import lucy.score as lscore
from sklearn.metrics import  silhouette_score
import scanpy as sc


def scores(data, projectionlabel = 'lsa'):
    y = data.obs['label'].tolist()
    ybatch = data.obs['batch'].tolist()
    sim = data.obsm[projectionlabel]

    score = lscore.neighbor_labelagreement(sim,y,5)
    silou = silhouette_score(sim,y)
    batchmix = -lscore.neighbor_labelagreement(sim,ybatch,5)
    return score, silou, batchmix




###########
# lucy wrapper..
###############

from lucy import load, adatas


def dolucy( data ,intra_neigh=10,inter_neigh=5, scaling_num_neighbors=1,embed_components=5,outlier_threshold = .75,
          scaling_threshold = .25,  pre_pca = 30, connect = 1231, nalg = 0, connect_ladder = 1): # connect should be 0..1 , but its nice to catch errors :)

    data = adatas.pca(data,dim = pre_pca, label = 'pca')
    if data[0].uns['timeseries']:
        dataset_adjacency = adatas.embed.make_adjacency(adatas.similarity(data), nalg, connect)
    else:
        dataset_adjacency = adatas.embed.make_sequence(adatas.similarity(data),  connect_ladder)

    lsa_graph = adatas.lapgraph(data,base = 'pca',
                                              intra_neigh = intra_neigh,
                                              inter_neigh = inter_neigh,
                                              scaling_num_neighbors = scaling_num_neighbors,
                                              outlier_threshold = outlier_threshold,
                                              scaling_threshold = scaling_threshold,
                                              dataset_adjacency =  dataset_adjacency)
    data = adatas.graph_embed(data,lsa_graph,n_components = embed_components, label = 'lsa')
    data = adatas.stack(data)
    return data




###########
# MNN wrapper..
###############

def domnn(adata):
    #
    mnn = sc.external.pp.mnn_correct(adata, n_jobs = 30)
    mnnstack = adatas.stack(mnn[0][0])

    data = adatas.stack(adata)
    data.obsm['lsa'] = mnnstack.X
    return data

