import lucy.score as lscore
from sklearn.metrics import  silhouette_score
import scanpy as sc
import ubergauss.tools as ut


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
          scaling_threshold = .25,  pre_pca = 30, connect = 1231, nalg = 0,use_ladder= 0,connect_ladder = 1): # connect should be 0..1 , but its nice to catch errors :)
    assert connect < 1.1, "parameters were not passed.. :)"


    data = adatas.pca(data,dim = pre_pca, label = 'pca')
    # if data[0].uns['timeseries']:
    if use_ladder:
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

# we use this later for the eval
def runscore(params, dataset):
    data = dolucy(dataset,**params)
    score = scores(data)
    return score





# just format a task like this: [{deescription dict for the optimizer}, train-instances, test-instances]


def evalscores(tasks,incumbents):

    r = []
    for task, params in zip(tasks,incumbents):
        for test in task[2]:
            ssdata = ut.loadfile(f'garbage/{test}.delme')
            #data = [s.copy() for s in ssdata[test]]
            score = runscore(params,ssdata)
            for scorename, value in zip('label shilouette batchmix'.split(), score):
                nuvalues = {'dataset':test,f'score':value, f'test':scorename, f'algo':f'lucy'}
                nuvalues.update(task[0])
                r.append(nuvalues)
    return r



def getmnndicts(tasks,incumbents, all_tests): # i could calculate all_tests from  the tasks, but its easier if i just pass it


    ssdata = [ut.loadfile(f'garbage/{test}.delme') for test in all_tests]
    score = {i:runmnn(ssdata[i]) for i in all_tests }
    print(f"scoring with mnn might destroy ss... i should run this last or copy.")

    r = []
    for task in tasks:
        for test in task[2]:
            score = scoress[test]
            for scorename, value in zip('label shilouette batchmix'.split(), score):
                nuvalues = {'dataset':test,f'score':value, f'test':scorename, f'algo':f'mnn'}
                nuvalues.update(task[0])
                r.append(nuvalues)
    return r


