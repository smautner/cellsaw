from lmz import Map,Zip,Filter,Grouper,Range,Transpose, Flatten
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

    # hyperopt best sucks because it will give me floats ...
    intra_neigh, inter_neigh, scaling_num_neighbors, embed_components, pre_pca, use_ladder, connect_ladder = Map(int, [intra_neigh, inter_neigh, scaling_num_neighbors, embed_components, pre_pca, use_ladder, connect_ladder ])
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

    # this used to work...
    #mnn = sc.external.pp.mnn_correct(adata, n_jobs = 30)
    #mnnstack = adatas.stack(mnn[0][0])

    # needs to be dense...
    import mnnpy
    for a in adata:
        a.X = ut.zehidense(a.X)
    mnnpy.settings.normalization = "single"
    mnn = mnnpy.mnn_correct(*adata, n_jobs = 1)


    data = adatas.stack(adata)
    data.obsm['lsa'] = mnn[0].X
    return data

# we use this later for the eval
def runscore(params, dataset):
    data = dolucy(dataset,**params)
    score = scores(data)
    return score





# just format a task like this: [{deescription dict for the optimizer}, train-instances, test-instances]

def _eval(x):
    task, params, test , datapath= x
    r=[]
    ssdata = ut.loadfile(f'{datapath}garbage/{test}.delme')
    #data = [s.copy() for s in ssdata[test]]
    score = runscore(params,ssdata)
    for scorename, value in zip('label shilouette batchmix'.split(), score):
        nuvalues = {'dataset':test,f'score':value, f'test':scorename, f'algo':f'lucy'}
        nuvalues.update(task[0])
        r.append(nuvalues)
    return r

def evalscores(tasks,incumbents, datapath = f''):

    # r = []
    # for task, params in zip(tasks,incumbents):
    #     for test in task[2]:
    #         r+= _evalhelper(task,params,test):

    r = ut.xmap(_eval, [ (t,p,id, datapath) for (t,p) in zip(tasks, incumbents) for id in t[2]])
    return Flatten(r)



def loadmnn(test, datapath = f''):
    ssdata = ut.loadfile(f'{datapath}garbage/{test}.delme')
    return  (test, scores(domnn(ssdata)))

def getmnndicts(tasks,incumbents, all_tests, datapath = f''): # i could calculate all_tests from  the tasks, but its easier if i just pass it
    rmnn = lambda x: loadmnn(x,datapath)
    scores=ut.xmap(loadmnn, all_tests)
    scores = dict(scores)

    print(f"scoring with mnn might destroy ss... i should run this last or copy.")

    r = []
    for task in tasks:
        for test in task[2]:
            score = scores[test]
            for scorename, value in zip('label shilouette batchmix'.split(), score):
                nuvalues = {'dataset':test,f'score':value, f'test':scorename, f'algo':f'mnn'}
                nuvalues.update(task[0])
                r.append(nuvalues)
    return r


def plot_grouped(data):
    '''
    mostly put this here as example... probably better to run in jupyter
    '''
    import seaborn as sns
    import pandas as pd
    from matplotlib import pyplot as plt

    data2 = pd.DataFrame(data+mnndata)
    data2["rank"] = data2.groupby(["test", 'silhouette', 'batchmix'])["score"].rank(method="dense", ascending=True)
    g = sns.FacetGrid(data2, col="silhouette", row="batchmix" )
    g.map_dataframe(sns.barplot, x="test", y = 'rank', hue = 'algo', palette = 'husl')
    plt.legend()
    plt.show()

