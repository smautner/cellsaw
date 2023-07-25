from lmz import Map,Zip,Filter,Grouper,Range,Transpose, Flatten
import lucy.score as lscore
from sklearn.metrics import  silhouette_score
import scanpy as sc
import ubergauss.tools as ut
import numpy as np


def scores(data, projectionlabel = 'lsa'):
    y = data.obs['label'].tolist()
    ybatch = data.obs['batch'].tolist()
    sim = data.obsm[projectionlabel]

    score = lscore.neighbor_labelagreement(sim,y,5)
    silou = silhouette_score(sim,y)
    batchmix = -lscore.neighbor_labelagreement(sim,ybatch,5)
    return score, silou, batchmix


def scores_paired(data, projectionlabel = 'lsa'):

    '''
    assume we have a timeseries dataset, we now report the mean of scores with a windowsize of 2
    '''


    y = np.array(data.obs['label'].tolist())
    ybatch = np.array(data.obs['batch'].tolist())
    projection = data.obsm[projectionlabel]

    batches = np.unique(ybatch)
    # breakpoint()
    selectors = [ np.logical_or(ybatch == batches[i] , y == batches[i+1] ) for i in range(len(batches) -1)  ] # selects (adjacent) pairs of batches

    score = np.mean([lscore.neighbor_labelagreement(projection[s],y[s],5) for s in selectors])
    silou = np.mean([silhouette_score(projection[s],y[s]) for s in selectors])
    batchmix = np.mean([-lscore.neighbor_labelagreement(projection[s],ybatch[s],5) for s in selectors ])

    return score, silou, batchmix



###########
# lucy wrapper..
###############

from lucy import load, adatas


def dolucy( data ,intra_neigh=10,inter_neigh=5, scaling_num_neighbors=1,embed_components=5,outlier_threshold = .75,
          scaling_threshold = .25,  pre_pca = 30, connect = 1231, nalg = 0,use_ladder= 0,connect_ladder = 1,
           **kwargs): # connect should be 0..1 , but its nice to catch errors :)
    '''
    this does our embedding,
    written such that the optimizer can do its thing
    <=> do mnn
    '''
    # hyperopt best sucks because it will give me floats ...
    intra_neigh, inter_neigh, scaling_num_neighbors, embed_components, pre_pca, use_ladder, connect_ladder = Map(int, [intra_neigh, inter_neigh, scaling_num_neighbors, embed_components, pre_pca, use_ladder, connect_ladder ])
    assert connect < 1.1, "parameters were not passed.. :)"

    data = adatas.pca(data,dim = pre_pca, label = 'pca')
    # if data[0].uns['timeseries']:
    if use_ladder:
        dataset_adjacency = adatas.embed.make_adjacency(adatas.similarity(data), nalg, connect)
    else:
        dataset_adjacency = adatas.embed.make_sequence(adatas.similarity(data),  connect_ladder)

    lsa_graph = adatas.to_linear_assignment_graph(data,base = 'pca',
                                              intra_neigh = intra_neigh,
                                              inter_neigh = inter_neigh,
                                              scaling_num_neighbors = scaling_num_neighbors,
                                              outlier_threshold = outlier_threshold,
                                              scaling_threshold = scaling_threshold,
                                              dataset_adjacency =  dataset_adjacency)

    data = adatas.graph_embed(data,lsa_graph,n_components = embed_components, label = 'lsa')
    data = adatas.stack(data)
    return data, lsa_graph




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






# just format a task like this: [{deescription dict for the optimizer}, train-instances, test-instances]

def _eval(x):
    task, params, test , datapath= x
    r=[]
    ssdata = ut.loadfile(f'{datapath}garbage/{test}.delme')
    #data = [s.copy() for s in ssdata[test]]


    data = dolucy(ssdata,**params)
    score = scores(data)
    score_paired = scores_paired(data)

    for scorename, value, sc in zip('label shilouette batchmix'.split(), score, score_paired):
        nuvalues = {'dataset':test,f'score':value, f'test':scorename, f'algo':f'lucy'}
        nuvalues.update({f"pair score":sc})
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
    mnn = domnn(ssdata)
    return  test, scores(mnn), scores_paired(mnn)

def getmnndicts(tasks,incumbents, all_tests, datapath = f''): # i could calculate all_tests from  the tasks, but its easier if i just pass it
    rmnn = lambda x: loadmnn(x,datapath)
    ds_score_pairscore = ut.xmap(rmnn, all_tests)

    ds_score_pairscore  = {s[0]:s for s in ds_score_pairscore}


    print(f"scoring with mnn might destroy ss... i should run this last or copy.")
    r = []
    for task in tasks:
        for test in task[2]:
            _, score, pairscore = ds_score_pairscore[test]

            for scorename, value, sc2 in zip('label shilouette batchmix'.split(), score, pairscore):
                nuvalues = {'dataset':test,f'score':value, f'test':scorename, f'algo':f'mnn', f'pair score': sc2}
                nuvalues.update(task[0])
                r.append(nuvalues)
    return r


def loadresults(path= f''):
    results = ut.loadfile(f'{path}params2.delme')
    results,trials = Transpose(results)
    tasks = ut.loadfile(f'{path}lasttasks.delme')
    return results, tasks

def evaluate(path= f'',numds = 4, mnn = True, lucy = True):
    results, tasks  = loadresults(path)
    #  print what we need to make the pandas table
    pandasdict =  getmnndicts(tasks, 0, Range(numds), datapath = path)  if mnn else []
    pandasdict += evalscores(tasks, results, datapath = path) if lucy else[]
    return  pandasdict

