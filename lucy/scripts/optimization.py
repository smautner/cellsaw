from lmz import Map,Zip,Filter,Grouper,Range,Transpose, Flatten
import numpy as np
import structout as so

from lucy import load, adatas
from sklearn.metrics import  silhouette_score
import ubergauss.tools as ut
import time
import lucy.score as lscore
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll import scope
from ubergauss import hyperopt as uopt
import wrappers
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold

from ubergauss.hyperopt import spaceship


space = {
    'intra_neigh' : scope.int(hp.quniform('intra_neigh',10,25,1)),
    'use_ladder' : hp.choice(f'use_ladder',[0,1]),
    'inter_neigh' : scope.int(hp.quniform('inter_neigh',1,5,1)),
    'scaling_num_neighbors' : scope.int(hp.quniform('scaling_num_neighbors',1,5,1)),
    'embed_components' : scope.int(hp.quniform('embed_components',4,30,1)),
    'scaling_threshold' : hp.uniform('scaling_threshold',.05,.95),
    'outlier_threshold' : hp.uniform('outlier_threshold',.5,.95),
    'pre_pca' : scope.int(hp.quniform('pre_pca',30,50,1)),
    'connect' : hp.uniform('connect',.3,1),
    'connect_ladder' : scope.int(hp.quniform('connect_ladder',1,4,1))
}




space = {
    'intra_neigh' : scope.int(hp.quniform('intra_neigh',10,25,1)),
    'use_ladder' : hp.choice(f'use_ladder',[0,1]),
    'inter_neigh' : scope.int(hp.quniform('inter_neigh',1,5,1)),
    'scaling_num_neighbors' : scope.int(hp.quniform('scaling_num_neighbors',1,5,1)),
    'embed_components' : scope.int(hp.quniform('embed_components',4,30,1)),
    'scaling_threshold' : hp.uniform('scaling_threshold',.05,.95),
    'outlier_threshold' : hp.uniform('outlier_threshold',.5,.95),
    'pre_pca' : scope.int(hp.quniform('pre_pca',30,50,1)),
    'connect' : hp.uniform('connect',.3,1),
    'connect_ladder' : scope.int(hp.quniform('connect_ladder',1,4,1))
}



def eval_single(ss_id = 0,score_weights=[],**kwargs): # connect should be 0..1 , but its nice to catch errors :)
    ssdata = ut.loadfile(f'garbage/{ss_id}.delme')
    data = [adatas.subsample_iflarger(s,num=1000,copy = False) for s in ssdata]
    data = wrappers.dolucy(data,**kwargs)
    scores = wrappers.scores(data)
    scores = wrappers.scores(data)
    return -np.dot(scores, score_weights)


def optimize(task):
    train = task[1]
    weights = [task[0][s] for s in f'label silhouette batchmix'.split()]

    def eval_set(kwargs):
        return np.mean(Map(eval_single, train , score_weights = weights,**kwargs))

    trials = Trials()
    best = fmin(eval_set,
          algo=tpe.suggest,
                trials = trials,
                space = space,
                max_evals=2)

    # print the improcement path...
    losses = trials.losses()
    so.lprint(losses)
    return best


def experiment_setup(scib = False, ts = False, batches = 3, tspath= '/home/ubuntu/repos/cellsaw/notebooks/'):
    datasets = load.load_scib() if scib else []
    datasets += load.load_timeseries(path = tspath) if ts else []

    ssdata = [[adatas.subsample(i,2000,31443)  for i in series[:batches]] for series in datasets]
    ssdata = Map(adatas.preprocess, ssdata)

    for i,s in enumerate(ssdata):
        ut.dumpfile(s, f'garbage/{i}.delme')
    return Range(ssdata)



if __name__ == '__main__':

    ds_ids = Range(4)#experiment_setup(scib = True, batches = 30)
    skf = KFold(n_splits=4, random_state=None, shuffle=True)
    tasks = []
    for train, test in skf.split(ds_ids):
        for batchmix in [.7,.8,.9]:
            for silhouette in [10,9,8,7]:
                tasks+= [[{f'label': 1, f'batchmix':batchmix, f'silhouette':silhouette}, train, test]]

    # tasks = tasks [:12]
    results =  ut.xmap(optimize,tasks, n_jobs = len(tasks))
    ut.dumpfile(results,f'params.delme')
    print( wrappers.evalscores(tasks, results))
    breakpoint()












# def evalscores(tp):
#     task,param = tp
#     params = param[0]
#     params = {k:p if len(str(p)) > 4 else int(p) for k,p in params.items()}
#     r = []
#     for test in task[2][1]:
#         scores = runscore(params,test)
#         for scorename,value in zip('label shilouette batchmix'.split(), scores):
#             r.append({f'mix':task[0],f'shape':task[1], f'dataset':test,f'score':value, f'test':scorename, f'algo':f'lucy'})
#     return r

# #we use this later for the eval
# def runscore(params, test_id):
#     data = wrappers.dolucy(ssdata[test_id],**params)
#     scores = wrappers.scores(data)
#     return scores

# # r= flatten(r)
# # print(r)
# # params,tasks, ssdata =  ut.loadfile( 'opt1.delme2')
# # r = ut.xmap( evalscores , Zip(tasks, params ))

# def getmnndicts(id):
#     data = wrappers.domnn(ssdata[id])
#     scores = wrappers.scores(data)
#     r = []
#     for mix in [.4,.6,.8]:
#         for shape in [2,4,6,8]:
#             for scorename,value in zip('label shilouette batchmix'.split(), scores):
#                 r.append({f'mix':mix,f'shape':shape, f'dataset':id,f'score':value, f'test':scorename, f'algo':f'mnn'})
#     return r
# r = ut.xmap( getmnndicts , Range(ssdata))
# print(r)
# breakpoint()

