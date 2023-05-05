import numpy as np
import structout as so
from lmz import *
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
from sklearn.model_selection import StratifiedKFold





datasets = load.load_scib() + load.load_timeseries(path= '/home/ubuntu/repos/cellsaw/notebooks/')
debug = False
if not debug:
    ssdata = [[adatas.subsample(i,750,31442)  for i in series[:10]]  for series in datasets]
    evals  = 150
else:
    ssdata = [[adatas.subsample(i,500,31443)  for i in series[:3]]  for series in datasets]
    evals  = 2

why = np.array([0]*4 + [1]*7)
skf = StratifiedKFold(n_splits=4, random_state=None, shuffle=True)
train_test = list(skf.split(why,why))
ssdata = Map(adatas.preprocess, ssdata)


space = {
    'intra_neigh' : scope.int(hp.quniform('intra_neigh',10,25,1)),
    'inter_neigh' : scope.int(hp.quniform('inter_neigh',1,5,1)),
    'scaling_num_neighbors' : scope.int(hp.quniform('scaling_num_neighbors',1,5,1)),
    'embed_components' : scope.int(hp.quniform('embed_components',4,30,1)),
    'scaling_threshold' : hp.uniform('scaling_threshold',.05,.95),
    'outlier_threshold' : hp.uniform('outlier_threshold',.5,.95),
    'pre_pca' : scope.int(hp.quniform('pre_pca',30,50,1)),
    'connect' : hp.uniform('connect',.3,1),
    'connect_ladder' : scope.int(hp.quniform('connect_ladder',1,4,1))
}




def eval_single( x = 0,score_weights=[],**kwargs): # connect should be 0..1 , but its nice to catch errors :)
    data = [s.copy() for s in ssdata[x]]
    data = wrappers.dolucy(data,**kwargs)
    scores = wrappers.scores(data)
    return -np.dot(scores, score_weights)


def optimize(x):
    mix, shape, (train,test) = x

    def eval_set(kwargs):
        return np.mean(Map(eval_single, train , score_weights = [1,shape,mix],**kwargs))

    trials = Trials()
    best = fmin(eval_set,
          algo=tpe.suggest,
                trials = trials,
                space = space,
                max_evals=evals)

    # print the improcement path...
    losses = trials.losses()
    so.lprint(losses)
    return best,trials

tasks = [(mix,shape,tt) for mix in [.4,.6,.8] for shape in [2,4,6,8] for tt in train_test]
params  = ut.xmap(optimize,tasks,n_jobs = len(tasks))


breakpoint()
# we use this later for the eval
def runscore(params, test_id):
    data = wrappers.dolucy(ssdata[test_id],**params)
    scores = wrappers.score(data)
    return scores
