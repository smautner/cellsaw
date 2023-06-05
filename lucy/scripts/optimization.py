from lmz import Map,Zip,Filter,Grouper,Range,Transpose, Flatten
import structout as so
from lucy import load, adatas
from sklearn.metrics import  silhouette_score
import ubergauss.tools as ut
import time
import lucy.score as lscore
from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll import scope
from ubergauss import hyperopt as uopt
import lucy.scripts.wrappers as wrappers
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from ubergauss.hyperopt import spaceship



space = {
    'intra_neigh' : scope.int(hp.quniform('intra_neigh',10,25,1)),
    'use_ladder' : hp.choice(f'use_ladder',[0,1]),
    'inter_neigh' : scope.int(hp.quniform('inter_neigh',1,5,1)),
    'scaling_num_neighbors' : scope.int(hp.quniform('scaling_num_neighbors',1,5,1)),
    'embed_components' : 8,#scope.int(hp.quniform('embed_components',4,30,1)),
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
                max_evals=100)

    # print the improcement path...
    losses = trials.losses()
    so.lprint(losses)
    return best, trials


def experiment_setup(scib = False, ts = False, batches = 3,
                     ibpath=f'/home/stefan/benchdata/', tspath= '/home/ubuntu/repos/cellsaw/notebooks/'):
    datasets = load.load_scib() if scib else []
    datasets += load.load_timeseries(path = tspath) if ts else []

    ssdata = [[adatas.subsample(i,1000,31443)  for i in series[:batches]] for series in datasets]
    ssdata = Map(adatas.preprocess, ssdata)

    for i,s in enumerate(ssdata):
        ut.dumpfile(s, f'garbage/{i}.delme')
    return Range(ssdata)

opts = '''
--scib bool False
--ts bool False
'''

if __name__ == '__main__':
    import  dirtyopts
    #ds_ids = Range(4)
    #ds_ids = experiment_setup(scib = True, batches = 3)
    kwargs = dirtyopts.parse(opts).__dict__
    ds_ids = experiment_setup(**kwargs, batches = 10)

    skf = KFold(n_splits=4, random_state=None, shuffle=True)
    tasks = []
    for train, test in skf.split(ds_ids):
        for batchmix in [.5,.6,.7]:
            for silhouette in [10,15,20]:
                tasks+= [[{f'label': 1, f'batchmix':batchmix, f'silhouette':silhouette}, train, test]]

    results =  ut.xmap(optimize,tasks, n_jobs = len(tasks))
    ut.dumpfile(tasks,f'lasttasks.delme')
    ut.dumpfile(results,f'params2.delme')






