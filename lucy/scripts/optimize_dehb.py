import numpy as np
from dehb import DEHB
import structout as so
from lmz import *
from lucy import load, adatas
from sklearn.metrics import  silhouette_score
import ubergauss.tools as ut
import time
import lucy.score as lscore
from hyperopt.pyll import scope
import wrappers
import numpy as np
from sklearn.model_selection import StratifiedKFold





datasets = load.load_scib() + load.load_timeseries(path= '/home/ubuntu/repos/cellsaw/notebooks/')

debug = True
if not debug:
    ssdata = [[adatas.subsample(i,1000,31442)  for i in series[:10]]  for series in datasets]
    evals  = 150
else:
    ssdata = [[adatas.subsample(i,1000,31443)  for i in series[:3]]  for series in datasets]
    evals  = 5

why = np.array([0]*4 + [1]*7)
skf = StratifiedKFold(n_splits=4, random_state=None, shuffle=True)
train_test = list(skf.split(why,why))
ssdata = Map(adatas.preprocess, ssdata)

from ConfigSpace import ConfigurationSpace

cs = ConfigurationSpace({
    'intra_neigh': (10,25),
    'inter_neigh': (1,5),
    'scaling_num_neighbors' : (1,5),
    'embed_components': (4,30),
    'scaling_threshold': (.05,.95),
    'outlier_threshold':(.5,.95),
    'pre_pca':(30,50),
    'connect':(.3,1),
    'connect_ladder':(1,4)

})

def eval_single( x = 0,score_weights=[],budget = 500,**kwargs): # connect should be 0..1 , but its nice to catch errors :)
    data = [adatas.subsample_iflarger(s,num=int(budget),copy = True) for s in ssdata[x]]

    assert kwargs['connect'] < 2
    data = wrappers.dolucy(data,**kwargs)
    scores = wrappers.scores(data)
    return -np.dot(scores, score_weights)


def target_function(x, config, budget, **kwargs):

    # if budget is None:
    #     max_budget = kwargs["max_budget"]
    #     budget = max_budget

    start = time.time()
    kwargs =  config.get_dictionary()
    assert kwargs['connect'] < 2
    mix, shape, (train,test) = x
    print(f"{budget=}")
    score =  np.mean(Map(eval_single, train ,budget=budget, score_weights = [1,shape,mix],**kwargs))


    result = {
        "fitness": score,
        "cost":  time.time() - start,
        "info": {
            # "test_score": test_accuracy, we could cheat and calc this also.. hmm
            "budget": budget
        }
    }

    return result


def optimize(x):

    def target(*z,**y):
        return target_function(x,*z,**y)
    print(f"start opti")

    dehb = DEHB(
        f=target,
        dimensions=len(cs.get_hyperparameters()),
        cs=cs,
        min_budget=100,
        max_budget=1000,
        output_path="./temp",
        n_workers=1        # set to >1 to utilize parallel workers
    )

    trajectory, runtime, history = dehb.run(
        total_cost=3600, verbose=True,
    )
    breakpoint()
    dehb.get_incumbents()




tasks = [(mix,shape,tt) for mix in [.4,.6,.8] for shape in [2,4,6,8] for tt in train_test]
optimize(tasks[5])

# params  = ut.xmap(optimize,tasks,n_jobs = len(tasks))
# # we use this later for the eval
# def runscore(params, test_id):
#     data = wrappers.dolucy(ssdata[test_id],**params)
#     scores = wrappers.score(data)
#     return scores
