from lmz import Map,Zip,Filter,Grouper,Range,Transpose, Filter
import numpy as np
from dehb import DEHB
import structout as so
from multiprocessing import freeze_support
from lucy import load, adatas
from sklearn.metrics import  silhouette_score
import ubergauss.tools as ut
import time
import lucy.score as lscore
from hyperopt.pyll import scope
import wrappers
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold
from ConfigSpace import ConfigurationSpace




cs = ConfigurationSpace({
    'intra_neigh': (10,26), # intrange a..b-1
    'inter_neigh': (1,6),
    'scaling_num_neighbors' : (1,6),
    'embed_components': (4,31),
    f'use_ladder': (0,2),
    'scaling_threshold': (.05,.95),
    'outlier_threshold':(.5,.95),
    'pre_pca':(30,51),
    'connect':(.3,1),
    'connect_ladder':(1,5)
})






def eval_single( ss_id = 0, score_weights=[], budget = 500,**kwargs):
    ssdata = ut.loadfile(f'garbage/{ss_id}.delme')
    data = [adatas.subsample_iflarger(s,num=int(budget),copy = True) for s in ssdata]
    assert kwargs['connect'] < 2
    data = wrappers.dolucy(data,**kwargs)
    scores = wrappers.scores(data)
    return -np.dot(scores, score_weights)


def target_function(task, config, budget, **kwargs):
    start = time.time()
    kwargs =  config.get_dictionary()
    assert kwargs['connect'] < 2
    weights = [task[0][s] for s in f'label silhouette batchmix'.split()]
    print(f"{budget=}")
    score =  np.mean(Map(eval_single, task[1] ,budget=budget, score_weights = weights,**kwargs))

    result = {
        "fitness": score,
        "cost":  time.time() - start,
        "info": {
            # "test_score": test_accuracy, we could cheat and calc this also.. hmm
            "budget": budget
        }
    }
    return result


def optimize(task):

    def target(*z,**y):
        return target_function(task,*z,**y)

    print(f"start opti")
    dehb = DEHB(
        f=target,
        dimensions=len(cs.get_hyperparameters()),
        cs=cs,
        min_budget=100,
        max_budget=1000,
        output_path="./temp",
        n_workers=32      # set to >1 to utilize parallel workers
    )

    trajectory, runtime, history = dehb.run(
        total_cost= 60*60*1, verbose=True,
    )
    bestparams = dehb.get_incumbents()[0]._values
    return bestparams


# just format a task like this: [{deescription dict for the optimizer}, train-instances, test-instances]

def experiment_setup(scib = False, ts = False, batches = 3):
    datasets = load.load_scib() if scib else []
    datasets += load.load_timeseries(path= '/home/ubuntu/repos/cellsaw/notebooks/') if ts else []

    ssdata = [[adatas.subsample(i,2000,31443)  for i in series[:batches]] for series in datasets]
    ssdata = Map(adatas.preprocess, ssdata)

    for i,s in enumerate(ssdata):
        ut.dumpfile(s, f'garbage/{i}.delme')
    return Range(ssdata)



if __name__ == '__main__':

    ds_ids = experiment_setup(scib = True, batches = 30)
    skf = KFold(n_splits=2, random_state=None, shuffle=True)

    tasks = []

    for train, test in skf.split(ds_ids):
        for batchmix in [.7,.8,.9]:
            for silhouette in [10,9,8,7]:
                tasks+= [[{f'label': 1, f'batchmix':batchmix, f'silhouette':silhouette}, train, test]]

    ut.dumpfile(tasks,f'tasks.delme')
    tasks = tasks [:12]
    results =  Map(optimize,tasks)
    print( wrappers.evalscores(tasks, results))
    ut.dumpfile(results,f'params.delme')
    breakpoint()


