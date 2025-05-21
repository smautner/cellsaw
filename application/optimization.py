import umap
from lmz import Map,Zip,Filter,Grouper,Range,Transpose, Flatten
import ubergauss.tools as ut
import numpy as np
from scalp.data import transform
from scalp import umapwrap
import scalp
from scalp.output import score
from ubergauss import optimization as opti
from scalp import data, test_config
from warnings import simplefilter
from scalp.data.similarity import make_stairs
import ubergauss.hyperopt as ho
simplefilter(action='ignore', category=FutureWarning)


def eval_single(**kwargs):
    args = dict(kwargs)
    ss_id = kwargs.pop('dataset')
    adata = ut.loadfile(f'garbage/{ss_id}.delme')
    matrix = scalp.mkgraph(adata, **kwargs)

    embedding = umapwrap.umap_last_experiment(adata,matrix, n_components=kwargs['dim'])

    adata.obsm['scalp'] = embedding
    adata.uns['integrated'] = ['scalp']
    scores = score.scalp_scores(adata)['scalp']

    args.update(scores)
    return args

# def evalparams(dataids, **params):
#     scores =  np.mean(Map(eval_single, dataids,  **params), axis = 0)
#     return dict(zip('class_cohesion silhouette batch_cohesion'.split(),scores))

import uuid
def experiment_setup(scib = False, ts = False,
                     batches = 10,
                     scibpath=False,
                     maxcells = 1000,
                     tspath= False, **kwargs):
    scibpath = scibpath or test_config.scib_datapath
    tspath = tspath or test_config.timeseries_datapath

    datasets = list(scalp.data.scib(scibpath,  maxdatasets=batches,  maxcells = maxcells )) if scib else []
    datasets += list(scalp.data.timeseries(tspath,  maxdatasets=batches,  maxcells = maxcells )) if ts else []

    fnames= []
    for i,s in enumerate(datasets):
        fname = uuid.uuid4().hex
        ut.dumpfile(s, f'garbage/{fname}.delme')
        fnames.append(fname)
    print(fnames)
    return fnames


def data_to_params(tasks, data):
    for t in tasks:
        for d in data:
            t2= dict(t)
            t2['dataset'] = d
            yield t2


def old():
    # we make some dataset- ids
    ds_ids =  experiment_setup(batches = 4,maxcells = 1000, scib = True)
    # ds_ids = ['b66774d7f379482d9bcc60e32fc961b5', '58fef6d785de49dd8dd75f714f1c1fb4', 'e9e3df848cf84aeea65bd9905e29ae4c', '0fbd407bbf384998b23748300e75146d']
    # for ds_id in ds_ids:
    #     adata = ut.loadfile(f'garbage/{ds_id}.delme')
    #     print(adata)
    #     exit()
    space3 = ho.spaceship(scalp.mkgraphParameters+'dim 10 15 1\n')
    tasks = [space3.sample() for i in range(300)]
    tasks = list(data_to_params(tasks, ds_ids))
    df = opti.gridsearch(eval_single, space3, tasks = tasks ,data= [],mp=True)
    df.to_csv('out.csv', index=False)

import pandas as pd


def eval_fast(X,y, **kwargs):
    grap = scalp.graph.integrate(X, **kwargs)
    grap = grap!=0
    proj = umap.UMAP(n_neighbors=10).fit_transform(grap)
    return scalp.score.getscores(proj,X.obs['label'], X.obs['batch'], 5)

import copy

# def gridsearch(func,data = None, *, param_dict = False, tasks = False, taskfilter =None,
#                score = 'score',mp = True,  df = True, param_string = False , timevar=f'time'):

def makespace():
    return ho.spaceship(scalp.graph.integrate_params)

def getdata():
    return  list(scalp.data.scib(scalp.test_config.scib_datapath, maxdatasets=4, maxcells=500))

def main():
    # load data.. first a little bit to check!
    datasets  =  getdata()
    # make my spaceship :
    space3 = makespace()
    tasks = [space3.sample() for i in range(500)]

    # df = [opti.gridsearch(eval_fast, data= (ds,0),  tasks =copy.deepcopy( tasks) ,mp=True) for ds in datasets]
    # df = pd.concat(df)
    df = opti.gridsearch(ho_eval, data_list= [datasets],  tasks =copy.deepcopy( tasks) ,mp=True)
    df.to_csv('out_small.csv', index=False)


def hyp():
    x = getdata()
    s = makespace()
    tr = ho.run(x,ho_eval,space = s, max_evals = 100)
    breakpoint()

def ho_eval(data, **kwargs):
    r = [eval_fast(d,0,**kwargs) for d in data]
    v = 1
    for di in r:
        v *= di['label_mean']
        v *= di['batch_mean']
    return v

if __name__ == '__main__':
    main()

