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


if __name__ == '__main__':

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




