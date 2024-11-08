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

    embedding = umapwrap.umap_last_experiment(adata,matrix, n_components=10)

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
    datasets += list(scalp.data.timeseries(tspath,  maxdatasets=batches,  maxcells = maxcells )) if scib else []

    fnames= []
    for i,s in enumerate(datasets):
        fname = uuid.uuid4().hex
        ut.dumpfile(s, f'garbage/{fname}.delme')
        fnames.append(fname)

    return fnames


def data_to_params(tasks, data):
    for t in tasks:
        for d in data:
            t2= dict(t)
            t2['dataset'] = d
            yield t2


if __name__ == '__main__':

    # we make some dataset- ids
    # ds_ids =  experiment_setup(batches = 5 , scib = True)
    ds_ids = ['bbbd347dc44a47ed83bfd0cf58fd3e70', '6585a6689e35429bb5f4de93e56edbff', 'a4a801d88f004f34866167b118edb1dc', '01ac26cf8f7a4d7eb5c81f5b6cf07660', '3217529d9eb546e5a9390cf1cfe0da0c', '8adcaf34689e4c40a28cdf30ced6cd36', '9bbac99788474db7b33d2141b9dee733', 'f36b99f9627b4e36b759c511c42606d9', '5c261e9b7043435c981a49ae3f064b41', 'd6f871342a9944b8be3ba76b74abbadf', '5cd61c7e9f6d4ad4acc3e44f08d81482']

    # for ds_id in ds_ids:
    #     adata = ut.loadfile(f'garbage/{ds_id}.delme')
    #     print(adata)
    #     exit()


    space3 = ho.spaceship(scalp.mkgraphParameters)
    tasks = [space3.sample() for i in range(4)]
    tasks = list(data_to_params(tasks, ds_ids))
    df = opti.gridsearch(eval_single, space3, tasks = tasks ,data= [])


    df.to_csv('out.csv', index=False)





