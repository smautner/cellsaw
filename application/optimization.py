from lmz import Map,Zip,Filter,Grouper,Range,Transpose, Flatten
import ubergauss.tools as ut
import numpy as np
from scalp.data import transform
from scalp import umapwrap

from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)



space= {
        'intra_neigh':[5,10,15],
        'intra_neighbors_mutual':[ True, False],
        'inter_neigh':[1,2,3],
        'add_tree':[True,False],
        'copy_lsa_neighbors':[ True,False],
        'inter_outlier_threshold':[ None, .7, .8 , .9],
        'pre_pca' : [40],
        'embed_comp' : [8],
        'inter_outlier_probabilistic_removal':[True,False]}

import scalp
from scalp.output import score
from ubergauss import optimization as opti
from scalp import data, test_config


def test_nya():
    p = opti.maketasks(space)[0]
    print(p)
    did=2
    r = eval_single(did, **p)
    return r



def eval_single(ss_id = 0,
                score_weights=[1,10,.7],
                **kwargs):

    ssdata = ut.loadfile(f'garbage/{ss_id}.delme')
    batches = [data.subsample_iflarger(s, num=100, copy = False) for s in ssdata]

    embed_comp = kwargs.pop('embed_comp')
    batches, matrix = scalp.mkgraph(batches,**kwargs)

    batches = umapwrap.graph_umap(batches,matrix,n_components=embed_comp,label = 'emb')
    batches = transform.stack(batches)
    # except:
    #     print(kwargs, ss_id)
    #     exit()

    scores = score.scores(batches,projectionlabel='emb')
    # return -np.dot(scores, score_weights)
    return scores




def evalparams(dataids, **params):
    # if params['isodim'] < params['inter_neigh']+ params['intra_neigh']:
    #     return None
    # return np.mean(Map(eval_single, dataids,  **params), axis = 0)
    scores =  np.mean(Map(eval_single, dataids,  **params), axis = 0)
    return dict(zip('class_cohesion silhouette batch_cohesion'.split(),scores))


def experiment_setup(scib = False, ts = False,
                     batches = 10,
                     scibpath=False,
                     tspath= False):
    scibpath = scibpath or test_config.scib_datapath
    tspath = tspath or test_config.timeseries_datapath
    datasets = data.loaddata_scib(scibpath, maxdatasets=batches) if scib else []
    datasets += data.loaddata_timeseries(tspath, maxdatasets=batches) if ts else []
    for i,s in enumerate(datasets):
        ut.dumpfile(s, f'garbage/{i}.delme')
    return Range(datasets)



opts = '''
--scib bool False
--ts bool False
--test bool False
'''

if __name__ == '__main__':
    import  dirtyopts
    kwargs = dirtyopts.parse(opts).__dict__

    if kwargs['test']:
        test_nya()
        exit()
    else:
        kwargs.pop('test')

    ds_ids = Range(3) # experiment_setup(**kwargs, batches = None)

    df = opti.gridsearch(evalparams, space, [ds_ids])
    print(df.corr(method=f'spearman'))
    opti.dfprint(df)
    ut.dumpfile(df, 'results_lol')
    # ubergauss has a caching function somethere!

    # skf = KFold(n_splits=4, random_state=None, shuffle=True)
    # for train, test in skf.split(ds_ids):

