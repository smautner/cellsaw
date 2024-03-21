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


# 'stairs':[0,1,2],
space3 = ho.spaceship('''
neighbors_total 10 100 1
neighbors_intra_fraction .05 .5
add_tree 0 1 1
copy_lsa_neighbors 0 1 1
inter_outlier_threshold 0.7 1
inter_outlier_probabilistic_removal 0 1 1
''')

def test_nya():
    p = opti.maketasks(space)[0]
    p = {'intra_neigh': 5, 'intra_neighbors_mutual': True, 'inter_neigh': 1, 'add_tree': False, 'copy_lsa_neighbors': True, 'inter_outlier_threshold': 0.8, 'pre_pca': 40, 'embed_comp': 8, 'inter_outlier_probabilistic_removal': True}
    print(p)
    did= 'b450f36d15cf4f0b921dadf2c7af7533'
    r = eval_single(did, **p)
    return r



def eval_single(**kwargs):
    ss_id = kwargs.pop('dataset')
    embed_comp = kwargs.pop('embed_comp', 8)


    batches = ut.loadfile(f'garbage/{ss_id}.delme')
    stairs = kwargs.pop('stairs',None)

    batches, matrix = scalp.mkgraph(batches, **kwargs)
                                    # dataset_adjacency= make_stairs(len(batches), Range(1,stairs+1)) if stairs else None, **kwargs)

    batches = umapwrap.graph_umap(batches,matrix,
                                  n_components=embed_comp,
                                  label = 'emb')
    batches = transform.stack(batches)

    # except:
    #     print(kwargs, ss_id)
    #     exit()

    scores = score.scores(batches,projectionlabel='emb')

    return scores

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

    datasets = data.loaddata_scib(scibpath, maxcells=maxcells, maxdatasets=batches) if scib else []
    datasets += data.loaddata_timeseries(tspath, maxcells=maxcells, maxdatasets=batches) if ts else []

    fnames= []
    for i,s in enumerate(datasets):
        fname = uuid.uuid4().hex
        ut.dumpfile(s, f'garbage/{fname}.delme')
        fnames.append(fname)

    return fnames



opts = '''
--scib bool False
--ts bool False
--test bool False
--out str results
--maxcells int 1000
--batches int 7
'''






def data_to_params(tasks, data):
    for t in tasks:
        for d in data:
            t2= dict(tasks)
            t2['dataset'] = d
            yield t2


if __name__ == '__main__':
    import  dirtyopts
    kwargs = dirtyopts.parse(opts).__dict__

    if kwargs['test']:
        test_nya()
        exit()
    else:
        kwargs.pop('test')

    # ds_ids =  experiment_setup(**kwargs)
    if kwargs['scib']:
        ds_ids=['38fe76f71bb744009a2cef28893e7823', '83dce4e7f0f94f3eb607fd76cb408c20', 'a5788a135bc642139769eb01dd42cc39', '90a17e5ceeda4329ab4e0f6cf6ee9be2']
    else:
        ds_ids=['eed0820f2f604c0e881a066fce833017', 'e35d2a29ea3a437ca92dfe06e2ee0c8d', '3eaff44b19084936b57c1bd627f4eceb', '4fe4b7cc44d5473cabfd0b07b442e10c', '39a84f031a2c4647a5e0f781459eb86e', '56f825c7978744b9910bb88e127a9005', '03e1f9cf9c0f4e8692c429554c7a67db']

    # scibNoStairs = lambda x: not (kwargs['scib'] and x['stairs'])
    # df = opti.gridsearch(evalparams, space,data= [ds_ids], taskfilter = scibNoStairs)
    # df = opti.gridsearch(evalparams,space3, tasks = [space3.sample() for i in range(60)] ,data= [ds_ids])

    tasks = [space3.sample() for i in range(60)]
    tasks = list(data_to_params(tasks, ds_ids))
    df = opti.gridsearch(eval_single, space3, tasks = tasks ,data= [])

    print(df.corr(method='spearman'))
    opti.dfprint(df)
    ut.dumpfile(df,kwargs['out'])
    print(f"{ds_ids=}")
    print(f"{kwargs=}")

    # ubergauss has a caching function somethere!
    # skf = KFold(n_splits=4, random_state=None, shuffle=True)
    # for train, test in skf.split(ds_ids):

