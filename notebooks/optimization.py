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


'''
THIS IS THE REAL OPTIMIZER :)
import optimization as opt
d= opt.getdata()
opt.arun(d, 'nu')
'''

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
    n_comp = kwargs.pop('n_comp', None)
    grap = scalp.graph.integrate(X,smartcut=False,**kwargs)
    grap = grap!=0 # way faster and better
    # proj = umap.UMAP(n_neighbors=10, n_components=2).fit_transform(grap)
    return scalp.score.getscores(grap,X.obs['label'], X.obs['batch'], 5)

import copy

# def gridsearch(func,data = None, *, param_dict = False, tasks = False, taskfilter =None,
#                score = 'score',mp = True,  df = True, param_string = False , timevar=f'time'):

def makespace():
    return ho.spaceship(scalp.graph.integrate_params)

def getdata(cells = 750, data  = 8, src= 'batch'):
    if src == 'batch':
        r  =  list(scalp.data.scib(scalp.test_config.scib_datapath, maxdatasets=data, maxcells=cells,filter_clusters=15, slow=1))
    else:
        r= list(scalp.data.timeseries(scalp.test_config.timeseries_datapath, maxdatasets=data, maxcells=cells))
    return [[rr] for rr in r]


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
    if kwargs.get('k',0) <1: return 0
    if kwargs.get('hub1_k',0) <1: return 0
    if kwargs.get('hub2_k',0) <1: return 0
    if kwargs.get('outlier_threshold',2) > 1: return 0
    kwargs.pop('config_id',None)
    for k,v in kwargs.items():
        if k.endswith('k'):
            kwargs[k] = int(v)
    r = eval_fast(data,0,**kwargs)
    return 1.5 * r['label_mean'] + r['batch_mean']



# if __name__ == '__main__': main()
space = '''
hub1_algo [0,1,2,3,4]
hub1_k 3 20 1
hub2_algo [0,1,2,3,4]
hub2_k 3 20 1
k 10 30 1
outlier_threshold .2 1
hub1_k -> hub1_algo
hub2_k -> hub2_algo
'''
from ubergauss.optimization import nutype, gatype, grid1type
import structout as so
def arun(d,typ):
    runs = 10
    if typ == 'ga':
        o= gatype.nutype(space,ho_eval, d, numsample= 32, hyperband=[])
    if typ == 'rd':
        o= nutype.nutype(space,ho_eval, d, numsample= 32*runs, hyperband=[])
        o.opti()
        o.print()
        return o.getmax()
    if typ == 'nu':
        o= nutype.nutype(space,ho_eval, d, numsample= 32, hyperband=[])

    [o.opti() for i in range(runs)]

    z= pd.concat(o.runs)
    so.lprint(z.score)
    # o.opti()
    #for i in range(5):o.opti()
    o.print()
    return o.getmax()


# 4      20          1      15  15           0.521994          0  1.776236  0.576978
# 4       9          1      10  15           0.549294          0  1.799206  0.578037
# 3      20          1      12  15           0.521006          0  1.809666  0.579953
# 2      13          0      20  11           1.000000               2.11
# 4      20          4      11  20           0.999917           2.086836



space2 = '''
neighbors_total 2 30 1
horizonCutoff [0, 5, 10, 20]
neighbors_intra_fraction .1 .9
distance_metric ['euclidean', 'cosine']
outlier_threshold .1 1
dataset_adjacency [False, True]
intra_neighbors_mutual [True, False]
copy_lsa_neighbors [True, False]
outlier_probabilistic_removal [True, False]
add_tree [True, False]
'''

def eval_old_system(data, **kwargs):
    kwargs.pop('config_id',None)
    kwargs.pop('n_comp', None)
    grap = scalp.graph.linear_assignment_integrate(
        [d.obsm['pca40'] for d in scalp.data.transform.split_by_obs(data)],
        **kwargs)
    grap = grap!=0 # way faster and better
    # proj = umap.UMAP(n_neighbors=10, n_components=2).fit_transform(grap)
    r=  scalp.score.getscores(grap,data.obs['label'], data.obs['batch'], 5)
    return r['label_mean'] * r['batch_mean']

def oldrun(d,typ):
    runs = 10
    if typ == 'ga':
        o= gatype.nutype(space2,eval_old_system, d, numsample= 32, hyperband=[])
    if typ == 'rd':
        o= nutype.nutype(space2,eval_old_system, d, numsample= 32*runs, hyperband=[])
        o.opti()
        return o.getmax()
    if typ == 'nu':
        o= nutype.nutype(space2,eval_old_system, d, numsample= 32, hyperband=[])

    [o.opti() for i in range(runs)]

    z= pd.concat(o.runs)
    so.lprint(z.score)
    # o.opti()
    #for i in range(5):o.opti()
    o.print()
    return o.getmax()

'''
OLD GA
add_tree                            False
copy_lsa_neighbors                  False
dataset_adjacency                    True
distance_metric                    cosine
horizonCutoff                           0
intra_neighbors_mutual              False
neighbors_intra_fraction         0.205861
neighbors_total                        17
outlier_probabilistic_removal       False
outlier_threshold                     1.0
config_id                              18
score                            1.887756
time                              5.25035
'''





def test_ga():
    ut.nuke()
    def example_function(data, x=None, y=None, some_boolean=None,**kwargs):
        score_from_x = - (x - 0.5)**2  # Max at x=0.5
        score_from_y = - (y - 10)**2 / 100.0 # Max at y=10
        score_from_bool = .1*some_boolean
        score_noise = np.random.normal(0, .1)
        return score_noise + score_from_x + score_from_y + score_from_bool

    example_space = """
    x 0.0 1.0
    y 1 20 1
    some_boolean [1, 0]
    """
    o = gatype.nutype(example_space,
                      example_function,
                      data=[[0]],
                      numsample=16)
    [o.opti() for _ in range(5)]
    o.print()
    # o.print_more()
