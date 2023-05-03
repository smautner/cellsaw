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
debug = True


datasets = load.load_scib() + load.load_timeseries(path= '/home/ubuntu/repos/cellsaw/notebooks/')

if not debug:
    ssdata = [[adatas.subsample(i,750,31442)  for i in series[:10]]  for series in datasets]
else:
    ssdata = [[adatas.subsample(i,500,31443)  for i in series[:3]]  for series in datasets]
# ssdata = Map(adatas.preprocess, ssdata)

if debug:
    probing = 0
    after_evals =2
else:
    probing = 10
    after_evals = 50


def old_eval_delme( x = 0,intra_neigh=10,inter_neigh=5, scaling_num_neighbors=1,embed_components=5,outlier_threshold = .75,
          scaling_threshold = .25,  pre_pca = 30, connect = 1231, nalg = 0, connect_ladder = 1): # connect should be 0..1 , but its nice to catch errors :)
    data = [z.copy() for z in ssdata[x]]
    data = adatas.pca(data,dim = pre_pca, label = 'pca')
    if data[0].uns['timeseries']:
        dataset_adjacency = adatas.embed.make_adjacency(adatas.similarity(data), nalg, connect)
    else:
        dataset_adjacency = adatas.embed.make_sequence(adatas.similarity(data),  connect_ladder)

    lsa_graph = adatas.lapgraph(data,base = 'pca',
                                              intra_neigh = intra_neigh,
                                              inter_neigh = inter_neigh,
                                              scaling_num_neighbors = scaling_num_neighbors,
                                              outlier_threshold = outlier_threshold,
                                              scaling_threshold = scaling_threshold,
                                              dataset_adjacency =  dataset_adjacency)#(adatas.similarity(data), connect, nalg)) # adjacency_matrix -> symmetrise and binarize
    data = adatas.graph_embed(data,lsa_graph,n_components = embed_components, label = 'lsa')
    data = adatas.stack(data)
    y = data.obs['label'].tolist()
    ybatch = data.obs['batch'].tolist()
    sim = data.obsm['lsa']
    score = lscore.neighbor_labelagreement(sim,y,5)+\
            4*silhouette_score(sim,y)-.5*lscore.neighbor_labelagreement(sim,ybatch,5)
    return -score


def evalp( x = 0,score_weights=[],**kwargs): # connect should be 0..1 , but its nice to catch errors :)
    data = [z.copy() for z in ssdata[x]]
    data = adatas.pca(data,dim = pre_pca, label = 'pca')
    data = wrappers.dolucy(data,**kwargs)
    scores = wrappers.score(data)
    return -np.dot(scores, score_weights)




space3 = {
    'intra_neigh' : scope.int(hp.quniform('intra_neigh',10,25,1)),
    'inter_neigh' : scope.int(hp.quniform('inter_neigh',1,5,1)),
    'scaling_num_neighbors' : scope.int(hp.quniform('scaling_num_neighbors',1,5,1)),
    'embed_components' : scope.int(hp.quniform('embed_components',4,30,1)),
    'scaling_threshold' : hp.uniform('scaling_threshold',.05,.95),
    'outlier_threshold' : hp.uniform('outlier_threshold',.5,.95),
    'pre_pca' : scope.int(hp.quniform('pre_pca',30,50,1)),
    'connect' : hp.uniform('connect',.3,1)
}


# trials = uopt.fffmin(evalp,
#                      items= Range(ssdata),
#                      probing_evals = 40,   # 50
#                      probing_parallel = 3, # 3
#                      after_evals =  40 , space=space3) # 200


def evalp_v(x=0, **kwargs):
    return np.mean(Map(evalp,Range(ssdata),**kwargs))

def evalp_h(kwargs):
    partial = lambda x: evalp(x,**kwargs)
    return np.mean(ut.xmap(partial,Range(ssdata)))



trials = uopt.fffmin2(evalp_v,evalp_h,
                     probing_evals = probing,   # 50
                     probing_parallel = 32, # 3
                     after_evals =  after_evals , space=space3) # 200

# print the improcement path...
losses = trials.losses()
so.lprint(losses)



# for trial in trials:
#     df = uopt.trial2df(trial)
#     eeeh = np.argpartition(df['loss'], 10)[:10]
#     print(df.iloc[eeeh])



import numpy as np
from sklearn.model_selection import StratifiedKFold
X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([0, 0, 1, 1])
why = np.array([0]*4 + [1]*7)
skf = StratifiedKFold(n_splits=4, random_state=None, shuffle=True)
for i, (train_index, test_index) in enumerate(skf.split(why, why)):
    print(f"{ i=}")
    print(f"{ test_index=}")
    print(f"{ train_index=}")
    print(f"{len(ssdata)=}")


