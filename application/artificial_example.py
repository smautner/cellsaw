from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten

import matplotlib
matplotlib.use('module://matplotlib-backend-sixel')
import matplotlib.pyplot as plt

import scalp
from scalp.output import draw
import lmz
import numpy as np
from scipy.sparse import csr_matrix
import scanpy as sc
import seaborn as sns
from scipy.sparse.csgraph import dijkstra
from scalp import graph as sgraph
import pacmap
from scalp.output import score


def scplot(dataset):
    stack = scalp.transform.stack(dataset)
    sc.pl.umap(stack, color=['batch', 'label'], s= 80)
    plt.show()

def makedata():
    # dataset = scalp.data.mkblobs(sizes = [[24,21],[25,20],[23,22]], clusterspread = .3, batchspread = 4)
    dataset = scalp.data.mkblobs(sizes = [[40,24,21],[40,25,20],[40,23,22]], clusterspread = .5, batchspread = 1.5)
    #dataset = scalp.data.mkblobs(sizes = [[3,5],[5,3]], clusterspread = .1, batchspread = 4)
    # for d in dataset: d.obsm['umap'] = d.X
    scplot(dataset)
    return dataset

def runscalp(dataset):
    parm = {'neighbors_total': 30,
             'neighbors_intra_fraction': .3,
             'add_tree': False,
             'epsilon'  : 1e-4,
              'copy_lsa_neighbors': False,
             'inter_outlier_threshold': .5,
             'inter_outlier_probabilistic_removal': False}
    dataset, graph = scalp.mkgraph(dataset,**parm)
    scalp.umapwrap.graph_umap(dataset,graph,label = 'umap', n_components = 2)
    return dataset

def heatmap(df, val = 'lin'):
    sns.heatmap(df.pivot(index = 'cspread', columns = 'degree', values= val))
    plt.show()


from ubergauss import optimization as uo
def run_scores():
    def f(_,cspread=0,bspread=1,degree = 0):

        dataset = scalp.data.mkblobs(sizes =
                                     [[40,24,21],[20,45,20],[20,23,42]],
                                     clusterspread = cspread,
                                     batchspread = bspread)
        dataset[1]=rotate(dataset[1],degree=degree)
        dataset = runscalp(dataset)
        dataset = scalp.transform.stack(dataset)
        return {'batch' : score.score_lin_batch(dataset),
                'lin': score.score_lin(dataset)}

    pdict = {'cspread' :np.arange(.1,1,.1), 'degree' :np.arange(0, 90, 10 )}

    df = uo.gridsearch(f,param_dict = pdict, data = [1])

    heatmap(df,'batch')
    heatmap(df, 'lin')
    return df

import pandas as pd
def pareto(df):
    df = df.melt(id_vars = ['cspread','degree'],
                 value_vars=['batch','lin'],
                 var_name='target', value_name='score')
    nus = uo.pareto_scores(df,method = 'cspread degree'.split(' '), data = None)
    stuff  = [dict(zip('cspread degree score'.split(' '),[c,b,score])) for (c,b),score in nus]
    stuff = pd.DataFrame(stuff)
    heatmap(stuff,'score')
    return stuff

'''
we shift one batch to the right and see how the score is affected
-> just shifting a batch sideways doesnt break the hungarian
'''
def shift_test():
    dataset = scalp.data.mkblobs(sizes = [[40,24,21],[40,25,20],[40,23,22]],
                                 clusterspread = .5, batchspread = 6)

    runscalp(dataset)
    scplot(dataset)

    dataset[1].X[:,0] += 40
    scplot(dataset)
    runscalp(dataset)
    scplot(dataset)

'''
test tehe rotation
'''
def rotate(anndata, degree=90):
    center = anndata.X.mean(axis=0)
    anndata.X -= center
    angle = degree/180*np.pi
    rot = np.array([[np.cos(angle),-np.sin(angle)],
                    [np.sin(angle),np.cos(angle)]])
    anndata.X = anndata.X @ rot
    anndata.X+= center
    anndata.obsm['umap']= anndata.X
    return anndata


def rotation_test():
    dataset = scalp.data.mkblobs(sizes = [[40,24,21],[40,25,20],[40,23,22]],
                                 clusterspread = .5, batchspread = 6)
    dataset[1] = rotate(dataset[1])
    scplot(dataset)




