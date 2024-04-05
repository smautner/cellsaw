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
    dataset = scalp.data.mkblobs(sizes = [[40,24,21],[40,25,20],[40,23,22]],
                                 clusterspread = .5, batchspread = 1.5)
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
             'inter_outlier_threshold': .9,
             'inter_outlier_probabilistic_removal': False}
    dataset, graph = scalp.mkgraph(dataset,**parm)
    scalp.umapwrap.graph_umap(dataset,graph,label = 'umap', n_components = 2)
    return dataset

def heatmap(df, val = 'lin'):
    '''
    - pivot targets
    - plot
    '''
    sns.heatmap(df.pivot(index = 'cspread', columns = 'bspread', values= val))
    plt.show()


from ubergauss import optimization as uo
def run_scores():
    def f(_,cspread=0,bspread=0):
        dataset = scalp.data.mkblobs(sizes = [[40,24,21],[40,25,20],[40,23,22]],
                                     clusterspread = cspread,
                                     batchspread = bspread)
        dataset = runscalp(dataset)
        dataset = scalp.transform.stack(dataset)
        return {'batch' : score.score_lin_batch(dataset),
                'lin': score.score_lin(dataset)}

    pdict = {'cspread' :[.5,.75], 'bspread' :[.5,2,8]}

    df = uo.gridsearch(f,param_dict = pdict, data = [1])
    heatmap(df, 'lin')
    heatmap(df,'batch')

import pandas as pd
def pareto(df):
    df = df.melt(id_vars = ['cspread','bspread'],
                 value_vars=['batch','lin'],
                 var_name='target', value_name='score')
    nus = uo.pareto_scores(df,method = 'cspread bspread'.split(' '), data = None)
    stuff  = [dict(zip('cspread bspread score'.split(' '),[c,b,score])) for (c,b),score in nus]
    stuff = pd.DataFrame(stuff)
    heatmap(stuff,'score')
    return stuff


