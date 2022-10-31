from lmz import Map,Zip,Filter,Grouper,Range,Transpose
import natto
import scanpy as sc
import natto
import pandas as pd
import numpy as np

def getmousecortex(subsample=1000):
    cortexfiles = ['e11', 'e13', 'e15', 'e17']
    cortexdata = [natto.input.loadCortex(subsample = subsample,
                                         pathprefix=f'/home/ubuntu/data/jack/MouseCortexData/raw{e}',
                                         batch = 1) for e in cortexfiles ]


########
# mouse cerebellumdata...
# we extracted mm and labels, and genenames from the R data
# now what?
###########

def cerebellum(who = 'raw'):
    # MC_corr.mm  MC_corr_celllabels.csv  MC_corr_genelabels.csv
    # MC_raw.mm  MC_raw_celllabels.csv  MC_raw_genelabels.csv

    d = sc.read_mtx(f"/home/ubuntu/data/MC_{who}.mm")
    d = d.T
    obs = pd.read_csv(f'/home/ubuntu/data/MC_{who}_celllabels.csv')
    d.obs['slice']=[ e[:e.find('_')] for e in obs.barcode]
    d.obs['barcode'] = obs.barcode.to_list()

    obs = pd.read_csv(f'/home/ubuntu/data/MC_barcode_cluster.csv')
    bc_clust  = {a:b for a,b in zip(obs.barcode.to_list(), obs.x.to_list())}
    d.obs['celltype'] = [bc_clust.get(e,-1) for e in d.obs['barcode'].to_list() ]

    genes = pd.read_csv(f'/home/ubuntu/data/MC_{who}_genelabels.csv')
    d.var['gene_names'] = genes.gene.to_list()
    d.write(f'/home/ubuntu/data/MC_{who}_all.h5', compression='gzip')
    return d
    #csv = pd.read_csv('/home/ubuntu/data/mousecereblabels.csv')

def loadcereb(timeNames = ['E10', 'E12', 'E14', 'E16', 'E18','P0', 'P5', 'P7', 'P14'],who='raw',subsample=None):
    d = sc.read(f'/home/ubuntu/data/MC_{who}_all.h5')
    def choose(item):
        z = d[d.obs['slice']==item]
        if subsample:
            sc.pp.subsample(z,n_obs= subsample)
        return z
    return [ choose(item) for item in timeNames]



def loadimm(subsample=1000):
    dir = '/home/ubuntu/repos/HungarianClustering/data/immune_stim/'
    return [natto.input.loadpbmc(path=dir+s, subsample=subsample) for s in '89']



def getwater(subsample=1000):
    #GSE126954
    d = sc.read_mtx(f"/home/ubuntu/data/jack/waterstone/genebycell.mm").T
    z = pd.read_csv('/home/ubuntu/data/jack/waterstone/cellannotation.csv')
    d.obs['label'] = z['cell.type'].to_list()
    d.obs['batch'] = z['batch'].to_list()
    d.write(f'/home/ubuntu/data/waterston.h5', compression='gzip')
    return z,d


def loadwater(subsample=None):
    d = sc.read(f'/home/ubuntu/data/waterston.h5')

    def choose(item):
        z = d[d.obs['batch']==item]
        if subsample:
            sc.pp.subsample(z,n_obs= subsample)
        return z
    names = np.unique(d.obs['batch'])
    return [ choose(item) for item in names], names




def pancreatic(subsample=1000):
    inputDirectory = "/home/ubuntu/data/jack/GSE101099_data/GSM269915"
    def load(item):
        anndata = natto.input.loadGSM(f'{inputDirectory}{item}/',
                                      subsample=subsample,
                                      cellLabels=True,
                                      labelFile='theirLabels.csv')
        nonBloodGenesList = [name for name in anndata.var_names if (not name.startswith('Hb') and not name.startswith('Ft'))]
        anndata= anndata[:, nonBloodGenesList]
        if subsample:
            sc.pp.subsample(anndata,n_obs=subsample)
        return anndata
    return [load(x) for x in ['6_E12_B2', '5_E14_B2', '7_E17_B2']]


def pancv2(subsample=1000):
    inputDirectory = "/home/ubuntu/data/jack/GSE101099_data/GSM314091"
    def load(item):
        anndata = natto.input.loadGSM(f'{inputDirectory}{item}/', subsample=subsample, cellLabels=True)
        if subsample:
            sc.pp.subsample(anndata,n_obs=subsample)
        return anndata
    return [load(x) for x in ['5_E12_v2', '6_E14_v2', '7_E17_1_v2', '8_E17_2_v2']]



def centers(arg):
    X,y = arg
    cents = []
    for i in np.unique(y):
        cents.append(X[y==i].mean(axis=0))
    return np.array(cents)

def getcenters(xx,yy):
    return np.vstack(Map(centers,Zip(xx,yy)))

from sklearn.neighbors import NearestNeighbors as nn
from sklearn.metrics.pairwise import euclidean_distances as ed

def score(m,proj,labels):
    y = [d.obs[labels] for d in m.data]
    m1 = ed(getcenters(m.projections[1],y))
    m2 = ed(getcenters(proj,y))

    from scipy.stats import spearmanr
    return np.mean([spearmanr(m11,m22) for m11,m22 in zip(m1,m2)])




