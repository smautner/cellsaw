from lmz import Map,Zip,Filter,Grouper,Range,Transpose
import natto
import scanpy as sc
import natto
import pandas as pd
import numpy as np

'''
there are loaders for timeseries data
'''


dataset_name_field ='shortname'



def getmousecortex(subsample=1000):
    cortexfiles = ['e11', 'e13', 'e15', 'e17']
    cortexdata = [natto.input.loadCortex(subsample = subsample,
                                         pathprefix=f'/home/ubuntu/data/jack/MouseCortexData/raw{e}',
                                         batch = 1) for e in cortexfiles ]

    for d,name in zip(cortexdata,cortexfiles):
        d.uns[dataset_name_field]=name
    return cortexdata


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

def loadcereb(timeNames = ['E10', 'E12', 'E14', 'E16', 'E18','P0', 'P5', 'P7', 'P14'],
              who='raw',subsample=1000):
    d = sc.read(f'/home/ubuntu/data/MC_{who}_all.h5')
    def choose(item):
        z = d[d.obs['slice']==item]
        if subsample:
            sc.pp.subsample(z,n_obs= subsample)
            z.uns[dataset_name_field] = item
        return z
    return [ choose(item) for item in timeNames]



def loadimm(subsample=1000):
    dir = '/home/ubuntu/repos/HungarianClustering/data/immune_stim/'
    r =  [natto.input.loadpbmc(path=dir+s, subsample=subsample)
                    for s in '89']

    for d,name in zip(r,'89'):
        d.uns[dataset_name_field]= f'immune_{name}'

    return r


def getwater(subsample=1000):
    #GSE126954
    d = sc.read_mtx(f"/home/ubuntu/data/jack/waterstone/genebycell.mm").T
    z = pd.read_csv('/home/ubuntu/data/jack/waterstone/cellannotation.csv')
    d.obs['label'] = z['cell.type'].to_list()
    d.obs['batch'] = z['batch'].to_list()
    d.write(f'/home/ubuntu/data/waterston.h5', compression='gzip')
    return z,d


def loadwater(subsample=1000):
    d = sc.read(f'/home/ubuntu/data/waterston.h5')

    def choose(item):
        z = d[d.obs['batch']==item]
        if subsample:
            z = z[z.obs['label']==z.obs['label']] # removes the nan
            sc.pp.subsample(z,n_obs= subsample)
        z.uns[dataset_name_field] = item
        return z

    names = np.unique(d.obs['batch'])

    r= [ choose(item) for item in names] #, names
    return r # r[:3]+r[5:] # removed 3rd dataset becasue it is the second batch of 2nd and is strange




def pancreatic(subsample=1000):
    inputDirectory = "/home/ubuntu/data/jack/GSE101099_data/GSM269915"
    def load(item):
        anndata = natto.input.loadGSM(f'{inputDirectory}{item}/',
                                      subsample=subsample,
                                      cellLabels=True,
                                      labelFile='theirLabels.csv')
        nonBloodGenesList = [name for name in anndata.var_names if (not name.startswith('Hb') and not name.startswith('Ft'))]
        anndata= anndata[:, nonBloodGenesList]
        anndata.uns[dataset_name_field] = item
        if subsample:
            sc.pp.subsample(anndata,n_obs=subsample)
        return anndata
    return [load(x) for x in ['6_E12_B2', '5_E14_B2', '7_E17_B2']]


def pancv2(subsample=1000):
    inputDirectory = "/home/ubuntu/data/jack/GSE101099_data/GSM314091"
    def load(item):
        anndata = natto.input.loadGSM(f'{inputDirectory}{item}/', subsample=subsample, cellLabels=True)
        anndata.uns[dataset_name_field] = item
        if subsample:
            sc.pp.subsample(anndata,n_obs=subsample)
        return anndata
    return [load(x) for x in ['5_E12_v2', '6_E14_v2', '7_E17_1_v2', '8_E17_2_v2']]


def centers(arg):
    X,y = arg
    cents = []
    for i in np.unique(y):
        m = X[y==i].mean(axis=0)
        cents.append(np.hstack([i,m]))
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





# from cellsaw import preprocess
# from lmz import *
# from cellsaw import merge
# from natto.process import kNearestNeighbours as knn
# import structout as so

loaders = [getmousecortex, loadwater, pancreatic,
               loadcereb, loadimm]
labels = ['labels','label','labels','celltype','labels']
ll = Zip(loaders, labels)
from cellsaw import similarity

# this is useful for drawing all the matrices...
# for l in jack.ll:   jack.so.heatmap(jack.getmatrix(*l).to_numpy())
def getmatrix(loader, label):
    ata = loader(subsample=1000)
    ranked_datasets_list, similarity_df = similarity.rank_by_similarity(
                                        target = ata,
                                        source = ata,
                                        method = 'meanexpression',
                                        numgenes = 500,
                                        return_similarity = True)
    return similarity_df
    #so.heatmap(similarity_df.to_numpy())


def scorematrix(m):
    error = 0
    l = m.shape[0]
    good, bad = 0,0
    for i in range(l-2):
        for j in range(i+2,l):

            if (m[i+1,j] > m[i,j]) and (m[i,j-1] > m[i,j]):
                good+=1
            else:
                bad +=1
                if (m[i+1,j] > m[i,j]):
                    error += m[i+1,j] - m[i,j]
                if (m[i,j-1] > m[i,j]):
                    error += m[i,j-1] - m[i,j]

    return good/(good+bad), error


import structout as so
def optimizescore(loader,label):
    ata = loader(subsample=1000)
    ata.pop(4)
    # for l in jack.ll:   jack.so.heatmap(jack.getmatrix(*l).to_numpy())

    scores = []
    params = []
    for method in ['meanexpression','natto','cell_ranger','seurat_v3']:
        for sim in ['jaccard','cosine' ]:
            if sim == 'jaccard':
                ng = list(range(1500,5000,200))
            else:
                ng = list(range(500,2000,100))
            for numgen in ng:
                ranked_datasets_list, similarity_df = similarity.rank_by_similarity(
                                                target = ata,
                                                source = ata,
                                                method = method,
                                                numgenes = numgen,
                                                similarity = sim,
                                                return_similarity = True)
                #so.heatmap(similarity_df.to_numpy(),legend=False)
                scores.append(scorematrix(similarity_df.to_numpy()))
                params.append(f'{method}{sim}{numgen}')
    return scores,params
    #so.heatmap(similarity_df.to_numpy())
    # for l in jack.ll:   jack.so.heatmap(jack.getmatrix(*l).to_numpy())




def is_pareto(costs):
    is_efficient = np.ones(costs.shape[0], dtype=bool)


    for i, c in enumerate(costs):
        if is_efficient[i]:
           is_efficient[is_efficient] = np.any(costs[is_efficient]<=c, axis=1)

    return is_efficient



def pareto(s,c):
    s = np.array(s)
    s[:,0] *= -1
    mask = is_pareto(s)
    for m,ss,cc in zip(mask,s,c):
        if m==1:
            print(cc,ss)


