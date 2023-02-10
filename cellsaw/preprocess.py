import matplotlib.pyplot as plt
from lmz import *
import sklearn
import seaborn as sns
from anndata import AnnData
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.sparse import csr_matrix
from lmz import *
import logging



def annotate_genescore(adatas, **kwargs):
    r = []
    for i,adata in enumerate(adatas):
        try:
            r.append(annotate_genescore_single(adata,**kwargs))
        except:
            print ('annotate genescore faild: ',i,adata.X.shape)
    return r



def normalize(adata):
    if 'normlog' not in adata.uns:
        sc.pp.normalize_total(adata, 1e4)
        # if "log1p" not in adata.uns_keys():
        sc.pp.log1p(adata)  # TODO look here, log1p is already set,, but its probably not transformed..
        adata.uns['normlog'] = True


def annotate_genescore_single(adata, selector='natto',
                        donormalize='auto',
                        nattoargs = {'mean':(0.015, 4),'bins':(.25, 1)},
                        mingenes = 200,
                        plot=False):
    '''will normlize in the end'''

    if selector in adata.varm:
        return adata

    incommingshape= adata.X.shape
    # sc.pp.filter_cells(adata, min_genes=mingenes, inplace=True)
    okgenes = sc.pp.filter_genes(adata, min_counts=3, inplace=False)[0]

    if isinstance(donormalize, str):
        if selector == 'seurat_v3':
            donormalize = False
        else:
            donormalize = True

    if donormalize:
        normalize(adata)

    #adata2 = adata.copy()
    adata2 = adata[:,okgenes].copy()
    if selector == 'meanexpression':
        if isinstance(adata2.X, csr_matrix):
            arr = adata2.X.todense().A
        else:
            arr = np.array(adata2.X)
        #print(f"{type(arr)=}")
        scores = np.nanmean(arr,axis = 0)

    elif selector == 'meanexpressionnolog':
        if isinstance(adata2.X, csr_matrix):
            arr = adata2.X.todense().A
        else:
            arr = np.array(adata2.X)
        #print(f"{type(arr)=}")
        scores = np.nanmean(np.expm1(arr),axis = 0)

    elif selector == 'preselected':
        self.preselected_genes = self.data[0].preselected_genes

    elif selector == 'natto':
        genes, scores = getgenes_natto(adata2, 1000, plot=plot, **nattoargs)

    elif selector == 'preselected':
        genes = [True if gene in self.preselected_genes else False for gene in adata2.var_names]
        scores = genes.as_type(int)

    else:
        hvg_df = sc.pp.highly_variable_genes(adata2, n_top_genes=5000, flavor=selector, inplace=False)
        genes = np.array(hvg_df['highly_variable'])
        if selector == 'seurat_v3':
            ### Best used for raw_count data
            scores = np.array(hvg_df['variances_norm'].fillna(0))
        else:
            scores = np.array(hvg_df['dispersions_norm'].fillna(0))

    fullscores = np.full(adata.X.shape[1],np.NINF,np.float)
    fullscores[okgenes==1] = scores
    adata.varm[selector]=  fullscores
    adata.uns['lastscores'] = selector
    adata.varm['genes'] = okgenes
    logging.info(f"transforming anndata (cells, genes): {incommingshape}  => {adata2.X.shape}")
    normalize(adata) # anyway in the end normalize
    return adata




def annotate_genescore2(adatas, **kwargs):
    r = []
    for i,adata in enumerate(adatas):
        try:
            r.append(annotate_genescore_single2(adata,**kwargs))
        except:
            print ('annotate genescore faild: ',i,adata.X.shape)
    return r
def annotate_genescore_single2(adata,
                        selector='natto',
                        nattoargs = {'mean':(0.015, 4),'bins':(.25, 1)},
                        mingenes = 200,
                        normrow= True,
                        log = True,
                        plot=False):
    '''
    trying to rewrite as i think i try to be too smart
    '''


    if selector in adata.varm:
        return adata

    incommingshape= adata.X.shape
    okgenes = sc.pp.filter_genes(adata, min_counts=3, inplace=False)[0]



    adata2 = adata[:,okgenes].copy()

    if normrow:
        sc.pp.normalize_total(adata2, 1e4)
    if log:
        sc.pp.log1p(adata2)


    if selector == 'meanexpression':
        if isinstance(adata2.X, csr_matrix):
            arr = adata2.X.todense().A
        else:
            arr = np.array(adata2.X)
        #print(f"{type(arr)=}")
        scores = np.nanmean(arr,axis = 0)

    elif selector == 'meanexpressionnolog':
        if isinstance(adata2.X, csr_matrix):
            arr = adata2.X.todense().A
        else:
            arr = np.array(adata2.X)
        #print(f"{type(arr)=}")
        scores = np.nanmean(np.expm1(arr),axis = 0)

    elif selector == 'natto':
        genes, scores = getgenes_natto(adata2, 1000, plot=plot, **nattoargs)

    else:
        hvg_df = sc.pp.highly_variable_genes(adata2, n_top_genes=5000, flavor=selector, inplace=False)
        genes = np.array(hvg_df['highly_variable'])
        if selector == 'seurat_v3':
            ### Best used for raw_count data
            scores = np.array(hvg_df['variances_norm'].fillna(0))
        else:
            scores = np.array(hvg_df['dispersions_norm'].fillna(0))

    fullscores = np.full(adata.X.shape[1],np.NINF,np.float)
    fullscores[okgenes==1] = scores
    adata.varm[selector]=  fullscores
    adata.uns['lastscores'] = selector
    adata.varm['genes'] = okgenes
    logging.info(f"transforming anndata (cells, genes): {incommingshape}  => {adata2.X.shape}")
    # normalize(adata) # anyway in the end normalize
    return adata


####
# ft select
###
def transform( means, var,plot, stepsize=.5, ran=3, minbin=0 ):
    x = np.arange(minbin * stepsize, ran, stepsize) #-> .5,1,1.5,2,2.5 ...

    #items = [(m, v) for m, v in zip(means, var)] #-> items = list(zip(means,var))
    items = Zip(means,var)

    boxes = [[i[1] for i in items if r < i[0] < r + (stepsize)] for r in x]


    ystdx = [(np.median(box),np.std(box),xcoo) for box,xcoo in zip(boxes,x) if box]

    #y = np.array([np.median(st) for st in boxes])
    #y_std = np.array([np.std(st) for st in boxes])
    y,y_std,x = map(np.array,Transpose(ystdx))

    x = x + (stepsize / 2)
    # draw regression points
    if plot:
        plt.scatter(x, y, label='Mean of bins', color='k')

    nonan = np.isfinite(y)
    x = x[nonan]
    y = y[nonan]
    y_std = y_std[nonan]
    x = x.reshape(-1, 1)
    return x, y, y_std


def get_expected_values(x, y, x_all, firstbinchoice = max):
    mod = sklearn.linear_model.HuberRegressor()
    mod.fit(x, y)
    res = mod.predict(x_all.reshape(-1, 1))
    firstbin = y[0]
    firstbin_esti = mod.predict([x[0]])
    res[x_all < x[0]] = firstbinchoice(firstbin, firstbin_esti)
    return res


def getgenes_natto(adata, selectgenes,
        mean=(.015,4),
        bins=(.25,1),
        plot=True):

    matrix= adata.to_df().to_numpy()
    a = np.expm1(matrix)
    var = np.var(a, axis=0)
    meanex = np.mean(a, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        # error will happen but we catch it later..
        disp = var / meanex
    Y = np.log(disp)
    X = np.log1p(meanex)


    #plt.scatter(X,Y)
    #plt.show()

    mask = np.array([not np.isnan(y) and me > mean[0] and me < mean[1] for y, me in zip(disp, X)])
    if plot:
        sns.set_style("whitegrid")
        plt.figure(figsize=(11, 4))
        plt.suptitle(f"gene selection", size=20, y=1.07)
        ax = plt.subplot(121)
        plt.scatter(X[mask], Y[mask], alpha=.2, s=3, label='all genes')

    x_bin, y_bin, ystd_bin = transform(X[mask].reshape(-1, 1),
                                            Y[mask],plot,
                                            stepsize=bins[0],
                                            ran=mean[1],
                                            minbin=bins[1] )


    y_predicted = get_expected_values(x_bin, y_bin, X[mask])
    std_predicted = get_expected_values(x_bin, ystd_bin, X[mask])
    Y[mask] -= y_predicted
    Y[mask] /= std_predicted

    srt = np.argsort(Y[mask])
    accept = np.full(Y[mask].shape, False)
    accept[srt[-selectgenes:]] = True

    if plot:
        srt = np.argsort(X[mask])
        plt.plot(X[mask][srt], y_predicted[srt], color='k', label='regression')
        plt.plot(X[mask][srt], std_predicted[srt], color='g', label='regression of std')
        plt.scatter(x_bin, ystd_bin, alpha=.4, label='Std of bins', color='g')
        plt.legend(bbox_to_anchor=(.6, -.2))
        plt.title("dispersion of genes")
        plt.xlabel('log mean expression')
        plt.ylabel('dispursion')
        ax = plt.subplot(122)
        plt.scatter(X[mask], Y[mask], alpha=.2, s=3, label='all genes')
        g = X[mask]
        d = Y[mask]
        plt.scatter(g[accept], d[accept], alpha=.3, s=3, color='r', label='selected genes')
        plt.legend(bbox_to_anchor=(.6, -.2))
        plt.title("normalized dispersion of genes")
        plt.xlabel('log mean expression')
        plt.ylabel('dispursion')
        plt.show()

        print(f"ft selected:{sum(accept)}")
    raw = np.zeros(len(mask))
    raw[mask] = Y[mask]

    mask[mask] = np.array(accept)
    return mask, raw
