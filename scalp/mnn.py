from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from scalp.data import transform
import ubergauss.tools as ut
import scanpy.external.pp as sep
import scanpy as sc
import scalp


'''
the methods here work like this:

function(adata_stacked, base = 'pca40', batchindicator = 'batch', label =  'function', dim=10)
-> set adata_stacked.obsm[label] = the new stuff
-> adata_stacked.uns['integrated'].append(label)

'''

import umap



def _scanorama(adatas, base = 'pca40', batchindicator = 'batch', label =  'scanorama'):
    adata = transform.stack(adatas)
    # sep.scanorama_integrate(adata, batchindicator, basis = base, adjusted_basis = label)
    sep.scanorama_integrate(adata, batchindicator, basis = base, adjusted_basis = label)
    return transform.split_by_obs(adata)




def scanorama(adata, base = 'pca40', batchindicator = 'batch', label =  'scanorama'):
    # adata = transform.stack(adatas)
    # sep.scanorama_integrate(adata, batchindicator, basis = base, adjusted_basis = label)
    assert base in adata.obsm

    sep.scanorama_integrate(adata, batchindicator, basis = base)

    # res = umap.UMAP(n_components = dim).fit_transform(adata.obsm['X_scanorama'])

    adata.obsm[label] = adata.obsm.pop('X_scanorama')
    adata.uns.setdefault('integrated',[])
    adata.uns['integrated'].append(label)

    return adata



def _combat(adatas, base = 'pca40', batchindicator = 'batch', label =  'combat'):
    adata = transform.stack(adatas)
    r = sc.pp.combat(adata,batchindicator, inplace=False)
    adata.obsm[label] = r
    return transform.split_by_obs(adata)

def combat(adata, base = 'pca40', batchindicator = 'batch', label =  'combat'):
    # adata = transform.stack(adatas)
    # densify adata.X
    adata.X = ut.zehidense(adata.X)
    r = sc.pp.combat(adata, batchindicator, inplace=False)
    adata.obsm[label] = r
    adata.uns.setdefault('integrated',[])
    adata.uns['integrated'].append(label)
    return adata

import bbknn
def _bbknnwrap(adatas, base = 'pca40', batchindicator = 'batch', dim = 2):
    adata = transform.stack(adatas)
    # sc.external.pp.bbknn(adata, batchindicator, use_rep=base)
    bbknn.bbknn(adata, use_rep = base)
    sc.tl.umap(adata,n_components=dim)
    return transform.split_by_obs(adata)

    # use this to do umap to a speciffic dim:
    # https://scanpy.readthedocs.io/en/latest/generated/scanpy.tl.umap.html#scanpy-tl-umap

def bbknnwrap(adata, base = 'pca40',label = 'bbknn', batchindicator = 'batch', dim = 10):
    # sc.external.pp.bbknn(adata, batchindicator, use_rep=base)
    bbknn.bbknn(adata, use_rep = base, batch_key= batchindicator)

    sc.tl.umap(adata,n_components=dim)
    adata.obsm[label] = adata.obsm.pop('X_umap')

    # adata.obsm[label] = adata.obsp.pop('connectivities')

    adata.uns.setdefault('integrated',[])
    adata.uns['integrated'].append(label)
    return adata

def mnn(adata, label = 'mnn'):

    # this used to work... grrr
    #mnn = sc.external.pp.mnn_correct(adata, n_jobs = 30)
    #mnnstack = adatas.stack(mnn[0][0])

    # needs to be dense...
    import mnnpy
    mnnpy.settings.normalization = "single"

    data = [ut.zehidense(a.X) for a in adata]
    matrixes = mnnpy.mnn_correct(*data,
                                 n_jobs = 1, do_concatenate = False, var_index = Range(data[0].shape[1]) )[0]
    # data = transform.stack(adata)
    # data.obsm[target] = mnn_matrix[0].X
    # adata = transform.attach_stack(adata,mnn_matrix,label)


    for a,m in zip(adata,matrixes):
        a.obsm[label] = m

    return adata




