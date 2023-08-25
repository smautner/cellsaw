from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from scalp.data import transform
import ubergauss.tools as ut
import scanpy.external.pp as sep

def scanorama(adatas, base = 'pca40', batchindicator = 'batch', obslabel =  'scanorama'):
    adata = transform.stack(adatas)
    sep.scanorama_integrate(adata,batchindicator, basis = base, adjusted_basis = obslabel)
    return transform.split_by_obs(adata)

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
