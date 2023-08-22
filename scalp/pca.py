from scalp.data.transform import stack, attach_stack
from scipy.sparse import issparse
from sklearn import decomposition
import scanpy as sc

def pca(adatas, dim=40, label = 'pca40'):

    if label in adatas[0].obsm:
        print('redundant pca :) ')
    # get a result
    data = stack(adatas)
    scaled = sc.pp.scale(data, zero_center=False, copy=True,max_value=10).X
    stackedPCA =  pca_on_scaled_data(scaled, dim)
    return attach_stack(adatas, stackedPCA, label)


def pca_on_scaled_data(scaled, dim):
    if  not issparse(scaled):
        stackedPCA = decomposition.PCA(n_components  = dim).fit_transform(scaled)
    else:
        stackedPCA = sc.pp._pca._pca_with_sparse(scaled,dim)['X_pca']
    return stackedPCA
