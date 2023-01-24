import numpy as np
import scanpy as sc
from cellsaw.merge.mergehelpers import hungarian
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.semi_supervised import LabelPropagation as lapro
from sklearn.semi_supervised import LabelSpreading as laspre
from MarkerCount.marker_count import MarkerCount_Ref, MarkerCount
import warnings
from cellsaw.merge import Merge
from cellsaw.merge.hungarianEM import linasEM

def mergewrap(a,b,umap_dim,**kwargs):
    assert  isinstance(umap_dim, int), 'umap_dim must be an integer'
    umaps = [] if umap_dim == 0 else [umap_dim]
    m =  Merge([a, b], umaps=umaps, **kwargs)
    return m

def linsum_copylabel(
        target,
        source,
        source_label = 'celltype',
        target_label= 'diffuseknn',
        premerged = False,
        n_genes = 800,
        pca_dim = 20, umap_dim = 0):

    pid = (pca_dim>0)+ (umap_dim>0)
    merged =   premerged or mergewrap(target,source,umap_dim,pca=pca_dim, sortfield = pid, selectgenes = n_genes)
    target.obs[target_label] = list(merged.data[1].obs[source_label])
    return target


def label_knn(target,source,
                       source_label = 'celltype',
                       target_label='knn',
                        n_genes = 800,
                       premerged = False,
                       pca_dim = 20, umap_dim = 0, k = 5):


    pid = (pca_dim>0)+ (umap_dim>0)
    merged = premerged or  mergewrap(target,source,umap_dim,pca=pca_dim, selectgenes = n_genes)
    a,b = merged.projections[pid]
    y = merged.data[1].obs['celltype']
    model = knn(n_neighbors=k).fit(a,y)
    target.obs[target_label] = model.predict(b)
    return target


def raw_diffusion(target, source, source_label ='celltype',
                  target_label='raw_diffusion',
                  premerged = False,
                  n_neighbors = 5, gamma = 10,
                  n_genes = 800,
                  pca_dim = 20, umap_dim = 0):
    pid = (pca_dim>0)+ (umap_dim>0)
    merged = premerged or mergewrap(target,source,umap_dim,pca=pca_dim,selectgenes=n_genes)
    a,b = merged.projections[pid]
    y = merged.data[1].obs[source_label]
    #diffusor = laspre( gamma = .1, n_neighbors = 5, alpha = .4).fit(b,y)
    diffusor = lapro( gamma = gamma, n_neighbors = n_neighbors).fit(b,y)
    target.obs[target_label] = diffusor.predict(a)
    return target



def tunnelclust(target, source, source_label ='celltype',
                  target_label='raw_diffusion',
                  premerged = False,
                  n_neighbors = 5,
                  n_genes = 800,
                  pca_dim = 20,
                umap_neigh  = 10,
                    umap_dim = 0):
    '''
    change params,
    write multitunnelclust wrapper to work with strings
    change the code in this function to call that wrapper
    '''
    pid = (pca_dim>0)+ (umap_dim>0)
    merged = premerged or mergewrap(target,source,umap_dim,pca=pca_dim,selectgenes=n_genes)
    a,b = merged.projections[pid]
    y = merged.data[1].obs[source_label]
    # diffusor = laspre( gamma = .1, n_neighbors = 5, alpha = .4).fit(b,y)
    # diffusor = lapro( gamma = gamma, n_neighbors = n_neighbors).fit(b,y)
    target.obs[target_label] = linasEM([a,b],y)
    return target



def markercount(target, source, source_label ='celltype',
                target_label='markercount_celltype', premerged = False,
                pca_dim = 20, umap_dim = 0):


    pid = (pca_dim>0)+ (umap_dim>0)

    # merged = premerged or mergewrap(target,source,umap_dim,pca=pca_dim)
    # X_ref=merged.data[1].to_df()
    # X_test=merged.data[0].to_df()
    # reflabels = merged.data[1].obs[source_label]

    X_ref=source.to_df()
    X_test=target.to_df()
    reflabels = source.obs[source_label]

    #print(f"{X_ref.shape=}{X_test.shape=}{reflabels=}")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            df_res = MarkerCount_Ref( X_ref, reflabels, X_test,
                                  cell_types_to_excl = ['Unknown'],
                                  log_transformed = True,
                                  file_to_save_marker = 'my_markers',
                                  verbose = False )
        except:
            # get results :D
            predict = ['failed to run markercount']*target.X.shape[0]
            target.obs[target_label] = predict
            return target
        # get results :D
        predict = df_res['cell_type_pred']
        target.obs[target_label] = predict
    return target


import anndata as ad
def raw_diffusion_combat(target, source, source_label ='celltype',
                  target_label='raw_diffusion',
                  premerged = False,
                  n_neighbors = 5, gamma = 10,
                  n_genes = 800,
                  pca_dim = 20, umap_dim = 0):
    pid = (pca_dim>0)+ (umap_dim>0)

    target.obs['batch'] = [1]*target.X.shape[0]
    source.obs['batch'] = [2]*source.X.shape[0]
    scoresvariable  = target.uns['lastscores']
    scr = [a.varm[scoresvariable] for a in [target,source]]
    z = ad.concat([target,source])
    z.obs_names_make_unique()
    sc.pp.combat(z,key= 'batch')
    target = z[z.obs['batch'] == 1]
    source = z[z.obs['batch'] == 2]
    #target.varm[scoresvariable]= scr[0]
    #source.varm[scoresvariable]= scr[1]
    #merged = premerged or mergewrap(target,source,umap_dim,pca=pca_dim,selectgenes= n_genes)
    #a,b = merged.projections[pid]
    '''
    todo dimension reduction pca
    dimred umap
    split a and b
    '''
    from sklearn.decomposition import PCA
    import umap
    x = PCA(n_components=pca_dim).fit_transform(z.X)
    if umap_dim > 0:
        x = umap.UMAP(n_components=umap_dim).fit_transform(x)

    a,b = x[z.obs['batch']==1], x[z.obs['batch']==2]

    y = source.obs[source_label]
    #diffusor = laspre( gamma = .1, n_neighbors = 5, alpha = .4).fit(b,y)
    diffusor = lapro( gamma = gamma, n_neighbors = n_neighbors).fit(b,y)
    target.obs[target_label] = diffusor.predict(a)
    return target


def scanorama_integrate_diffusion(target,source,
                       source_label = 'celltype',
                       target_label='scanorama',
                       pca_dim = 20, umap_dim = 0, gamma = .75):


    import scanorama
    from sklearn.decomposition import PCA
    import umap


    # make names unique and integrate
    target.obs['batch'] = [1]*target.X.shape[0]
    source.obs['batch'] = [2]*source.X.shape[0]
    z = ad.concat([target,source])
    z.obs_names_make_unique()
    a,b = z[z.obs['batch']==1], z[z.obs['batch']==2]
    scanorama.integrate_scanpy([a,b])

    # stack to do dimension reduction
    z = ad.concat([a,b])
    data = z.obsm['X_scanorama']
    x = PCA(n_components=pca_dim).fit_transform(data)
    if umap_dim > 0:
        x = umap.UMAP(n_components=umap_dim).fit_transform(x)

    # un-stack to do label transfer
    a,b = x[z.obs['batch']==1], x[z.obs['batch']==2]
    y = source.obs[source_label]
    #diffusor = laspre( gamma = .1, n_neighbors = 5, alpha = .4).fit(b,y)
    diffusor = lapro( gamma = gamma).fit(b,y)
    target.obs[target_label] = diffusor.predict(a)
    return target


