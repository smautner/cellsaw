
import scanpy as sc
from cellsaw.merge.mergehelpers import hungarian
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.semi_supervised import LabelPropagation as lapro
from sklearn.semi_supervised import LabelSpreading as laspre
from MarkerCount.marker_count import MarkerCount_Ref, MarkerCount
import warnings
from cellsaw.merge import Merge

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
        pca_dim = 20, umap_dim = 0):

    pid = (pca_dim>0)+ (umap_dim>0)
    merged =   premerged or mergewrap(target,source,umap_dim,pca=pca_dim, sortfield = pid)
    target.obs[target_label] = list(merged.data[1].obs[source_label])
    return target


def label_knn(target,source,
                       source_label = 'celltype',
                       target_label='knn',
                       premerged = False,
                       pca_dim = 20, umap_dim = 0, k = 5):


    pid = (pca_dim>0)+ (umap_dim>0)
    merged = premerged or  mergewrap(target,source,umap_dim,pca=pca_dim)
    a,b = merged.projections[pid]
    y = merged.data[1].obs['celltype']
    model = knn(n_neighbors=k).fit(a,y)
    target.obs[target_label] = model.predict(b)
    return target


def raw_diffusion(target, source, source_label ='celltype',
                  target_label='raw_diffusion',
                  premerged = False,
                  n_neighbors = 5, gamma = 10,
                  pca_dim = 20, umap_dim = 0):
    pid = (pca_dim>0)+ (umap_dim>0)
    print(f"{pid=}")
    merged = premerged or mergewrap(target,source,umap_dim,pca=pca_dim)
    a,b = merged.projections[pid]
    y = merged.data[1].obs[source_label]
    #diffusor = laspre( gamma = .1, n_neighbors = 5, alpha = .4).fit(b,y)
    diffusor = lapro( gamma = gamma, n_neighbors = n_neighbors).fit(b,y)
    target.obs[target_label] = diffusor.predict(a)
    return target


def markercount(target, source, source_label ='celltype',
                target_label='markercount_celltype', premerged = False,
                pca_dim = 20, umap_dim = 0):


    pid = (pca_dim>0)+ (umap_dim>0)
    merged = premerged or mergewrap(target,source,umap_dim,pca=pca_dim)

    X_ref=merged.data[1].to_df()
    X_test=merged.data[0].to_df()
    reflabels = merged.data[1].obs[source_label]
    #print(f"{X_ref.shape=}{X_test.shape=}{reflabels=}")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df_res = MarkerCount_Ref( X_ref, reflabels, X_test,
                                  cell_types_to_excl = ['Unknown'],
                                  log_transformed = True,
                                  file_to_save_marker = 'my_markers',
                                  verbose = False )
        # get results :D
        predict = df_res['cell_type_pred']
        target.obs[target_label] = predict
    return target

