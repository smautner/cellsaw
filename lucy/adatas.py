from lmz import Map,Zip,Filter,Grouper,Range,Transpose
from scipy.sparse import issparse
from sklearn import decomposition
import numpy as np
from scipy.optimize import linear_sum_assignment
import ubergauss.tools as ut
from lucy import draw
from sklearn.metrics import pairwise_distances
from anndata._core.merge import concat
import scanpy as sc
import umap as uumap


# plotting
def plot_confusion_matrix_normalized_raw(adatas, label = 'label', alignmentbase = 'pca40'):
    assert len(adatas) == 2
    adatas = align(adatas, base= alignmentbase)
    draw.plot_confusion_matrix_twice(*[x.obs['label'] for x in adatas])

def plot(adatas, projection = 'umap2', label= 'label', **kwargs):
    X = to_arrays(adatas, base=projection)
    labels = [a.obs[label] for a in adatas]
    batch_labels = [a.obs['batch'][0] for a in adatas]
    draw.plot_X(X, labels,titles = batch_labels,**kwargs)



# similarity
def jaccard_distance(a,b, num_genes):
    binarized_hvg = np.array([ ut.binarize(d,num_genes) for d in [a,b] ])
    union = np.sum(np.any(binarized_hvg, axis=0))
    intersect = np.sum(np.sum(binarized_hvg, axis=0) ==2)
    return intersect/union


def similarity(adatas, hvg_name = 'cell_ranger', num_genes = 2000):
    assert hvg_name in adatas[0].var, 'not annotated..'
    #genescores = [a.var[hvg_name] for a in adatas]
    genescores = [a.uns[hvg_name] for a in adatas]
    res = [[ jaccard_distance(a,b, num_genes = num_genes) for a in genescores] for b in genescores]
    return np.array(res)



# checking data
def check_adatas(adatas):
        assert isinstance(adatas, list), f'merge wants a list, not {type(adatas)}'
        assert all([a.X.shape[1] == adatas[0].X.shape[1] for a in adatas])



# sometimes we need to stack the adatas to work on the concatenated values...


# def to_arraylist(adatas):
#    # make a list of tupples for use in sklearn
#     return X,cell_name, genename

def stack(adatas):
    assert 'batch' in adatas[0].obs
    return concat(adatas)

def stack_single_attribute(adatas, attr = ""):
    if not attr:
        data = [a.X for a in adatas]
    else:
        data = [a.obsm[attr] for a in adatas]
    return ut.vstack(data)

def split_by_adatas(adatas, stack):
    batch_ids = np.hstack([ a.obs['batch'] for a in adatas])
    return [ stack [batch_ids == batch] for batch in np.unique(batch_ids)]

def attach_stack(adatas, stack, label):
    '''
    if we generate data for all cells on the stacked-adatas,
    this function can split the data and assign it to the adatas
    '''
    stack_split = split_by_adatas(adatas,stack)
    for a,s in zip(adatas,stack_split):
        a.obsm[label] = s
    return adatas



def preprocess(adatas,cut_ngenes = 2000, cut_old = False, hvg = 'cell_ranger', make_even = True):
    check_adatas(adatas)
    if hvg == 'cell_ranger':
        adatas = cell_ranger(adatas)
    else:
        assert False

    selector = hvg_ids_from_union if cut_old else hvg_ids_from_union_limit

    for a in adatas: # saving this info to be able to calculate similarity
        a.uns[hvg] = a.var[hvg]

    adatas = hvg_cut(adatas, selector(adatas,cut_ngenes,hvg_name=hvg))
    if make_even:
        adatas = subsample_to_min_cellcount(adatas)
    return adatas

def hvg_cut(adatas,hvg_ids):
    [d._inplace_subset_var(hvg_ids) for d in adatas]
    return adatas

def hvg_ids_from_union(adatas, numGenes, hvg_name= 'cell_ranger'):
    scores = [a.var[hvg_name] for a in adatas]
    hvg_ids_per_adata = np.argpartition(scores, -numGenes)[:,-numGenes:]
    hvg_ids = np.unique(hvg_ids_per_adata.flatten())
    return hvg_ids


def hvg_ids_from_union_limit(adatas,numgenes,hvg_name = 'cell_ranger'):

    scores = [a.var[hvg_name] for a in adatas]
    ar = np.array(scores)
    ind = np.argsort(ar)

    def top_n_union(array,n):
        indices = array[:,-n:]
        indices = np.unique(indices.flatten())
        return indices

    def findcutoff(low,high, lp = -1):
        probe = int((low+high)/2)
        y = top_n_union(ind,probe) # hvg_ids_from_union(adatas,probe)
        if probe == lp:
            return y
        if len(y) > numgenes:
            return findcutoff(low,probe,probe)
        else:
            return findcutoff(probe,high,probe)

    indices = findcutoff(0,numgenes)
    return indices

def subsample_to_min_cellcount(adatas):
        smallest = min([e.X.shape[0] for e in adatas])
        for a in adatas:
            if a.X.shape[0] > smallest:
                sc.pp.subsample(a, n_obs=smallest,
                                random_state=0,
                                copy=False)
        return adatas

import warnings
def subsample(data,num=1000, seed=None, copy = False):
    np.random.seed(seed)
    obs_indices = np.random.choice(data.n_obs, size=num, replace=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r=  data[obs_indices]
        if copy:
            r = r.copy()
        r.obs_names_make_unique()
    return r

def subsample_preprocess(adatas,num = 1000 ,copy = False, **preprocessargs):
    data = Map(subsample,adatas,num=num, copy=copy)
    return preprocess(data,**preprocessargs)



def align(adatas, base = 'pca40' ):
    for i in range(len(adatas)-1):
        hung, _ = hung_adatas(adatas[i], adatas[i+1], base= base)
        adatas[i+1]= adatas[i+1][hung[1]]

def to_array(ad,base):
    return ad.X if not base else ad.obsm[base]

def to_arrays(adatas,base):
    return Map(to_array, adatas, base=base)

def hungarian(adata, adata2, base):
        X = to_arrays([adata, adata2], base=base)
        hung, dist = hung_nparray(*X)
        return hung, dist[hung]

def hung_nparray(X1, X2, debug = False,metric='euclidean'):
    distances = pairwise_distances(X1,X2, metric=metric)
    row_ind, col_ind = linear_sum_assignment(distances)
    if debug:
        x = distances[row_ind, col_ind]
        num_bins = 100
        print("hungarian: debug hist")
        plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
        plt.show()
    return (row_ind, col_ind), distances



def cell_ranger(adatas, mingenes = 200,
                        normrow= True,
                        log = True):
    if 'cell_ranger' in adatas[0].var:
        return adatas

    return Map( lambda x:cell_ranger_single(x, mingenes=mingenes, normrow= normrow,  log= log), adatas)

def cell_ranger_single(adata,
                        mingenes = 200,
                        normrow= True,
                        log = True):

    okgenes = sc.pp.filter_genes(adata, min_counts=3, inplace=False)[0]
    sc.pp.normalize_total(adata, 1e4)
    sc.pp.log1p(adata)
    adata2 = adata[:,okgenes].copy()
    sc.pp.highly_variable_genes(adata2, n_top_genes=5000,
                                         flavor='cell_ranger',
                                        inplace=True)

    fullscores = np.full(adata.X.shape[1],np.NINF,np.float)
    fullscores[okgenes]  = adata2.var['dispersions_norm']
    adata.var['cell_ranger']=  fullscores
    return adata




from lucy import embed

def lapgraph(adatas,base = 'pca40', intra_neigh = 15, inter_neigh = 1,
              scaling_num_neighbors = 2, outlier_threshold = .8,
              scaling_threshold = .9, dataset_adjacency = None):

    X = to_arrays(adatas, base)
    graph =  embed.linear_assignment_integrate(X,
                            intra_neigh=intra_neigh,
                            inter_neigh = inter_neigh,
                            scaling_num_neighbors = scaling_num_neighbors,
                            outlier_threshold = outlier_threshold,
                            scaling_threshold=scaling_threshold,
                                dataset_adjacency=dataset_adjacency)
    return (graph, base)

def graph_embed(adatas, lapgraph, n_components= 2, label = 'lap'):
    # do the embedding
    graph, base = lapgraph
    X = to_arrays(adatas, base)
    projection = embed.distmatrixumap(X,graph, components = n_components)
    adatas = attach_stack(adatas, projection,label)
    return adatas




def pca(adatas, dim=40, label = 'pca40'):

    if label in adatas[0].obsm:
        print('redundant pca :) ')
    # get a result
    data = stack(adatas)
    scaled = sc.pp.scale(data, zero_center=False, copy=True,max_value=10).X
    stackedPCA =  pca_on_scaled_data(scaled, dim)
    return attach_stack(adatas, stackedPCA ,label)

def pca_on_scaled_data(scaled, dim):
    if  not issparse(scaled):
        stackedPCA = decomposition.PCA(n_components  = dim).fit_transform(scaled)
    else:
        stackedPCA = sc.pp._pca._pca_with_sparse(scaled,dim)['X_pca']
    return stackedPCA


def umap(adatas, dim = 10, label = 'umap10', start = 'pca40'):

    if label in adatas[0].obsm:
        print('redundant umap :) ')

    attr = data.obsm.get(start,'')
    X = stack_single_attribute(adatas, attr = attr)
    res = uumap.UMAP(n_components = dim).fit_transform(X)
    return attach_stack(adatas, res ,label)


from sklearn.preprocessing import scale
def project(adatas,start='', **kwargs):

    data = stack_single_attribute(adatas,attr= start)

    if 'pca' in kwargs:
        scaled = scale(data, with_mean= False)
        dim = kwargs['pca']
        data =  pca_on_scaled_data(scaled, dim)
        start = f"pca{dim}"
        attach_stack(adatas, data , start)

    if 'umap' in kwargs:
        dim = kwargs['umap']
        data = uumap.UMAP(n_components = dim).fit_transform(data)
        start = f"umap{dim}"
        attach_stack(adatas, data , start)

    if  'lapumap' in kwargs:
        dim = kwargs['lapumap']
        sim = embed.make_adjacency(similarity(adatas),algo=0,neighbors=10)
        gr = lapgraph(adatas,base=start,dataset_adjacency = sim)
        start = f"lapumap{dim}"
        graph_embed(adatas,gr,n_components = dim, label = start)

    return adatas



# predict function that just diffuses labels
def predict(adatas, tartget_id, labelname= 'predictedlabels'):
    pass


