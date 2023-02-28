import numpy as np
from scipy.optimize import linear_sum_assignment
import ubergauss.tools as ut
from lucy import draw
from sklearn.metrics import pairwise_distances

def confuse2(adatas, label = 'label', alignmentbase = 'pca40'):
    assert len(adatas) == 2
    adatas = align(adatas, base= alignmentbase)
    draw.confuse2(*[x.obs['label'] for x in adatas])

def plot(adatas, projection = 'umap2', label= 'label', **kwargs):
    X = to_arrays(adatas, base=base)
    labels = [a.obs[label] for a in adatas]
    draw.plot_X(X, labels,**kwargs)


default_hvg_name = 'cell_ranger'

def jaccard_distance(a,b, num_genes):
    binarized_hvg = np.array([ ut.binarize(d,num_genes) for d in [a,b] ])
    union = np.sum(np.any(binarized_hvg, axis=0))
    intersect = np.sum(np.sum(binarized_hvg, axis=0) ==2)
    return intersect/union


def similarity(adatas, hvg_name = default_hvg_name, num_genes = 2000):
    assert hvg_name in adatas[0].varm, 'not annotated..'
    genescores = [a.varm[hvg_name] for a in adatas]
    res = [[ jaccard_distance(a,b, num_genes = num_genes) for a in scorelist] for b in scorelist]
    return np.array(res)


def check_adatas(adatas):
        assert isinstance(adatas, list), f'merge wants a list, not {type(adatas)}'
        assert all([a.X.shape[1] == adatas[0].X.shape[1] for a in adatas])


from anndata._core.merge import concat

def stack(adatas):
    # TODO we need to say that batch keeps them seperate
    return concat(adatas)

def unstack(adata, key= 'batch'):
   return  [z[z.obs['batch']==i] for i in z.obs[key].unique()]



def preprocess(adatas,cut_ngenes = 2000, cut_old = False, hvg = 'cell_ranger'):
    check_adatas(adatas)
    if hvg == 'cell_ranger':
        adatas = cell_ranger(adatas)
    else:
        assert False
    selector = hvg_ids_from_union if cut_old else hvg_ids_from_union_limit
    adatas = hvg_cut(adatas, selector(adatas,cut_ngenes,hvg_name=hvg))
    if make_even:
        adatas = subsample_to_min_cellcount(adatas)
    return adatas

def hvg_cut(adatas,hvg_ids):
    [d._inplace_subset_var(hvg_ids) for d in adatas]
    return adatas

def hvg_ids_from_union(adatas, numGenes, hvg_name= default_hvg_name):
    scores = [a.varm[hvg_name] for a in adatas]
    hvg_ids_per_adata = np.argpartition(scores, -numGenes)[:,-numGenes:]
    hvg_ids = np.unique(hvg_ids_per_adata.flatten())
    return hvg_ids


def hvg_ids_from_union_limit(adadas,numgenes,hvg_name = default_hvg_name):
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


def pca(adatas, dim, label = 'pca'):

    if not label:
        label = 'pca'+str(dim)
    if label in adatas[0].obsm:
        return adatas

    data = stack(adatas)
    scaled = sc.pp.scale(data, zero_center=False, copy=True,max_value=10).X
    res = decomposition.PCA(n_components  = dim).fit_transform(scaled)
    data.obsm[label] = res
    return unstack(data)


def umap(adatas, dim, label = '', start = 'pca40'):
    if not label:
        label = 'umap'+str(dim)
    if label in adatas[0].obsm:
        return adatas
    X = data.obsm.get(start,data.X)
    res = umap.UMAP(n_components = dim).fit_transform(X)
    data.obsm[label] = res
    return unstack(data)



def align(adatas, base = 'pca40' ):
    for i in range(len(adatas)-1):
        hung, _ = hung_adatas(adatas[i], adatas[i+1], base= base)
        adatas[i+1]= adatas[i+1][hung[1]]

def to_array(ad,base):
    return ad.X if not base else ad.obsm[base]

def to_arrays(adatas,base):
    return PMap(to_array, adatas, base=base)

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
    if 'cell_ranger' in adatas[0].varm:
        return adatas

    return Map( lambda x:cell_ranger_single(x, mingenes=mingenes, normrow= normrow,  log= log), adatas)

def cell_ranger_single(adata,
                        mingenes = 200,
                        normrow= True,
                        log = True):

    sc.pp.normalize_total(adata, 1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000,
                                         flavor='cell_ranger',
                                        inplace=True)
    adata.varm['cell_ranger']=  data.var['dispersions_norm']
    return adata





