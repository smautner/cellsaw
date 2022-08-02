import numpy as np
import scanpy as sc
import umap
from sklearn import decomposition
from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment


def make_even(data):
        # assert all equal
        size = data[0].X.shape[1]
        assert all([size == other.X.shape[1] for other in data])

        # find smallest
        counts = [e.X.shape[0] for e in data]
        smallest = min(counts)

        for a in data:
            if a.X.shape[0] > smallest:
                sc.pp.subsample(a,
                                fraction=None,
                                n_obs=smallest,
                                random_state=0,
                                copy=False)
        return data


def unioncut(scores, numGenes, data):
    indices = np.argpartition(scores, -numGenes)[:,-numGenes:]
    indices = np.unique(indices.flatten())
    return [d[:,indices].copy() for d in data]


def dimension_reduction(adatas, scale, zero_center, PCA, umaps, joint_space=True):


    # get a (scaled) dx
    if scale or PCA:
        adatas= [sc.pp.scale(adata, zero_center=False, copy=True,max_value=10) for adata in adatas]
    dx = [adata.to_df().to_numpy() for adata in adatas]
    if joint_space == False:
        return disjoint_dimension_reduction(dx, PCA, umaps)


    res = []


    if PCA:
        pca = decomposition.PCA(n_components=PCA)
        pca.fit(np.vstack(dx))
        #print('printing explained_variance\n',list(pca.explained_variance_ratio_))# rm this:)
        dx = [ pca.transform(e) for e in dx  ]
        res.append(dx)

    for dim in umaps:
        assert 0 < dim < PCA
        res.append(umapify(dx,dim))

    return res


def umapify(dx, dimensions):
    mymap = umap.UMAP(n_components=dimensions,
                      n_neighbors=10,
                      random_state=1337).fit(np.vstack(dx))
    return [mymap.transform(a) for a in dx]


def disjoint_dimension_reduction(dx, PCA, umaps):
    res = []
    if PCA:
        pcaList = [decomposition.PCA(n_components=PCA) for x in dx]
        for pca, x in zip(pcaList, dx):
            pca.fit(np.vstack(x))
        #print('printing explained_variance\n',list(pca.explained_variance_ratio_))# rm this:)
        dx = [ pca.transform(e) for pca, e in zip(pcaList,dx)  ]
        res.append(dx)

    for dim in umaps:
        assert 0 < dim < PCA
        res.append(umapify(dx,dim))


    return res

def hungarian(X1, X2, debug = False,metric='euclidean'):
    # get the matches:
    # import time
    # print("STARTING CALC")
    # now = time.time()
    distances = pairwise_distances(X1,X2, metric=metric)
    #print(f" distcalc: {time.time() -now}")
    #distances = ed(X1, X2)

    #if solver != 'scipy':
    '''
    from time import time
    now = time()
    from lapjv import lapjv
    row_ind, col_ind, _ = lapjv(distances)
    print(f"  {time() - now}s"); now =time()
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(distances)
    print(f"  {time() - now}s"); now =time()
    from lapsolver import solve_dense
    row_ind,col_ind = solve_dense(distances)
    print(f"  {time() - now}s"); now =time()
    '''
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(distances)

    if debug:
        x = distances[row_ind, col_ind]
        num_bins = 100
        print("hungarian: debug hist")
        plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
        plt.show()

    return (row_ind, col_ind), distances
