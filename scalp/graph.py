from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import scalp.graph
from scalp.data.transform import to_arrays
from scipy import sparse
from sklearn import neighbors as nbrs
from sklearn.utils import check_symmetric
from sklearn import metrics
from scipy.sparse.csgraph import minimum_spanning_tree
import numpy as np
import time
from scipy.optimize import linear_sum_assignment
from sklearn.neighbors import NearestNeighbors
import structout as so
import ubergauss.tools as ut
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import rankdata
from collections import defaultdict
from sklearn import preprocessing as skprep

# import matplotlib
# matplotlib.use('module://matplotlib-backend-sixel')
# import matplotlib.pyplot as plt

def steal_neighbors(mlsa, mcopyfrom):
    '''
    from lsa block, copy the neighbors of the partner instead of only the partner
    '''
    res = np.zeros(mlsa.shape)
    mlsa = csr_matrix(mlsa)
    mcopyfrom = csr_matrix(mcopyfrom)
    for i, row in enumerate(mlsa):
        ij_ctr = {}
        for j in row.indices:
            newvalues = mcopyfrom[j]
            # average distance to neighbors :)
            res[i] += newvalues
            avg = np.mean(newvalues.data)
            if not np.isnan(avg):
                ij_ctr[(i,j)]  = avg
        for (i,j), n in ij_ctr.items(): res[i,j] = n


    # plt.matshow(mlsa.todense())
    # plt.show()
    # plt.matshow(mcopyfrom.todense())
    # plt.show()
    # # dim= (50,50)
    # # so.heatmap(mlsa.todense(),dim=dim)
    # # so.heatmap(mcopyfrom.todense(),dim=dim)
    # # so.heatmap(res,dim=dim)
    # plt.matshow(res)
    # plt.show()

    # assert not np.isnan(res).any()
    return sparse.lil_matrix(res)



def iterated_linear_sum_assignment(distances, repeats):
    def linear_sum_assignment_iteration(distances):
        r, c = linear_sum_assignment(distances)
        d = distances[r,c].copy()
        distances [r,c] = np.inf
        return r,c,d
    cc, rr, dist  = Transpose([ linear_sum_assignment_iteration(distances) for _ in range(repeats)  ])
    return np.hstack(cc) , np.hstack(rr), np.hstack(dist)

def test_cdf_remove():
    ar = np.arange(10, dtype= float)
    print(cdf_remove(ar,ar[::-1]))

from scipy.stats import norm

def cdf_remove(data, bias):
    loc, scale = norm.fit(data)
    probs  = 1 - norm.cdf(data, loc, scale)
    return  np.random.random(len(data)) > (probs+bias)



def lin_asi_thresh(ij_euclidian_distances,inter_neigh=1, outlier_threshold=.9,
                   outlier_probabilistic_removal=False, prune = 0):

    # get the (repeated) assignment
    i_ids,j_ids, ij_lsa_distances = iterated_linear_sum_assignment(ij_euclidian_distances,inter_neigh)
    # i_ids,j_ids, ij_lsa_distances = repeated_subsample_linear_sum_assignment(ij_euclidian_distances,inter_neigh,100)

    # remote outliers
    sorted_ij_assignment_distances  = np.sort(ij_lsa_distances)

    if outlier_probabilistic_removal and outlier_threshold > 0:
        ij_lsa_distances[cdf_remove(ij_lsa_distances,outlier_threshold/2)] = 0
    elif  outlier_threshold > 0:
        lsa_outlier_thresh = sorted_ij_assignment_distances[int(len(ij_lsa_distances)*outlier_threshold)]
        outlier_ids = ij_lsa_distances >  lsa_outlier_thresh
        ij_lsa_distances[outlier_ids] = 0

    mask = ij_lsa_distances != 0
    return i_ids[mask], j_ids[mask], ij_lsa_distances[mask]
    # return i_ids, j_ids, ij_lsa_distances

def lsa_sample(distance_matrix, num_instances):
    # get random indices
    row_indices = np.random.choice(distance_matrix.shape[0], num_instances, replace = False)
    col_indices = np.random.choice(distance_matrix.shape[1], num_instances, replace = False)

    # get matches and distances
    subdistance_matrix = distance_matrix[row_indices][:,col_indices]
    r, c = linear_sum_assignment(subdistance_matrix)
    d =    subdistance_matrix[r,c].copy()

    # since subdistances is basically reindexing, we need to undo that:
    r = ut.spacemap(row_indices).decode(r)
    c = ut.spacemap(col_indices).decode(c)
    return r,c,d

def repeated_subsample_linear_sum_assignment(distances, repeats, num_instances):
    rr, cc, dist  = Transpose([ lsa_sample(distances,num_instances) for _ in range(repeats)  ])
    return np.hstack(rr) , np.hstack(cc), np.hstack(dist)






def fast_neighborgraph(distancemat, neighbors):
    part = np.argpartition(distancemat, neighbors, axis = 1)[:,:neighbors]
    neighborsgraph = np.zeros_like(distancemat)
    np.put_along_axis(neighborsgraph, part, np.take(distancemat,part), axis = 1)
    return neighborsgraph

def mutualNN(mtx):
    mask = mtx != 0
    mask &= mask.T
    return mtx * mask

from collections import Counter
def spanning_tree_neighborgraph(x, neighbors,add_tree=True, neighbors_mutual = True):
    '''
    the plan:
    '''

    model = NearestNeighbors(n_neighbors=neighbors*2).fit(x)
    distances, indices = model.kneighbors(x)
    counts = np.bincount(indices)
    counts_mat = counts[indices]
    cnt_srt = np.argsort(counts_mat, axis = 1)
    indices_new = np.take_along_axis(indices, cnt_srt, axis =1)[:,:neighbors]
    neighborsgraph = np.zeros_like((x.shape[0],x.shape[0]))
    np.put_along_axis(neighborsgraph,indices_new,1,axis=1)

    # distancemat = metrics.euclidean_distances(x)


    # min spanning tree
    # tree = minimum_spanning_tree(distancemat) if add_tree else np.zeros_like(distancemat)
    # tree= ut.zehidense(tree)
    # faster version of neighborsgraph
    # neighborsgraph = fast_neighborgraph(distancemat, neighbors)
    #anti_neighborsgraph = -fast_neighborgraph(-distancemat, neighbors)+100
    # make mutual
    # if neighbors_mutual:
    # neighborsgraph = mutualNN(neighborsgraph)
    # combine and return
    #combinedgraph = np.stack((neighborsgraph,neighborsgraph.T,tree,tree.T, anti_neighborsgraph,anti_neighborsgraph.T), axis =2)
    combinedgraph = np.stack((neighborsgraph,neighborsgraph.T ), axis =2)
    combinedgraph = combinedgraph.max(axis=2)
    np.fill_diagonal(combinedgraph,0)
    check_symmetric(combinedgraph,raise_exception=True) # graph_umap needs a symmetic matrix
    return combinedgraph

def symmetric_spanning_tree_neighborgraph(distancemat, neighbors,add_tree=True, neighbors_mutual = True):

    # distancemat = metrics.euclidean_distances(x)
    # distancemat = skprep.normalize(distancemat, axis =0) # DOTO this should get a flag

    #return fast_neighborgraph(distancemat, neighbors)
    # min spanning tree
    tree = minimum_spanning_tree(distancemat) if add_tree else np.zeros_like(distancemat)
    tree= ut.zehidense(tree)

    # faster version of neighborsgraph
    neighborsgraph = fast_neighborgraph(distancemat, neighbors)
    #anti_neighborsgraph = -fast_neighborgraph(-distancemat, neighbors)+100

    # make mutual
    if neighbors_mutual:
        neighborsgraph = mutualNN(neighborsgraph)

    # combine and return
    #combinedgraph = np.stack((neighborsgraph,neighborsgraph.T,tree,tree.T, anti_neighborsgraph,anti_neighborsgraph.T), axis =2)

    combinedgraph = np.stack((neighborsgraph,neighborsgraph.T,tree,tree.T ), axis =2)
    combinedgraph = combinedgraph.max(axis=2)
    np.fill_diagonal(combinedgraph,0)
    check_symmetric(combinedgraph,raise_exception=True)


    return combinedgraph



def avgdist(a,numneigh = 2):
    #  return mean distance to the first numneigh neighbors for each item in a
    nbrs = NearestNeighbors(n_neighbors=1+numneigh).fit(a)
    distances, indices = nbrs.kneighbors(a)
    return np.mean(distances[:,1:], axis = 1)


# def average_knn_distance(I,J,i_ids,j_ids,numneigh):
#     d1  = avgdist(I,numneigh)[i_ids]
#     d2  = avgdist(J,numneigh)[j_ids]
#     stack = np.vstack((d1,d2))
#     return np.mean(stack, axis=0).T


def linear_assignment_integrate(Xlist, base = 'pca',
                                neighbors_total = 20,
                                horizonCutoff = 0,
                                neighbors_intra_fraction = .5,
                                intra_neigh=15,
                                inter_neigh = 1,
                                scaling_num_neighbors = 2,
                                distance_metric = 'euclidean',
                                outlier_threshold = .8,
                                dataset_adjacency = False,intra_neighbors_mutual = True,
                                copy_lsa_neighbors = True,outlier_probabilistic_removal = True,
                                add_tree = True, epsilon = 1e-4 ):

    if 'anndata' in str(type(Xlist[0])):
        Xlist = to_arrays(Xlist, base)



    if len(Xlist) ==1:
                r = symmetric_spanning_tree_neighborgraph(Xlist[0],
                # r = spanning_tree_neighborgraph(Xlist[0],
                                                          int(neighbors_total),
                                                          add_tree = add_tree,
                                                          neighbors_mutual=intra_neighbors_mutual)
                r = sparse.lil_matrix(r)
                return r

    intra_neigh = int(max(1,np.ceil(neighbors_total*neighbors_intra_fraction)))
    inter_neigh_total = neighbors_total - intra_neigh #max(1,np.rint(neighbors_total*(1-neighbors_intra_fraction)))
    inter_neigh_desired = inter_neigh_total / (len(Xlist)-1) # this is how many we want
    inter_neigh = int(np.ceil(inter_neigh_desired)) #int(max(1,np.rint(inter_neigh_total / len(Xlist)))) # this is how many we sample


    def adjacent(i,j):
        if isinstance( dataset_adjacency, np.ndarray):
            return  dataset_adjacency[i][j] == 1
        else:
            return True

    def make_distance_matrix(ij):
            i,j = ij
            if i == j:
                # use the maximum between neighborgraph and min spanning tree to make sure all is connected
                distances = metrics.pairwise_distances(Xlist[i],metric = distance_metric)

                r = symmetric_spanning_tree_neighborgraph(distances, intra_neigh,
                                                          add_tree = add_tree,
                                                          neighbors_mutual=intra_neighbors_mutual)
                r = sparse.lil_matrix(r)

                # cutoffs =  [heapq.nlargest(horizonCutoff,values) for values in r.data] if horizonCutoff else 0
                cutoffs = np.partition(distances, horizonCutoff, axis=1)[:, horizonCutoff] if horizonCutoff else 0
                return r, cutoffs


            elif not adjacent(i,j):
                # just fill with zeros :)
                return  sparse.lil_matrix((Xlist[i].shape[0],Xlist[j].shape[0]), dtype=np.float32) ,0
            else:
                '''
                based = hungdists/= .25
                dm = avg 1nn dist in both
                based*=dm
                '''
                # hungarian
                # ij_euclidian_distances= metrics.euclidean_distances(Xlist[i],Xlist[j])
                ij_euclidian_distances= metrics.pairwise_distances(Xlist[i],Xlist[j],
                                                                   metric=distance_metric)



                res = sparse.lil_matrix(ij_euclidian_distances.shape,dtype=np.float32)
                if inter_neigh==0:
                    return res

                i_ids,j_ids, ij_lsa_distances = lin_asi_thresh(ij_euclidian_distances,
                                                               inter_neigh,outlier_threshold,
                                                               outlier_probabilistic_removal)


                res[i_ids,j_ids] = ij_lsa_distances



                for i,row in enumerate(res):
                    neighbors_have = np.sum(row > 0)
                    if neighbors_have > inter_neigh_desired:
                        assert neighbors_have > inter_neigh_desired > (neighbors_have -1), 'something is wrong with the neighbrs'
                        # we will just remove one, so:
                        if np.random.rand() < ( neighbors_have - inter_neigh_desired): # ok one needs to go
                            targets = np.array(row.rows[0])
                            z = np.random.choice(targets,1)[0]
                            res[i,z] = 0



                if epsilon > 0:
                    res[res > 0] = epsilon

                # zero is the horizon cutoff. which we dont need here
                return res,0

    n_datas = len(Xlist)
    tasks =  [(i,j) for i in range(n_datas) for j in range(i,n_datas)]
    parts = ut.xxmap( make_distance_matrix, tasks)
    # parts = Map( make_distance_matrix, tasks)
    # select p[0] to rm horizon cutoff values
    getpart = dict(zip(tasks,[p[0] for p in parts]))
    getcuts = dict(zip(tasks,[p[1] for p in parts]))


    def steal_neigh(ij):
        i,j = ij
        return i,j,steal_neighbors(getpart[(i,j)], getpart[(j,j)])

    if copy_lsa_neighbors:
        tasks =  [(i,j) for i in range(n_datas) for j in range(i+1,n_datas)]
        parts = ut.xxmap( steal_neigh, tasks)
        for i,j,block in parts:
            getpart[(i,j)]  = block


    # then we built  rows:
    rows = []
    for i in range(n_datas):
        row = []
        for j in range(n_datas):
            if i <= j:
                distance_matrix = getpart[(i,j)]
            else:
                distance_matrix = rows[j][i].T.copy()
            row.append(distance_matrix)

        rows.append(row)


    if horizonCutoff > 0:
        for i in range(n_datas):
            # for c in row:
            #     plt.matshow(c.todense())
            #     plt.roworbar()
            #     plt.show()
            #plt.matshow(np.hstack([c.todense() for c in row]))
            rows[i] = list(removeInstancesFartherThanHorizon(getcuts[(i,i)],rows[i],i,horizonCutoff))

        # now we make it symmetric
        for i in range(n_datas-1):
            for j in range(i+1,n_datas):
                rows[i][j]= rows[i][j].maximum(rows[j][i])
                rows[j][i]= rows[i][j].T

    rowss = [sparse.hstack(row) for row in rows]
    distance_matrix = sparse.vstack(rowss)

    # check_symmetric(distance_matrix,raise_exception=True)
    # so.heatmap(distance_matrix.todense(), dim = (100,100))

    return  distance_matrix

from matplotlib import pyplot as plt
import heapq
def removeInstancesFartherThanHorizon(cutoffs, targets :list,rId:int, nthNeighbor:int):
    '''
    remove instances that are farther than the nth neighbor in the reference in all matrixes contained in targets
    '''
    # cutoffs = np.partition(reference,nthNeighbor,axis=1)[:,nthNeighbor]
    # cutoffs =  [heapq.nlargest(nthNeighbor,values)[-1] for values in reference.data]

    #plt.hist(cutoffs);plt.show()
    #print(f"{cutoffs[0]=} {targets[0][0].data=} {targets[1][0].data=}")
    for j,target in enumerate(targets):
        # values larger than cutoffs (per row) are set to zero
        if j!=rId:
            for i,cut in enumerate(cutoffs):
                badind = [ind for ind,val in zip(target.rows[i], target.data[i]) if val > cut]
                target[i,badind] = 0
        yield target

def testHorizonCutOff():
    random = np.random.random((5,5))
    print(random)
    cutoffs = np.partition(random,3 ,axis=1)[:,3]
    print(cutoffs)
    random= lil_matrix(random)
    for i,cut in enumerate(cutoffs):
        badind = [ind for ind,val in zip(random.rows[i], random.data[i]) if val > cut]
        random[i,badind] = 0
    breakpoint()

# testHorizonCutOff()



def negstuff(Xlist,
            base = 'pca',
            neighbors_total = 20,
             neighbors_intra_fraction = .5, **kwargs):

    if 'anndata' in str(type(Xlist[0])):
        Xlist = to_arrays(Xlist, base)

    # intra_neigh = int(max(1,np.ceil(neighbors_total*neighbors_intra_fraction)))
    neg_used = neighbors_total
    neg_samples= int(Xlist[0].shape[0]/2)
    assert neg_used <= neg_samples, f"{neg_used=}  {neg_samples=}"
    def make_distance_matrix(ij):
        i,j = ij
        if i == j:
            distancemat = metrics.euclidean_distances(Xlist[i])
            part = np.argpartition(-distancemat, neg_samples, axis = 1)[:,:neg_samples]
            # part = np.argsort(-distancemat, axis = 1)[:,neg_samples//2:neg_samples]
            np.random.shuffle(part.T)
            part  =part[:,:neg_used]
            neighborsgraph = np.zeros_like(distancemat)
            np.put_along_axis(neighborsgraph, part, np.take(distancemat,part), axis = 1)
            return sparse.lil_matrix(neighborsgraph)
        return  sparse.lil_matrix((Xlist[i].shape[0],Xlist[j].shape[0]), dtype=np.float32)

    n_datas = len(Xlist)
    tasks =  [(i,j) for i in range(n_datas) for j in range(i,n_datas)]
    #parts = ut.xxmap( make_distance_matrix, tasks)
    parts = Map( make_distance_matrix, tasks)
    getpart = dict(zip(tasks,parts))

    # then we built a row:
    row = []
    for i in range(n_datas):
        col = []
        for j in range(n_datas):
            if i <= j:
                distance_matrix = getpart[(i,j)]
            else:
                distance_matrix = row[j][i].T
            col.append(distance_matrix)
        row.append(col)
    rows = [sparse.hstack(col) for col in row]
    distance_matrix = sparse.vstack(rows)
    return  csr_matrix(distance_matrix)














# import numpy as np
# from scipy.optimize import linear_sum_assignment
# from sklearn.neighbors import NearestNeighbors

import itertools
from scipy.spatial.distance import cdist
from scipy.sparse import coo_matrix, csr_matrix

def aiSlopSolution(matrices, k, h):
    """
    Constructs a sparse adjacency matrix based on k-NN within each matrix,
    adds cross-edges using linear assignment pairwise between matrices,
    and filters edges based on horizon h.

    Parameters:
    - matrices: list of numpy arrays or anndata, each of shape (n_i, d)
    - k: int, number of nearest neighbors within each matrix
    - h: int, horizon parameter for filtering edges

    Returns:
    - adjacency: scipy.sparse.csr_matrix of shape (total_instances, total_instances)
    """

    # fixing input
    if 'anndata' in str(type(matrices[0])):
        matrices = to_arrays(matrices, 'pca40')

    num_matrices = len(matrices)
    sizes = [mat.shape[0] for mat in matrices]
    total = sum(sizes)

    # Compute cumulative sizes for indexing
    cum_sizes = np.cumsum([0] + sizes)

    # Step 1: Find k-NN within each matrix
    knn_info = []
    h_distances = []

    for idx, mat in enumerate(matrices):
        nbrs = NearestNeighbors(n_neighbors=min(k, mat.shape[0]-1), algorithm='auto').fit(mat)
        distances, indices = nbrs.kneighbors(mat)
        knn_info.append((distances, indices))

        # Compute h-th nearest neighbor distances
        if h <= distances.shape[1]:
            h_dist = distances[:, h-1]
        else:
            # Recompute with h neighbors if h > current neighbors
            nbrs_h = NearestNeighbors(n_neighbors=min(h, mat.shape[0]-1), algorithm='auto').fit(mat)
            distances_h, _ = nbrs_h.kneighbors(mat)
            if h <= distances_h.shape[1]:
                h_dist = distances_h[:, h-1]
            else:
                # If h is still larger, take the maximum distance
                h_dist = distances_h[:, -1]
        h_distances.append(h_dist)

    # Prepare lists for adjacency matrix
    row = []
    col = []
    data = []

    # Add within-matrix edges
    for matrix_idx, (distances, indices) in enumerate(knn_info):
        base_idx = cum_sizes[matrix_idx]
        for i in range(sizes[matrix_idx]):
            for j in range(distances.shape[1]):
                neighbor = indices[i, j]
                row.append(base_idx + i)
                col.append(base_idx + neighbor)
                data.append(distances[i, j])

    # Step 2: Linear Assignment between each unique pair of matrices
    for (i, j) in itertools.combinations(range(num_matrices), 2):
        A = matrices[i]
        B = matrices[j]
        n_A = sizes[i]
        n_B = sizes[j]
        base_A = cum_sizes[i]
        base_B = cum_sizes[j]

        # Compute cost matrix
        cost_matrix = cdist(A, B, metric='euclidean')

        # Determine the number of assignments (min(n_A, n_B))
        num_assignments = min(n_A, n_B)

        # Solve linear assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        # If there are more assignments needed than min(n_A, n_B), limit them
        if len(row_ind) > num_assignments:
            row_ind = row_ind[:num_assignments]
            col_ind = col_ind[:num_assignments]

        # Add cross-edges
        for a, b in zip(row_ind, col_ind):
            distance = cost_matrix[a, b]
            row.append(base_A + a)
            col.append(base_B + b)
            data.append(distance)
            # Assuming undirected graph, add reverse edge
            row.append(base_B + b)
            col.append(base_A + a)
            data.append(distance)

    # Step 3: Horizon filtering
    # Convert lists to arrays for processing
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)

    # Identify cross-edges (between different matrices)
    # A cross-edge exists if the source and target belong to different matrices
    def find_matrix(idx):
        # Binary search to find which matrix the index belongs to
        matrix_idx = np.searchsorted(cum_sizes, idx, side='right') - 1
        return matrix_idx

    # Vectorized approach to find matrix indices
    matrix_indices_row = np.searchsorted(cum_sizes, row, side='right') - 1
    matrix_indices_col = np.searchsorted(cum_sizes, col, side='right') - 1

    cross_mask = matrix_indices_row != matrix_indices_col

    # Get indices of cross-edges
    cross_indices = np.where(cross_mask)[0]

    # Get the distances of cross-edges
    cross_distances = data[cross_indices]

    # Get endpoints for cross-edges
    endpoints_row = row[cross_indices]
    endpoints_col = col[cross_indices]

    # Compute min(h_distance of endpoints)
    # For each endpoint, determine which matrix it belongs to and fetch h_distance
    # Create an array to store h_distances for each instance
    # Initialize an array with total size
    h_distance_all = np.zeros(total)
    for m_idx in range(num_matrices):
        start = cum_sizes[m_idx]
        end = cum_sizes[m_idx + 1]
        h_distance_all[start:end] = h_distances[m_idx]

    # Get h_distances for endpoints
    h_dist_row = h_distance_all[endpoints_row]
    h_dist_col = h_distance_all[endpoints_col]

    # Compute the minimum h_distance for each cross-edge
    min_h_dist = np.minimum(h_dist_row, h_dist_col)

    # Determine which cross-edges to keep
    keep_mask = cross_distances <= min_h_dist

    # Indices to remove (cross-edges not satisfying the condition)
    remove_indices = cross_indices[~keep_mask]

    # Remove these edges
    row = np.delete(row, remove_indices)
    col = np.delete(col, remove_indices)
    data = np.delete(data, remove_indices)

    # Step 4: Construct sparse adjacency matrix
    adjacency = coo_matrix((data, (row, col)), shape=(total, total))

    # Optionally, make the matrix symmetric (if not already)
    adjacency = adjacency.maximum(adjacency.transpose())

    # Convert to CSR format for efficient arithmetic and slicing
    adjacency = adjacency.tocsr()

    return adjacency










def mkblock(matrix, i,j):
    '''
    we return a new array, the size of matrix
    row[i] of the new matrix, has the data of row[j] of the old matrix
    matrix and res should be lil_matrix
    '''

    # res = np.zeros((matrix.shape[0],matrix.shape[0]),dtype=np.float32)

    # print(f"{ res.shape=}")
    # print(f"{ matrix.shape=}")
    # print(f"{ i.shape=}")
    # print(f"{ j.shape=}")

    res = sparse.lil_matrix(matrix.shape)
    # matrix = matrix.T
    res[i] = matrix[j]
    return res

from scalp import transform
import sklearn


def stack_blocks(n_datas, getpart):
    '''
    getpart giges us the blocks, and we need to stack them into a matrix
    '''
    rows = []
    for i in range(n_datas):
        row = []
        for j in range(n_datas):
            # if i <= j:
            if (i,j) in getpart:
                distance_matrix = getpart[(i,j)]
            else:
                distance_matrix = rows[j][i].T.copy()
            row.append(distance_matrix)
        rows.append(row)
    rowss = [sparse.hstack(row) for row in rows]
    return sparse.vstack(rowss)



integrate_params = '''
hub1_k 3 20 1
hub2_k 3 20 1
hub1_algo 1 5 1
hub2_algo 1 5 1
outlier_threshold .65 .9
k 7 17 1
'''

# metric ['cosine']


def integrate(adata,*, base = 'pca40',
              k=13,
              metric = 'cosine',
              dataset_adjacency=False,
              hub1_k = 5,
              hub2_k = 5,
              hub1_algo = 1,
              hub2_algo = 2,
              outlier_threshold= .76):

    # make sure the input is in the right format
    # there are 3 options: adata, list of adata and list of np.array
    # case 1:
    if 'anndata' in str(type(adata)):
        adata = transform.split_by_obs(adata)

    # ok now we have a list... it could be in the wrong format
    if 'anndata' in str(type(adata[0])):
        adata = to_arrays(adata, base)

    Xlist =adata

    if len(Xlist) ==1:
        assert False, 'why do you only provide 1 dataset?'

    def adjacent(i,j):
        if isinstance( dataset_adjacency, np.ndarray):
            return  dataset_adjacency[i][j] == 1
        else:
            return True

    def make_distance_matrix(ij):
            i,j = ij

            if not adjacent(i,j):
                return [],[]# sparse.lil_matrix((Xlist[i].shape[0],Xlist[j].shape[0]), dtype=np.float32)

            distances= metrics.pairwise_distances(Xlist[i],Xlist[j], metric=metric)

            if i == j:
                distances = hubness(distances, hub1_k, hub1_algo)
                distances = fast_neighborgraph(distances, k)
                distances = sparse.lil_matrix(distances)
                return distances
            # this is a case for k-start :) from ug.hubness
            distances = hubness(distances, hub2_k, hub2_algo)
            i_ids,j_ids, ij_lsa_distances = lin_asi_thresh(distances, 1,outlier_threshold, False)
            return i_ids, j_ids

    n_datas = len(Xlist)
    tasks =  [(i,j) for i in range(n_datas) for j in range(i,n_datas)]
    parts = Map( make_distance_matrix, tasks)
    getpart = dict(zip(tasks,parts))

    blockdict = {(i,i): getpart[(i,i)] for i in range(n_datas)}


    tasks =  [(i,j) for i in range(n_datas) for j in range(i+1,n_datas)]
    for i,j in tasks:
        blockdict[(i,j)]  = mkblock( getpart[(j,j)] , *getpart[(i,j)])

    for i,j in tasks:
        # ok so we fill the mirror too... breaking symmetry
        # i,i is the reference now
        imatch, jmatch  = getpart[(i,j)]
        blockdict[(j,i)]  = mkblock( getpart[(i,i)] , jmatch, imatch )
    # MAKE THE MATRIX
    # check_symmetric(distance_matrix,raise_exception=True)
    # so.heatmap(distance_matrix.todense(), dim = (100,100))
    return  stack_blocks(n_datas, blockdict)

def test_integrate():
    # make 2 random 10x10 matrices
    # and run integrate on them
    # import scalp.data as data
    # a = data.scib(scalp.test_config.scib_datapath, maxdatasets=3, maxcells = 100, datasets = ["Immune_ALL_hum_mou"]).__next__()

    a= (np.random.random((10,200)),np.random.random((10,200)))
    integrate(a)


def MP(distance_matrix, k=6):
    n = distance_matrix.shape[0]
    k = int(np.sqrt(n))
    nbrs = NearestNeighbors(n_neighbors=k).fit(distance_matrix)
    distances, indices = nbrs.kneighbors(distance_matrix)  # Sorted by distance

    # Initialize rank matrix (high rank = less proximity)
    ranks = np.zeros((n, n))
    for i in range(n):
        ranks[i, indices[i]] = np.arange(1, k + 1)  # Rank 1 is nearest

    # Step 2: Convert ranks to empirical probabilities (P_i(x_j))
    P = ranks / n  # P_i(x_j) = rank_i(x_j) / n

    # Step 3: Compute MP as P_i(x_j) * P_j(x_i)
    MP = P * P.T
    return MP



def hubness(distance_matrix, k=6, algo = 0):
    """
    0 -> do nothing
    1 -> normalize by norm
    2 -> csls
    3 -> ls
    4 -> nicdm
    """
    if algo == 0:
        return distance_matrix
    if algo == 1:
        return sklearn.preprocessing.normalize(distance_matrix, axis = 0)

    # if algo == 2:
    #     return MP(distance_matrix, k + 15)

    funcs = [csls_, ls, nicdm, ka, another]
    f = funcs[algo-2]

    n = distance_matrix.shape[0]
    # scaled_distances = distance_matrix.copy()
    knn = np.partition(distance_matrix, k+1, axis=1)[:, :k+1]  # +1 to account for self
    knn = np.sort(knn, axis = 1)
    knn = knn[:,1:].mean(axis = 1)

    # Apply scaling
    for i in range(n):
        for j in range(n):
            v = distance_matrix[i,j]
            distance_matrix[i,j]  =  f(v,knn[i],knn[j])
    return distance_matrix

def csls_(v,i,j):
    return v*2 -i -j
def ls(v,i,j):
    return 1- np.exp(- v**2/(i*j) )
def nicdm(v,i,j):
    return v /  np.sqrt(i*j)
def ka(v,i,j):
    return v / i +  v/j
def another(v,i,j):
    return v * j ** .5
