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
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
from collections import defaultdict

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
    return res



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



def lin_asi_thresh(ij_euclidian_distances,inter_neigh, outlier_threshold,
                   outlier_probabilistic_removal, prune = 0):

    i_ids,j_ids, ij_lsa_distances = iterated_linear_sum_assignment(ij_euclidian_distances,inter_neigh)
    # i_ids,j_ids, ij_lsa_distances = repeated_subsample_linear_sum_assignment(ij_euclidian_distances,inter_neigh,100)

    # remove worst 25% hits
    sorted_ij_assignment_distances  = np.sort(ij_lsa_distances)

    if outlier_probabilistic_removal:
        ij_lsa_distances[cdf_remove(ij_lsa_distances,outlier_threshold/2)] = 0

    elif  outlier_threshold is not None:
        lsa_outlier_thresh = sorted_ij_assignment_distances[int(len(ij_lsa_distances)*outlier_threshold)]
        outlier_ids = ij_lsa_distances >  lsa_outlier_thresh
        ij_lsa_distances[outlier_ids] = 0



    return i_ids, j_ids, ij_lsa_distances

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

def symmetric_spanning_tree_neighborgraph(x, neighbors,add_tree=True, neighbors_mutual = True):

    distancemat = metrics.euclidean_distances(x)

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


def average_knn_distance(I,J,i_ids,j_ids,numneigh):
    d1  = avgdist(I,numneigh)[i_ids]
    d2  = avgdist(J,numneigh)[j_ids]
    stack = np.vstack((d1,d2))
    return np.mean(stack, axis=0).T


def linear_assignment_integrate(Xlist, base = 'pca',
                                neighbors_total = 20,
                                neighbors_intra_fraction = .5,
                                intra_neigh=15,
                                inter_neigh = 1,
                                scaling_num_neighbors = 2,
                                outlier_threshold = .8,
                                dataset_adjacency = False,intra_neighbors_mutual = True,
                                copy_lsa_neighbors = True,outlier_probabilistic_removal = True,
                                add_tree = True, epsilon = 1e4 ):

    if 'anndata' in str(type(Xlist[0])):
        Xlist = to_arrays(Xlist, base)

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
                r = symmetric_spanning_tree_neighborgraph(Xlist[i], intra_neigh,
                                                          add_tree = add_tree,
                                                          neighbors_mutual=intra_neighbors_mutual)
                r = sparse.lil_matrix(r)
                return r


            elif not adjacent(i,j):
                # just fill with zeros :)
                return  sparse.lil_matrix((Xlist[i].shape[0],Xlist[j].shape[0]), dtype=np.float32)
            else:
                '''
                based = hungdists/= .25
                dm = avg 1nn dist in both
                based*=dm
                '''
                # hungarian
                ij_euclidian_distances= metrics.euclidean_distances(Xlist[i],Xlist[j])
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

                if epsilon:
                    res[res > 0] = epsilon


                return res

    n_datas = len(Xlist)
    tasks =  [(i,j) for i in range(n_datas) for j in range(i,n_datas)]
    #parts = ut.xxmap( make_distance_matrix, tasks)
    parts = Map( make_distance_matrix, tasks)
    getpart = dict(zip(tasks,parts))

    def steal_neigh(ij):
        i,j = ij
        return i,j,steal_neighbors(getpart[(i,j)], getpart[(j,j)])

    if copy_lsa_neighbors:
        tasks =  [(i,j) for i in range(n_datas) for j in range(i+1,n_datas)]
        parts = ut.xxmap( steal_neigh, tasks)
        for i,j,block in parts:
            getpart[(i,j)]  = block


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

    # check_symmetric(distance_matrix,raise_exception=True)
    # so.heatmap(distance_matrix.todense(), dim = (100,100))

    return  distance_matrix



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
