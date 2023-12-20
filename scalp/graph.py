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
        for j in row.indices:
            newvalues = mcopyfrom[j]
            # average distance to neighbors :)
            res[i,j] = np.mean(newvalues.data)
            res[i] += newvalues


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

    return res



def iterated_linear_sum_assignment(distances, repeats):
    def linear_sum_assignment_iteration(distances):
        r, c = linear_sum_assignment(distances)
        d = distances[r,c].copy()
        distances [r,c] = np.inf
        return r,c,d
    cc, rr, dist  = Transpose([ linear_sum_assignment_iteration(distances) for _ in range(repeats)  ])
    return np.hstack(cc) , np.hstack(rr), np.hstack(dist)


def lin_asi_thresh(ij_euclidian_distances,inter_neigh, outlier_threshold):
    if inter_neigh < 10:
        i_ids,j_ids, ij_lsa_distances = iterated_linear_sum_assignment(ij_euclidian_distances,inter_neigh)
    else:
        i_ids,j_ids, ij_lsa_distances = repeated_subsample_linear_sum_assignment(ij_euclidian_distances,inter_neigh,100)

    # remove worst 25% hits
    sorted_ij_assignment_distances  = np.sort(ij_lsa_distances)
    if outlier_threshold < 1:
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






def neighborgraph(distancemat, neighbors):

    # z= nbrs.kneighbors_graph(x,neighbors,mode='distance')
    # z= ut.zehidense(z)
    # z = np.stack((z,z.T), axis =2).max(axis=2)
    # # diff = z-z.T
    # # diff[diff > 0 ] = 0
    # # z-= diff
    # check_symmetric(z,raise_exception=True)
    # return z

    part = np.argpartition(distancemat, neighbors, axis = 1)[:,:neighbors]
    neighborsgraph = np.zeros_like(distancemat)
    np.put_along_axis(neighborsgraph, part, np.take(distancemat,part), axis = 1)

    return neighborsgraph




def symmetric_spanning_tree_neighborgraph(x, neighbors,add_tree=True):


    # min spanning tree
    distancemat = metrics.euclidean_distances(x)
    tree = minimum_spanning_tree(distancemat) if add_tree else np.zeros_like(distancemat)
    tree= ut.zehidense(tree)

    # faster version of neighborsgraph
    neighborsgraph = neighborgraph(distancemat, neighbors)


    # neighborsgraph = np.zeros_like(distancemat)
    # i_ids,j_ids, ij_lsa_distances = lin_asi_thresh(distancemat,neighbors,.8)
    # neighborsgraph[i_ids,j_ids] = ij_lsa_distances



    # combine and return
    combinedgraph = np.stack((neighborsgraph,neighborsgraph.T,tree,tree.T), axis =2)
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
                                intra_neigh=15,
                                inter_neigh = 1,
                                scaling_num_neighbors = 2,
                                outlier_threshold = .8,
                                scaling_threshold=.9,
                                dataset_adjacency = False,
                                add_tree = True, epsilon = 1e-6 ):

    if 'anndata' in str(type(Xlist[0])):
        Xlist = to_arrays(Xlist, base)

    def adjacent(i,j):
        if isinstance( dataset_adjacency, np.ndarray):
            return  dataset_adjacency[i][j] == 1
        else:
            return True

    def make_distance_matrix(ij):
            i,j = ij
            if i == j:
                # use the maximum between neighborgraph and min spanning tree to make sure all is connected
                r = symmetric_spanning_tree_neighborgraph(Xlist[i], intra_neigh, add_tree = add_tree)
                r = sparse.lil_matrix(r)
                return r


            elif not adjacent(i,j):
                # just fill with zeros :)
                return  sparse.lil_matrix((Xlist[i].shape[0],Xlist[j].shape[0]), dtype=np.float32),0,0
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

                i_ids,j_ids, ij_lsa_distances = lin_asi_thresh(ij_euclidian_distances,inter_neigh,outlier_threshold)

                # normalize

                # if scaling_threshold < 1:
                #     lsa_normalisation_factor = sorted_ij_assignment_distances[int(len(ij_lsa_distances)*scaling_threshold)]
                #     ij_lsa_distances /= lsa_normalisation_factor

                #
                #dm = (avgdist(Xlist[i], hoodsize)+avgdist(Xlist[j],hoodsize))/2
                #dab *=dm
                # average_knn_distance_factors = average_knn_distance(Xlist[i],Xlist[j], i_ids , j_ids, scaling_num_neighbors)
                # ij_lsa_distances *= average_knn_distance_factors
                # make a matrix
                res[i_ids,j_ids] = ij_lsa_distances
                return res

    n_datas = len(Xlist)
    tasks =  [(i,j) for i in range(n_datas) for j in range(i,n_datas)]
    parts = ut.xxmap( make_distance_matrix, tasks)
    getpart=dict(zip(tasks,parts))


    if False:
        # insane enhancement idea :D
        for i in range(n_datas):
            for j in range(n_datas):
                if i < j:
                    getpart[(i,j)] = steal_neighbors(getpart[(i,j)], getpart[(j,j)])

                    ## use the linsum targets as a source for new neighbors
                    #currentmatrix = getpart[(i,j)]
                    #sourceNeighborsFrom = getpart[(j,j)]
                    ## some are removed due to the outlier threshold rule..
                    #currentMatrixNonEmpty = [True if len(r) >0 else False for r in currentmatrix.rows]
                    ## overwrite  relevant rows
                    #targets = [r[0] for r in currentmatrix[currentMatrixNonEmpty].rows]
                    #new_neighbors = sourceNeighborsFrom[targets]
                    #selfdist = sparse.csr_matrix(currentmatrix)
                    ## selfdist.data = np.full_like(selfdist.data, epsilon) # overwrite distances with epsilon
                    #selfdist = selfdist[currentMatrixNonEmpty].astype(bool) * epsilon
                    #new_neighbors += selfdist
                    #getpart[(i,j)][currentMatrixNonEmpty] =  new_neighbors
                    ## so.heatmap(getpart[(i,j)].todense(),dim=(50,50))
                    ## part = np.argpartition(distancemat, neighbors, axis = 1)[:,:neighbors]
                    ## neighborsgraph = np.zeros_like(distancemat)
                    ## np.put_along_axis(neighborsgraph,part, np.take(distancemat,part), axis = 1)
                    ##so.heatmap(getpart[(i,j)].todense(),dim=(50,50))
                    ##print('##############')

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
    distance_matrix = sparse.vstack([sparse.hstack(col) for col in row])

    # check_symmetric(distance_matrix,raise_exception=True)
    # so.heatmap(distance_matrix.todense(), dim = (100,100))

    return  distance_matrix
