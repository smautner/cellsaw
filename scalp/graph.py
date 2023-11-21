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

def to_linear_assignment_graph(adatas,base = 'pca40',
                               intra_neigh = 15, inter_neigh = 1,
              scaling_num_neighbors = 2, outlier_threshold = .8,
              scaling_threshold = .9, dataset_adjacency = None, add_tree= True,):

    X = to_arrays(adatas, base)
    graph =  linear_assignment_integrate(X,
                                                         intra_neigh=intra_neigh,add_tree=add_tree,
                                                         inter_neigh = inter_neigh,
                                                         scaling_num_neighbors = scaling_num_neighbors,
                                                         outlier_threshold = outlier_threshold,
                                                         scaling_threshold=scaling_threshold,
                                                         dataset_adjacency=dataset_adjacency)

    return graph


def iterated_linear_sum_assignment(distances, repeats):
    def linear_sum_assignment_iteration(distances):
        r, c = linear_sum_assignment(distances)
        d = distances[r,c].copy()
        distances [r,c] = np.inf
        return r,c,d
    cc, rr, dist  = Transpose([ linear_sum_assignment_iteration(distances) for _ in range(repeats)  ])
    return np.hstack(cc) , np.hstack(rr), np.hstack(dist)

def repeated_subsample_linear_sum_assignment(distances, repeats, num_instances):
    # BUILDME
    def linear_sum_assignment_iteration(distances):
        r, c = linear_sum_assignment(distances)
        d = distances[r,c].copy()
        distances [r,c] = np.inf
        return r,c,d
    cc, rr, dist  = Transpose([ linear_sum_assignment_iteration(distances) for _ in range(repeats)  ])
    return np.hstack(cc) , np.hstack(rr), np.hstack(dist)



import ubergauss.tools as ut



def neighborgraph(x, neighbors):

    z= nbrs.kneighbors_graph(x,neighbors,mode='distance')
    z= ut.zehidense(z)
    z = np.stack((z,z.T), axis =2).max(axis=2)
    # diff = z-z.T
    # diff[diff > 0 ] = 0
    # z-= diff
    check_symmetric(z,raise_exception=True)
    return z


import structout as so

def symmetric_spanning_tree_neighborgraph(x, neighbors,add_tree=True):

    # neighbors graph
    neighborsgraph= nbrs.kneighbors_graph(x,neighbors,mode='distance')
    neighborsgraph.data += 0.0000001 # if we densify, connections disapear
    # so.lprint( [ len(x.data) for x in neighborsgraph] )
    neighborsgraph= ut.zehidense(neighborsgraph)

    # min spanning tree
    distancemat = metrics.euclidean_distances(x)
    tree = minimum_spanning_tree(distancemat) if add_tree else neighborsgraph
    tree= ut.zehidense(tree)

    # combine and return
    combinedgraph = np.stack((neighborsgraph,neighborsgraph.T,tree,tree.T), axis =2)
    combinedgraph = combinedgraph.max(axis=2)
    check_symmetric(combinedgraph,raise_exception=True)
    return combinedgraph



def avgdist(a,numneigh = 2):
    nbrs = NearestNeighbors(n_neighbors=1+numneigh).fit(a)
    distances, indices = nbrs.kneighbors(a)
    return np.mean(distances[:,1:], axis = 1)


def average_knn_distance(I,J,i_ids,j_ids,numneigh):
    d1  = avgdist(I,numneigh)[i_ids]
    d2  = avgdist(J,numneigh)[j_ids]
    stack = np.vstack((d1,d2))
    return np.mean(stack, axis=0).T


def linear_assignment_integrate(Xlist,
                                intra_neigh=15,
                                inter_neigh = 1,
                                scaling_num_neighbors = 2,
                                outlier_threshold = .8,
                                scaling_threshold=.9,
                                dataset_adjacency = False,
                                add_tree = True,
                                showtime = False):

    lsatime = 0.0
    eutime = 0.0



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
                r = sparse.lil_matrix(r),0,0
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
                eustart=time.time()
                ij_euclidian_distances= metrics.euclidean_distances(Xlist[i],Xlist[j])
                eutime = time.time() - eustart
                res = sparse.lil_matrix(ij_euclidian_distances.shape,dtype=np.float32)
                if inter_neigh==0:
                    return res

                lsastart=time.time()
                i_ids,j_ids, ij_lsa_distances = iterated_linear_sum_assignment(ij_euclidian_distances,inter_neigh)
                lsatime=(time.time()-lsastart)

                # remove worst 25% hits
                sorted_ij_assignment_distances  = np.sort(ij_lsa_distances)
                if outlier_threshold < 1:
                    lsa_outlier_thresh = sorted_ij_assignment_distances[int(len(ij_lsa_distances)*outlier_threshold)]
                    outlier_ids = ij_lsa_distances >  lsa_outlier_thresh
                    ij_lsa_distances[outlier_ids] = 0

                # normalize
                if scaling_threshold < 1:
                    lsa_normalisation_factor = sorted_ij_assignment_distances[int(len(ij_lsa_distances)*scaling_threshold)]
                    ij_lsa_distances /= lsa_normalisation_factor
                #
                #dm = (avgdist(Xlist[i], hoodsize)+avgdist(Xlist[j],hoodsize))/2
                #dab *=dm
                average_knn_distance_factors = average_knn_distance(Xlist[i],Xlist[j], i_ids , j_ids, scaling_num_neighbors)
                ij_lsa_distances *= average_knn_distance_factors


                # make a matrix
                res[i_ids,j_ids] = ij_lsa_distances
                return res, lsatime, eutime

    n_datas = len(Xlist)
    tasks =  [(i,j) for i in range(n_datas) for j in range(i,n_datas)]
    parts = ut.xxmap( make_distance_matrix, tasks)
    getpart=dict(zip(tasks,parts))

    # then we built a row:
    row = []
    for i in range(n_datas):
        col = []
        for j in range(n_datas):
            if i <= j:
                distance_matrix,a,b = getpart[(i,j)]
                lsatime+=a
                eutime += b
            else:
                distance_matrix = row[j][i].T

            col.append(distance_matrix)
        row.append(col)

    distance_matrix = sparse.vstack([sparse.hstack(col) for col in row])

    if showtime:
        print(f"{eutime=}")
        print(f"{lsatime=}")
    check_symmetric(distance_matrix,raise_exception=True)
    return  distance_matrix
