from lmz import Map,Zip,Filter,Grouper,Range,Transpose
from pprint import pprint

from scipy.optimize import linear_sum_assignment
from sklearn import neighbors as nbrs, metrics
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from ubergauss import tools
import seaborn as sns




def iterated_linear_sum_assignment(distances, repeats):
    def linear_sum_assignment_iteration(distances):
        r, c = linear_sum_assignment(distances)
        d = distances[r,c].copy()
        distances [r,c] = np.inf
        return r,c,d

    cc, rr, dist  = Transpose([ linear_sum_assignment_iteration(distances) for _ in range(repeats)  ])
    return np.hstack(cc) , np.hstack(rr), np.hstack(dist)



def linear_sum_assignment_matrices(x1, x2, repeats, dist = True, dense = True, debug = False):
        x= metrics.euclidean_distances(x1,x2)
        a,b, distances = iterated_linear_sum_assignment(x, repeats)


        def mkmtx(shape):
            return  np.zeros(shape) if dense else sparse.csr_matrix(shape, dtype=np.float32)

        r = mkmtx(x.shape)
        distance_a = distances[a]
        r[a,b] = distance_a if dist else 1

        # TODO can i return the transpose?
        r2 = mkmtx((x.shape[1], x.shape[0]))
        r2[b,a] = distance_a if dist else 1

        if debug:
            pprint(Zip(a,b,distances))
        return r,r2

from matplotlib import pyplot as plt
def linear_assignment_kernel(x1,x2, neighbors = 3,
        neighbors_inter= 1, sigmafac = 1):


    '''
    X are the stacked projections[0] (normalized read matrices)
    since this is a kernel, we return a similarity matrix

    - we can split it by 2 to get the original projections
    - we do neighbors to get quadrant 2 and 4
    - we do hungarian to do quadrants 1 and 3
    - we do dijkstra to get a complete distance matrix

    '''
    # TODO use LabelPropagation
    # TODO try uneven length x1 and x2
    # TODO disappear this branch
    if id(x1) == id(x2):
        # just apply gaussian kernel
        print("SAME OBJECT")


        distance_matrix = nbrs.kneighbors_graph(x1,
                neighbors,
                mode='distance').todense()
        # sns.heatmap(distance_matrix);plt.show()
        distance_matrix[distance_matrix==0] = np.inf
        np.fill_diagonal(distance_matrix, 0)

        sigma = sigmafac * np.mean([ np.min(row[row!=0]) for row in distance_matrix])
        similarity_matrix = np.exp(-distance_matrix/sigma)
        # sns.heatmap(similarity_matrix);plt.show()
        return similarity_matrix

    # get the quadrants
    # TODO make everything sparse
    q2 = nbrs.kneighbors_graph(x1,neighbors,mode='distance').todense()
    q4 = nbrs.kneighbors_graph(x2,neighbors,mode ='distance').todense()
    # q2= metrics.euclidean_distances(x1)
    # q4= metrics.euclidean_distances(x2)
    q1,q3 = linear_sum_assignment_matrices(x1,x2, neighbors_inter,
                                            dist = True,
                                            dense = True,
                                            debug = True)


    # TODO make a factor here and expose it
    q1 = q1
    q3 = q3

    for a in [q1,q2,q3,q4]:
        a[a==0] = np.inf



    # TODO make neighbos symetric!
    # combine and fill missing
    # distance_matrix = sparse.hstack((sparse.vstack((q2,q3)),sparse.vstack((q1,q4)))).todense()
    distance_matrix = np.hstack((np.vstack((q2,q3)),np.vstack((q1,q4))))


    print(f"raw dist")
    sns.heatmap(distance_matrix);plt.show()

    sigma = sigmafac* np.mean([ np.min(row[row!=0]) for row in distance_matrix])
    print(f"sigme:{sigma}")
    # TODO READ dijkstra documentation...
    distance_matrix = dijkstra(distance_matrix)
    print('dijkstra');sns.heatmap(distance_matrix);  plt.show()
    distance_matrix = distance_matrix[:x1.shape[0],x1.shape[0]:]
    print('dijkstra zoom');sns.heatmap(distance_matrix); plt.xlabel('target'); plt.show()
    similarity_matrix = np.exp(-distance_matrix/sigma)

    # np.fill_diagonal(similarity_matrix,1)
    print('gaussed')
    # print(similarity_matrix)
    sns.heatmap(similarity_matrix); plt.show()

    return  similarity_matrix


