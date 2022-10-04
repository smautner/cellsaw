from lmz import Map,Zip,Filter,Grouper,Range,Transpose
from pprint import pprint

from scipy.optimize import linear_sum_assignment
from sklearn import neighbors as nbrs, metrics
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from ubergauss import tools
import seaborn as sns
from matplotlib import pyplot as plt


####################
#  LINEAR SUM ASSIGNMENT
####################

def iterated_linear_sum_assignment(distances, repeats):
    def linear_sum_assignment_iteration(distances):
        r, c = linear_sum_assignment(distances)
        d = distances[r,c].copy()
        distances [r,c] = np.inf
        return r,c,d

    cc, rr, dist  = Transpose([ linear_sum_assignment_iteration(distances) for _ in range(repeats)  ])
    return np.hstack(cc) , np.hstack(rr), np.hstack(dist)



def linear_sum_assignment_matrices(x1, x2, repeats,
                                   dist = True,
                                   dense = True):
        x= metrics.euclidean_distances(x1,x2)
        a,b, distances = iterated_linear_sum_assignment(x, repeats)
        r = np.zeros(shape) if dense else sparse.csr_matrix(x.shape, dtype=np.float32)
        # print(f"{a.shape=} {b.shape=} {distances.shape=}]")
        # print(f"{a=} {b=} {distances=}]")
        r[a,b] = (distances + 0.0001) if dist else 1


        return r,r.T

#######
# 1nn distance
###########
def avg1nndistance(listofdistancemat):
    def calculate1nn_distance(q):
        return np.mean([ np.min(q[q!=0])  for row in q])
    list1nn_means = Map(calculate1nn_distance,listofdistancemat)
    return np.mean(list1nn_means)


###################
# graphbla
##################


def neighborgraph(x, neighbors):
    z= nbrs.kneighbors_graph(x,neighbors,mode='distance')
    diff = z-z.T
    diff[diff > 0 ] = 0
    z-= diff
    return z


from matplotlib import pyplot as plt
def linear_assignment_kernel(x1,x2, neighbors = 3,
        neighbors_inter= 1, sigmafac = 1, linear_assignment_factor = 1):


    '''
    X are the stacked projections[0] (normalized read matrices)
    since this is a kernel, we return a similarity matrix

    - we can split it by 2 to get the original projections
    - we do neighbors to get quadrant 2 and 4
    - we do hungarian to do quadrants 1 and 3
    - we do dijkstra to get a complete distance matrix

    '''
    q1,q3 = linear_sum_assignment_matrices(x1,x2, neighbors_inter,
                                            dist = True,
                                            dense = False)

    q2,q4 = [neighborgraph(x,neighbors) for x in [x1,x2]]



    q1 = q1*linear_assignment_factor
    q3 = q3*linear_assignment_factor




    distance_matrix = sparse.hstack((sparse.vstack((q2,q3)),sparse.vstack((q1,q4)))).todense()


    # print(f"raw dist")
    # sns.heatmap(distance_matrix);plt.show()


    distance_matrix = dijkstra(distance_matrix, directed = False)
    dijkstraQ1 = distance_matrix[:x1.shape[0],x1.shape[0]:]

    sigma = avg1nndistance([q2,q4])*sigmafac
    similarity_matrix = np.exp(-dijkstraQ1/sigma)

    # print('dijkstra zoom');sns.heatmap(dijkstraQ1); plt.xlabel('target'); plt.show()
    # print(f'gaussed  sigme:{sigma}')
    # sns.heatmap(similarity_matrix); plt.show()

    return  similarity_matrix


