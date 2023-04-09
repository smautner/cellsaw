from lmz import Map,Zip,Filter,Grouper,Range,Transpose
from scipy.sparse.csgraph import minimum_spanning_tree
from pprint import pprint
import time
from scipy.optimize import linear_sum_assignment
from sklearn import neighbors as nbrs, metrics
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from ubergauss import tools
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.utils import check_symmetric


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
    r = np.zeros(x.shape) if dense else sparse.csr_matrix(x.shape, dtype=np.float32)
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
    check_symmetric(z,raise_exception=True)

    return z


from matplotlib import pyplot as plt
def linear_assignment_kernel(x1,x2, neighbors = 3,
                             neighbors_inter= 1, sigmafac = 1, linear_assignment_factor = 1, return_dm = False):


    '''
    X are the stacked projections[0] (normalized read matrices)
    since this is a kernel, we return a similarity matrix

    - we do neighbors to get quadrant 2 and 4
    - we do hungarian to do quadrants 1 and 3
    - we do dijkstra to get a complete distance matrix

    - dijkstra wants dense matrices, so we can not go sparse

    '''
    q1,q3 = linear_sum_assignment_matrices(x1,x2, neighbors_inter,
                                           dist = True,
                                           dense = True)

    q2,q4 = [neighborgraph(x,neighbors).todense() for x in [x1,x2]]


    q1 = q1*linear_assignment_factor
    q3 = q3*linear_assignment_factor

    #distance_matrix = sparse.hstack((sparse.vstack((q2,q3)),sparse.vstack((q1,q4)))).todense()
    distance_matrix = np.hstack((np.vstack((q2,q3)),np.vstack((q1,q4))))


    # print(f"raw dist")
    # sns.heatmap(distance_matrix);plt.show()


    distance_matrix = dijkstra(distance_matrix, directed = False)

    if return_dm:
        return distance_matrix

    dijkstraQ1 = distance_matrix[:x1.shape[0],x1.shape[0]:]

    sigma = avg1nndistance([q2,q4])*sigmafac
    similarity_matrix = np.exp(-dijkstraQ1/sigma)

    # print('dijkstra zoom');sns.heatmap(dijkstraQ1); plt.xlabel('target'); plt.show()
    # print(f'gaussed  sigme:{sigma}')
    # sns.heatmap(similarity_matrix); plt.show()

    return  similarity_matrix

def linear_assignment_kernel_XXX(x1,x2, neighbors = 3,
                                 neighbors_inter= 1, sigmafac = 1, linear_assignment_factor = 1, tsizes = None):


    '''
    now we have multiple target datasets:
        i guess we build the target line by line
    '''

    # rist we split x2:

    dslist = [x1]+np.split(x2,np.add.accumulate(tsizes))

    if dslist[2].shape[0] == 0: # happens on self similarity
        dslist = dslist[:2]
    if dslist[-1].shape[0] == 0: # np splitting likes to add 0-frames when things fit perfectly :(
        dslist = dslist[:-1]

    # then we built a row:
    diaglist = []
    row = []
    for i in range(len(dslist)):
        col = []
        for j in range(len(dslist)):
            if i == j:
                block = neighborgraph(dslist[i],neighbors).todense()
                diaglist.append(block)
            if (i+1) == j:
                block,_ = linear_sum_assignment_matrices(dslist[i],dslist[j], neighbors_inter, dist = True, dense = True)
                block*=linear_assignment_factor
            if (j+1) == i:
                _,block = linear_sum_assignment_matrices(dslist[i],dslist[j], neighbors_inter, dist = True, dense = True)
                block*=linear_assignment_factor
            else:
                np.zeros((dslist[i].shape[0],dslist[j].shape[0]))
            col.append(block)
        row.append(np.hstack(col))
    distance_matrix = np.vstack(row)
    distance_matrix = dijkstra(distance_matrix, directed = False)
    dijkstraQ1 = distance_matrix[:x1.shape[0],x1.shape[0]:]
    sigma = avg1nndistance(diaglist)*sigmafac
    similarity_matrix = np.exp(-dijkstraQ1/sigma)

    # print('dijkstra zoom');sns.heatmap(dijkstraQ1); plt.xlabel('target'); plt.show()
    # print(f'gaussed  sigme:{sigma}')
    # sns.heatmap(similarity_matrix); plt.show()

    return  similarity_matrix





from sklearn.neighbors import NearestNeighbors
from cellsaw.merge import mergehelpers


def avgdist(a,numneigh = 2):
    nbrs = NearestNeighbors(n_neighbors=1+numneigh).fit(a)
    distances, indices = nbrs.kneighbors(a)
    return np.mean(distances[:,1:], axis = 1)

def average_knn_distance(I,J,i_ids,j_ids,numneigh):
    d1  = avgdist(I,numneigh)[i_ids]
    d2  = avgdist(J,numneigh)[j_ids]
    stack = np.vstack((d1,d2))
    return np.mean(stack, axis=0).T

from sklearn.neighbors import NearestNeighbors
def neighborgraph_p_weird(x, neighbors):
    # neighbors = max(1, int(x.shape[0]*(neighbors_perc/100)))
    z= nbrs.kneighbors_graph(x,neighbors)
    diff = z-z.T
    diff[diff > 0 ] = 0
    z-= diff
    return z

def neighborgraph_p_real(x, neighbors):
    z = np.zeros_like(x)
    np.fill_diagonal(x,np.NINF)
    for i,row in enumerate(x):
        sr = np.argsort(row)
        z[i][sr[-neighbors:]]  = 1
    diff = z-z.T
    diff[diff > 0 ] = 0
    z-= diff
    return z





def make_adjacency(similarity, algo=0, neighbors=10):
        n_perc = neighbors = max(1, int(similarity.shape[0]*(neighbors/100)))
        simm = neighborgraph_p_weird if algo == 1 else neighborgraph_p_real
        return simm(similarity, n_perc)

def merge_adjacency(*args):
    if len(args) == 1:
        return args[0]
    return np.logical_or(args[0],merge_adjacency(*args[1:]))

def make_star(size = 5, center =2):
    ret = np.zeros((size,size))
    ret[center] = 1
    ret[:,center] = 1
    return ret

def make_sequence(size = 5, indices = [0,1]):
    ret = np.zeros((size,size))
    for i in range(size):
        for j in indices:
            if i+j < size and i+j >= 0:ret[i,i+j] = 1
    return ret


def test_matrixmaker():
    m = make_star(size = 5, center = 1)
    s = make_sequence(size=5, indices=[0,1,2])
    print(s,m,merge_adjacency(s,m))


def linear_assignment_integrate(Xlist,
                                intra_neigh=15,
                                inter_neigh = 1,
                                scaling_num_neighbors = 2,
                                outlier_threshold = .8,
                                scaling_threshold=.9,
                                dataset_adjacency = False,
                                showtime = False):

    lsatime = 0.0
    eutime = 0.0



    def adjacent(i,j):
        if isinstance( dataset_adjacency, np.ndarray):
            return  dataset_adjacency[i][j] == 1
        else:
            return True

    def make_distance_matrix(i,j):
            if i == j:
                # use the maximum between neighborgraph and min spanning tree to make sure all is connected
                neighborAdj = sparse.lil_matrix(neighborgraph(Xlist[i],intra_neigh))
                distancemat = metrics.euclidean_distances(Xlist[i])
                tree = sparse.lil_matrix(minimum_spanning_tree(distancemat))
                return tree.maximum(neighborAdj), 0,0


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
                lsa_outlier_thresh = sorted_ij_assignment_distances[int(len(ij_lsa_distances)*outlier_threshold)]
                outlier_ids = ij_lsa_distances >  lsa_outlier_thresh
                ij_lsa_distances[outlier_ids] = 0

                # normalize
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


    # then we built a row:
    row = []
    for i in range(len(Xlist)):
        col = []
        for j in range(len(Xlist)):
            if i <= j:
                distance_matrix,a,b = make_distance_matrix(i,j)
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
    return  distance_matrix




def KNNFormater(Data, precomputedKNNIndices, precomputedKNNDistances):
    from pynndescent import NNDescent
    # print('asd1')
    pyNNDobject = NNDescent(np.vstack(Data), metric='euclidean', random_state=1337,n_jobs = 1)
    # print('asd2')
    pyNNDobject._neighbor_graph = (precomputedKNNIndices.copy(), precomputedKNNDistances.copy())
    precomputedKNN = (precomputedKNNIndices, precomputedKNNDistances, pyNNDobject)
    return precomputedKNN

def stack_n_fill(a,val):
    '''
    reformats the data
    '''
    maxlen = max(Map(len,a))
    res = np.full((len(a),maxlen),val)
    for i,val in enumerate(a):
        res[i,:len(val)] = val
    return res



from umap import UMAP
def distmatrixumap(dataXlist,dm,components = 10):
    sparseMatrix = sparse.csr_matrix(dm)
    precomputedKNNIndices = []
    precomputedKNNDistances = []
    # for ip in range(len(sparseMatrix.indptr)-1):
    #         start = sparseMatrix.indptr[ip]
    #         end = sparseMatrix.indptr[ip+1]
    #         precomputedKNNIndices.append(sparseMatrix.indices[start:end])
    #         precomputedKNNDistances.append(sparseMatrix.data[start:end])
    for row in sparseMatrix:
        precomputedKNNIndices.append(row.indices)
        precomputedKNNDistances.append(row.data)
    umapknn = stack_n_fill(precomputedKNNIndices,-1), stack_n_fill( precomputedKNNDistances,np.inf)


    precomputedKNN = KNNFormater(dataXlist, *umapknn)
    n_neighbors = precomputedKNN[0].shape[1]

    mymap = UMAP(n_components=components, #Dimensions to reduce to
                 n_neighbors=n_neighbors,
                 random_state=1337,
                 metric='euclidean',
                 precomputed_knn=precomputedKNN,
                 force_approximation_algorithm=True)
    r=  mymap.fit_transform(np.vstack(dataXlist))
    return r


