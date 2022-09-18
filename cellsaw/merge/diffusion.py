from lmz import Map,Zip,Filter,Grouper,Range,Transpose


from scipy.optimize import linear_sum_assignment
from sklearn import neighbors as nbrs, metrics
import numpy as np
from scipy import sparse
from sklearn.semi_supervised import LabelSpreading
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



def linear_sum_assignment_matrices(x1, x2, repeats, dist = True, dense = True):
        x= metrics.euclidean_distances(x1,x2)
        a,b, distances = iterated_linear_sum_assignment(x, repeats)


        def mkmtx(shape):
            return  np.zeros(shape) if dense else sparse.csr_matrix(shape, dtype=np.float32)

        r = mkmtx(x.shape)
        r[a,b] = distances[a] if dist else 1

        r2 = mkmtx((x.shape[1], x.shape[0]))
        r2[b,a] = distances[a] if dist else 1
        return r,r2

"""
def linear_assignment_kernel_old(X1,X2,x1len=False, neighbors = 3,
        neighbors_inter= 1, sigmafac = 1):

    assert x1len, 'need to know how large the first dataset ist :)'

    '''
    X are the stacked projections[0] (normalized read matrices)
    since this is a kernel, we return a similarity matrix

    - we can split it by 2 to get the original projections
    - we do neighbors to get quadrant 2 and 4
    - we do hungarian to do quadrants 1 and 3
    - we do dijkstra to get a complete distance matrix

    '''
    x1,x2 = np.split(X1,[x1len])

    q2 = nbrs.kneighbors_graph(x1,neighbors,mode='distance')
    q4 = nbrs.kneighbors_graph(x2,neighbors,mode ='distance')
    #q2 = nbrs.kneighbors_graph(x1,neighbors,mode='distance').todense()
    #q4 = nbrs.kneighbors_graph(x2,neighbors,mode ='distance').todense()
    q1,q3 = linear_sum_assignment_matrices(x1,x2, neighbors_inter, dist = True)

    # q2avg = np.mean([ np.min(row[row>0]) for row in q2])
    # q4avg = np.mean([ np.min(row[row>0]) for row in q4])
    # q2avg = np.mean(q2.data)
    # q4avg = np.mean(q4.data)
    # avgsim = 1/np.mean([q2avg,q4avg])

    distance_matrix = sparse.hstack((sparse.vstack((q2,q3)),sparse.vstack((q1,q4)))).todense()
    distance_matrix = dijkstra(distance_matrix)

    # dist1nn = lambda x: np.mean([ np.min(row[row!=0]) for row in x.todense()])
    # def dist_to_sim(x):
    #     x.data = np.exp(-x.data/(sigmafac*dist1nn(x)))  # or use cosine in the calculation
    #     return x

    # q1,q2,q3,q4 = [ dist_to_sim(x) for x in [q1,q2,q3,q4]]
    # averages = tools.Map(dist1nn,[q2,q4])
    # sigma  = sigmafac * np.mean(averages)
    # similarity_matrix = np.exp(-distance_matrix/sigma)
    # similarity_matrix = dist_to_sim(distance_matrix).todense()

    sigma = sigmafac* np.mean([ np.min(row[row!=0]) for row in distance_matrix])
    similarity_matrix = np.exp(-distance_matrix/sigma)
    np.fill_diagonal(similarity_matrix,1)

    #q1.data = np.full_like(q1.data,avgsim)
    #q3.data = np.full_like(q3.data,avgsim)
    #similarity_matrix = sparse.hstack((sparse.vstack((q2,q3)),sparse.vstack((q1,q4))))
    #graph = dijkstra(graph)
    sns.heatmap(similarity_matrix)
    return  similarity_matrix

"""
from matplotlib import pyplot as plt
def linear_assignment_kernel(X1,X2,x1len=False, neighbors = 3,
        neighbors_inter= 1, sigmafac = 1):

    assert x1len, 'need to know how large the first dataset ist :)'

    '''
    X are the stacked projections[0] (normalized read matrices)
    since this is a kernel, we return a similarity matrix

    - we can split it by 2 to get the original projections
    - we do neighbors to get quadrant 2 and 4
    - we do hungarian to do quadrants 1 and 3
    - we do dijkstra to get a complete distance matrix

    '''
    x1,x2 = np.split(X1,[x1len])


    # get the quadrants
    q2 = nbrs.kneighbors_graph(x1,neighbors,mode='distance').todense()
    q4 = nbrs.kneighbors_graph(x2,neighbors,mode ='distance').todense()

    # q2= metrics.euclidean_distances(x1)
    # q4= metrics.euclidean_distances(x2)

    q1,q3 = linear_sum_assignment_matrices(x1,x2, neighbors_inter, dist = True, dense = True)


    for a in [q1,q2,q3,q4]:
        a[a==0] = np.inf




    # combine and fill missing
    # distance_matrix = sparse.hstack((sparse.vstack((q2,q3)),sparse.vstack((q1,q4)))).todense()
    distance_matrix = np.hstack((np.vstack((q2,q3)),np.vstack((q1,q4))))



    print(f"raw dist")
    sns.heatmap(distance_matrix);plt.show()


    # distance_matrix = dijkstra(distance_matrix)
    # print(f"dijk")
    # sns.heatmap(distance_matrix);plt.show()


    # gaussian kernel...
    sigma = sigmafac* np.mean([ np.min(row[row!=0]) for row in distance_matrix])
    similarity_matrix = np.exp(-distance_matrix/sigma)


    #np.fill_diagonal(similarity_matrix,1)
    print('gauzzed')
    # print(similarity_matrix)
    sns.heatmap(similarity_matrix)
    return  similarity_matrix #np.power(distances,2)



class Diffusion:

    def __init__(self,
                n_neighbors_intra = 7,
                n_neighbors_inter=1,
                sigmafac = 1,
                lp_model= LabelSpreading(kernel = None, alpha=.8, max_iter=1000),
                kernel = linear_assignment_kernel):

        """we just run diffusion as sklearn would i.e. expect no string labels etc"""

        self.neighbors_intra = n_neighbors_intra
        self.neighbors_inter = n_neighbors_inter
        self.lp_model = lp_model
        self.kernel = kernel
        self.sigmafac = sigmafac


    def fit(self,X,y):
        self.X = X
        self.y = np.array(y)
        self.train_ncells = X.shape[0]
        kernel = lambda x1, x2: self.kernel(x1,x2,x1len=self.train_ncells,
                                            neighbors = self.neighbors_intra,
                                            neighbors_inter = self.neighbors_inter,
                                            sigmafac = self.sigmafac)
        self.lp_model.set_params(kernel = kernel)


    def predict(self,X,y = False):
        if not y:
            y = np.full(X.shape[0],-1)
        assert self.X.shape == X.shape, 'not sure why i assert this... probably because i havent tested it'
        assert self.y.shape == y.shape, 'assert everything!'


        # self.lp_model.fit(self.X, self.y)
        # return self.lp_model.predict(X)

        Ystack = np.hstack((self.y,y))
        Xstack = np.vstack(Map(tools.zehidense, (self.X,X)))
        all_labels = self.lp_model.fit(Xstack, Ystack).transduction_
        self.correctedlabels = all_labels[self.train_ncells]
        return  all_labels[self.train_ncells:]



def stringdiffuse(mergething, labels, pid = 1,
                                sigmafac = 1,
                              neighbors_inter = 1,
                              neighbors_intra = 7):

    '''
    will use the first dataset to train and return prediction on the second
    '''

    # the data i use has theese properties:
    # -1 -> no true label
    # "Unknown" -> it was decided that we treat this as a normal label
    # "pangalo error" -> does not happen often, will be treated as normal label
    # thus encoding labels like this is fine:
    sm = tools.spacemap(
            np.unique(
                [xx for xx in labels if isinstance(xx, str)]
            ))


    diffusor = Diffusion(n_neighbors_inter = neighbors_inter,sigmafac= sigmafac, n_neighbors_intra  = neighbors_intra)
    diffusor.fit(mergething.projections[pid][0], sm.encode(labels))

    intresults =  diffusor.predict(mergething.projections[pid][1])
    return sm.decode(intresults)





