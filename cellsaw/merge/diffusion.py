from lmz import Map,Zip,Filter,Grouper,Range,Transpose


from scipy.optimize import linear_sum_assignment
from sklearn import neighbors as nbrs, metrics
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from scipy.sparse.csgraph import dijkstra
from ubergauss import tools




def iterated_linear_sum_assignment(distances, repeats):
    def linear_sum_assignment_iteration(distances):
        r, c = linear_sum_assignment(distances)
        distances [r,c] = np.inf
        return r,c
    cc, rr  = Transpose([ linear_sum_assignment_iteration(distances) for _ in range(repeats)  ])
    return np.hstack(cc) , np.hstack(rr)

def linear_sum_assignment_matrices(x1, x2, repeats):
        x= metrics.euclidean_distances(x1,x2)
        r = np.zeros_like(x)
        a,b = iterated_linear_sum_assignment(x, repeats)
        r[a,b] = 1
        r2 = np.zeros((x.shape[1], x.shape[0]))
        r2[b,a] = 1 # rorated :)
        return r,r2


def linear_assignment_kernel(x1len=False, neighbors = 3, neighbors_inter= 1,  X=None, _=None, return_graph = False):
    assert x1len, 'need to know how large the first dataset ist :)'
    '''
    X are the stacked projections[0] (normalized read matrices)
    since this is a kernel, we return a similarity matrix

    - we can split it by 2 to get the original projections
    - we do neighbors to get quadrant 2 and 4
    - we do hungarian to do quadrants 1 and 3
    - we do dijkstra to get a complete distance matrix

    '''
    x1,x2 = np.split(X,[x1len])
    q2 = nbrs.kneighbors_graph(x1,neighbors).todense()
    q4 = nbrs.kneighbors_graph(x2,neighbors).todense()
    q1,q3 = linear_sum_assignment_matrices(x1,x2, neighbors_inter)

    q1 = q1*.01
    q3 = q3*.01
    graph = np.hstack((np.vstack((q2,q3)),np.vstack((q1,q4))))

    '''
    connect = dijkstra(graph, unweighted = True, directed = False)
    if return_graph:
        return graph, connect
    distances = -connect # invert
    distances -= distances.min() # longest = 4
    distances /= distances.max() # between 0 and 1 :)
    distances[distances < np.median(np.unique(distances))] = 0
    '''

    return  graph #np.power(distances,2)



class Diffusion:

    def __init__(self,
                n_neighbors_intra = 7,
                n_neighbors_inter=1,
                lp_model= LabelSpreading(kernel = None, alpha=.8, max_iter=1000), kernel = linear_assignment_kernel):

        """we just run diffusion as sklearn would i.e. expect no string labels etc"""

        self.neighbors_intra = n_neighbors_intra
        self.neighbors_inter = n_neighbors_inter
        self.lp_model = lp_model
        self.kernel = kernel


    def fit(self,X,y):
        self.X = X
        self.y = y
        self.train_ncells = X.shape[0]
        kernel = lambda x, y: self.kernel(self.train_ncells, self.neighbors_intra, self.neighbors_inter, x, y)
        self.lp_model.set_param(kernel = kernel)


    def predict(self,X,y = False):
        if not y:
            y = np.full(X.shape[0],-1)
        assert self.X.shape == X.shape, 'not sure why i assert this... probably because i havent tested it'

        Ystack = np.hstack((self.y,y))
        Xstack = np.vstack(Map(tools.zehidense, (self.X,X)))

        all_labels = self.lp_model.fit(Xstack, Ystack).transduction_

        self.correctedlabels = all_labels[self.train_ncells]
        return  all_labels[self.train_ncells:]


def stringdiffuse(mergething, labels, pid = 1,
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


    diffusor = diffusion(neighbors_inter = neighbors_inter, neighbors_intra  = neighbors_intra)
    diffusor.fit(mergething.projections[pid][0], sm.encode(labels))

    intresults =  diffusor.predict(mergething.projections[pid][1])
    return sm.decode(intresults)





