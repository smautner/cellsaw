

from scipy.optimize import linear_sum_assignment as lsa
from sklearn import neighbors as nbrs, metrics
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from scipy.sparse.csgraph import dijkstra
from lmz import Map
from ubergauss import tools


def diffuse(mergething, labelslist, pid = 1, neighbors = 30):


    Xlist = mergething.projections[pid]

    #def diffuseandscore(y1, y2, Xlist, neighbors=30):
    assert Xlist[0].shape == Xlist[1].shape, 'not sure why i assert this... probably because i havent tested it'



    y1len = Xlist[0].shape[0]

    lp_model = LabelSpreading(kernel=lambda x, y: mykernel(y1len, neighbors, x, y),
                              alpha=.2,
                              max_iter=30)

    startlabels = np.hstack(labelslist)
    args = np.vstack(Map(tools.zehidense, Xlist)), startlabels

    try:
        lp_model.fit(*args)
    except:
        print(args[0].shape)
        print(args[1].shape)

    all_labels = lp_model.transduction_
    return all_labels[:y1len], all_labels[y1len:]








def hungmat(x1, x2):
        x= metrics.euclidean_distances(x1,x2)
        r = np.zeros_like(x)
        a,b = lsa(x)
        r[a,b] = 1
        r2 = np.zeros((x.shape[1], x.shape[0]))
        r2[b,a] = 1 # rorated :)
        return r,r2


def mykernel(x1len=False, neighbors = 3, X=None, _=None, return_graph = False):
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
    q1,q3 = hungmat(x1,x2)

    graph = np.hstack((np.vstack((q2,q3)),np.vstack((q1,q4))))


    connect = dijkstra(graph,unweighted = True, directed = False)

    if return_graph:
        return graph, connect
    distances = -connect # invert
    distances -= distances.min() # longest = 4
    distances /= distances.max() # between 0 and 1 :)
    distances[distances < np.median(np.unique(distances))] = 0

    return np.power(distances,2)


