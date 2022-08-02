from cellsaw.similarity.measures import cosine, jaccard, precision
import numpy as np
from ubergauss import tools as ut
from cellsaw.similarity import draw

def matrixmap(method, instances, repeats = 5):
    l = len(instances)
    res = np.zeros((l,l,repeats))
    for i,obj_i in enumerate(instances):
        for j,obj_j in enumerate(instances[i:]):
            r = [method(obj_i,obj_j,x) for x in range(repeats)]
            for x,val in enumerate(r):
                res[i,j,x] = val
                res[j,i,x] = val
    return res


def matrixmap_mp(method, instances, repeats = 5):
    # pool.maxtasksperchild
    l = len(instances)
    res = np.zeros((l,l,repeats))

    def func(stuff):
        a,b,seeds,i,j = stuff
        r = [ method(a,b,seed)  for seed in seeds ]
        return r,i,j

    tmptasks =([instances[i],instances[j],list(range(repeats)), i,j]
               for i in range(l) for j in range(i+1,l))

    for r,i,j in ut.xmap(func,tmptasks):
        for x,val in enumerate(r):
            res[i,j,x] = val
            res[j,i,x] = val

    return res


def getNeighbors(matrix, labels, k = 2 ):
    sm = ut.spacemap(labels)
    codes = getnn(matrix,k)
    res = [[sm.getitem[z] for z in a] for a in codes]
    return res

def getnn(m,n):
    np.fill_diagonal(m, np.NINF)
    srt=  np.argsort(m, axis= 1)
    return [ [i]+srt[i,-n:].tolist() for i in range(srt.shape[0]) ]

