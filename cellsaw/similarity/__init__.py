from lmz import Map,Zip,Filter,Grouper,Range,Transpose
from cellsaw.similarity.measures import cosine, jaccard, precision
import numpy as np
from ubergauss import tools as ut
from cellsaw.similarity import draw
import pandas as pd

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


def neighbors(loader, sim = cosine , filenames=[], kNeighbors = 1):
    simFunc = lambda a,b,c: sim(loader(a),loader(b))
    distances = matrixmap_mp(simFunc,filenames,repeats = 2)
    distances = np.mean(distances, axis =2)
    return getNeighbors(distances,filenames,k=kNeighbors)


def matrixmap_odd(method, targets,sources, repeats = 5):
    # instances not symmetrical
    l = len(targets)
    l2 = len(sources)
    res = np.zeros((l,l2,repeats))

    def func(stuff):
        a,b,seeds,i,j = stuff
        r = [ method(a,b,seed)  for seed in seeds ]
        return r,i,j

    tmptasks =([targets[i],sources[j],list(range(repeats)), i,j]
               for i in range(l) for j in range(l2))

    for r,i,j in ut.xmap(func,tmptasks):
        for x,val in enumerate(r):
            res[i,j,x] = val

    return res



'''
ranked_datasets_list, similarity_df = rank_by_similarity(target = target_datasets,
                                    source = source_datasets,
                                    return_similarity = True)
'''

from cellsaw.load.preprocess import annotate_genescores # TODO should we move this to bla

def rank_by_similarity(target = False,
                        source = False,
                        return_similarity = True):
    '''
    target: the ones we want to annotate
    source: the database
    return_similarity: returns (list, similarity_matrix) otherwise only the list
    '''

    # todo: minscore is set to 200 here,.,,  should be dont in the laoder
    source = Map(annotate_genescores, source)
    target = Map(annotate_genescores, target)



    ff = lambda a,b,c: cosine(a,b, numgenes=500)
    distances = matrixmap_odd(ff,target,source,repeats = 2)
    distances = np.mean(distances, axis =2)
    ranksim = np.argsort(-distances)
    ranklist = [[source[col] for col in row]for row in ranksim]


    def getname(ada):
        return ada.uns['tissue5id']

    ind = Map(getname,target)
    col = Map(getname,source)

    distances = pd.DataFrame( distances, index = ind, columns = col)


    if return_similarity:
        return ranklist, distances
    else:
        return ranklist


def plot_dendrogram(similarity_df):
    draw.dendro_degen(similarity_df.to_numpy(),similarity_df.columns ,similarity_df.index)





