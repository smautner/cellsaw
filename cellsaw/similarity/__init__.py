from lmz import Map,Zip,Filter,Grouper,Range,Transpose

import cellsaw.draw
from cellsaw.similarity.measures import cosine, jaccard, precision, mkshortnames
import numpy as np
from ubergauss import tools as ut
import pandas as pd
import logging
import time

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

    #for r,i,j in ut.xmap(func,tmptasks):
    for r,i,j in map(func,tmptasks):
        for x,val in enumerate(r):
            res[i,j,x] = val

    return res



'''
ranked_datasets_list, similarity_df = rank_by_similarity(target = target_datasets,
                                    source = source_datasets,
                                    return_similarity = True)
'''

from cellsaw.preprocess import annotate_genescore_single




from collections import defaultdict

def rank_by_sim_splitbyname(datasets, names, **kwargs):
    sn = mkshortnames(names)
    d = defaultdict(list)
    for name,item in zip(sn,datasets):
        d[name].append(item)


    if kwargs.get('return_similarity', True):
        print ('well just turn return_similarity off :)')
        kwargs['return_similarity'] = False

    flattened_results = [ line for data in d.values()
             for line in rank_by_similarity(target=data,source=data,**kwargs)]

    return flattened_results







def rank_by_similarity(target = False,
                        source = False,
                        numgenes = 2500,
                        similarity = 'jaccard',
                        dataset_name_field ='tissue5id',
                        return_similarity = True, method = 'seurat_v3'):
    '''
    target: the ones we want to annotate
    source: the database
    return_similarity: returns (list, similarity_matrix) otherwise only the list
    '''
    starttime = time.time()
    #source = ut.xmap(lambda x: annotate_genescores(x,selector=method), source)
    source = [annotate_genescore_single(s,selector = method) for s in source]
    target = [annotate_genescore_single(t,selector = method) for t in target]
    # print('got target')
    logging.info(f'obtained genescores {time.time()-starttime}')
    # source = Map(annotate_genescores, source)
    # target = Map(annotate_genescores, target)


    #breakpoint()

    if similarity  == 'cosine':
        ff = lambda a,b,c: cosine(a,b, numgenes=numgenes, scores = method)
    elif similarity == 'jaccard':
        ff = lambda a,b,c: jaccard(a,b, numgenes=numgenes, scores =method)
    else:
        raise('similarity should be either cosine or jaccard')

    distances = matrixmap_odd(ff,target,source,repeats = 1)
    distances = np.mean(distances, axis =2)
    ranksim = np.argsort(-distances)
    ranklist = [[source[col] for col in row]for row in ranksim]

    logging.info(f'did matrixmap {time.time()-starttime}')

    def getname(ada):
        return ada.uns.get(dataset_name_field,'no name')

    ind = Map(getname,target)
    col = Map(getname,source)

    distances = pd.DataFrame( distances, index = ind, columns = col)


    logging.info(f'nearly done...{time.time()-starttime}')
    if return_similarity:
        return ranklist, distances
    else:
        return ranklist






def plot_dendrogram(similarity_df):
    cellsaw.draw.dendro_degen(similarity_df.to_numpy(), similarity_df.columns, similarity_df.index)





