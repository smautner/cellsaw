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
from scipy.sparse import csr_matrix, lil_matrix
from scipy.stats import rankdata
from collections import defaultdict
from sklearn import preprocessing as skprep
from ubergauss import hubness as uhub

# import matplotlib
# matplotlib.use('module://matplotlib-backend-sixel')
# import matplotlib.pyplot as plt

def steal_neighbors(mlsa, mcopyfrom):
    '''
    from lsa block, copy the neighbors of the partner instead of only the partner
    '''
    res = np.zeros(mlsa.shape)
    mlsa = csr_matrix(mlsa)
    mcopyfrom = csr_matrix(mcopyfrom)
    for i, row in enumerate(mlsa):
        ij_ctr = {}
        for j in row.indices:
            newvalues = mcopyfrom[j]
            # average distance to neighbors :)
            res[i] += newvalues
            avg = np.mean(newvalues.data)
            if not np.isnan(avg):
                ij_ctr[(i,j)]  = avg
        for (i,j), n in ij_ctr.items(): res[i,j] = n


    # plt.matshow(mlsa.todense())
    # plt.show()
    # plt.matshow(mcopyfrom.todense())
    # plt.show()
    # # dim= (50,50)
    # # so.heatmap(mlsa.todense(),dim=dim)
    # # so.heatmap(mcopyfrom.todense(),dim=dim)
    # # so.heatmap(res,dim=dim)
    # plt.matshow(res)
    # plt.show()

    # assert not np.isnan(res).any()
    return sparse.lil_matrix(res)



def iterated_linear_sum_assignment(distances, repeats):
    def linear_sum_assignment_iteration(distances):
        r, c = linear_sum_assignment(distances)
        d = distances[r,c].copy()
        distances [r,c] = np.inf
        return r,c,d
    cc, rr, dist  = Transpose([ linear_sum_assignment_iteration(distances) for _ in range(repeats)  ])
    return np.hstack(cc) , np.hstack(rr), np.hstack(dist)

def test_cdf_remove():
    ar = np.arange(10, dtype= float)
    print(cdf_remove(ar,ar[::-1]))

from scipy.stats import norm

def cdf_remove(data, bias):
    loc, scale = norm.fit(data)
    probs  = 1 - norm.cdf(data, loc, scale)
    return  np.random.random(len(data)) > (probs+bias)



def lin_asi_thresh(ij_euclidian_distances,inter_neigh=1, outlier_threshold=.9,
                   outlier_probabilistic_removal=False, prune = 0):

    # get the (repeated) assignment
    i_ids,j_ids, ij_lsa_distances = iterated_linear_sum_assignment(ij_euclidian_distances,inter_neigh)
    # i_ids,j_ids, ij_lsa_distances = repeated_subsample_linear_sum_assignment(ij_euclidian_distances,inter_neigh,100)

    # remote outliers
    sorted_ij_assignment_distances  = np.sort(ij_lsa_distances)

    if outlier_probabilistic_removal and outlier_threshold > 0:
        ij_lsa_distances[cdf_remove(ij_lsa_distances,outlier_threshold/2)] = 0
    elif  0< outlier_threshold < 1:
        lsa_outlier_thresh = sorted_ij_assignment_distances[int(len(ij_lsa_distances)*outlier_threshold)]
        outlier_ids = ij_lsa_distances >  lsa_outlier_thresh
        ij_lsa_distances[outlier_ids] = 0

    mask = ij_lsa_distances != 0
    return i_ids[mask], j_ids[mask], ij_lsa_distances[mask]
    # return i_ids, j_ids, ij_lsa_distances

def lsa_sample(distance_matrix, num_instances):
    # get random indices
    row_indices = np.random.choice(distance_matrix.shape[0], num_instances, replace = False)
    col_indices = np.random.choice(distance_matrix.shape[1], num_instances, replace = False)

    # get matches and distances
    subdistance_matrix = distance_matrix[row_indices][:,col_indices]
    r, c = linear_sum_assignment(subdistance_matrix)
    d =    subdistance_matrix[r,c].copy()

    # since subdistances is basically reindexing, we need to undo that:
    r = ut.spacemap(row_indices).decode(r)
    c = ut.spacemap(col_indices).decode(c)
    return r,c,d

def repeated_subsample_linear_sum_assignment(distances, repeats, num_instances):
    rr, cc, dist  = Transpose([ lsa_sample(distances,num_instances) for _ in range(repeats)  ])
    return np.hstack(rr) , np.hstack(cc), np.hstack(dist)




def fast_neighborgraph(distancemat, neighbors):
    part = np.argpartition(distancemat, neighbors, axis = 1)[:,:neighbors]
    neighborsgraph = np.zeros_like(distancemat)
    np.put_along_axis(neighborsgraph, part, np.take(distancemat,part), axis = 1)
    return neighborsgraph


def pac_neighborgraph_far(D, k):
    '''
    pick mid and far pairs from farther away
    '''
    n, s = D.shape[0], D.shape[0] // 6
    res = np.zeros_like(D, dtype=np.int8)
    idx = np.argsort(D, axis=1)
    row = np.arange(n)[:, None]
    # 1. Near: Absolute closest
    res[row, idx[:, 1:k+1]] = 1
    # 2. Mid: Sample from 2nd sextile [N/6 : 2N/6]
    m_pool = idx[:, s*3 : 4*s]
    m_samp = m_pool[row, np.random.randint(0, s, (n, k * 2))]
    res[row, m_samp] = 3
    # 3. Far: Sample from 3rd sextile [2N/6 : 3N/6]
    f_pool = idx[:, 2*s : 3*s]
    f_samp = f_pool[row, np.random.randint(0, s, (n, k // 2))]
    res[row, f_samp] = 2
    return res

def pac_neighborgraph(D, k):
    n, s = D.shape[0], D.shape[0] // 6
    res = np.zeros_like(D, dtype=np.int8)
    idx = np.argsort(D, axis=1)
    row = np.arange(n)[:, None]
    # 1. Near: Absolute closest
    res[row, idx[:, 1:k+1]] = 1
    # 2. Mid: Sample from 2nd sextile [N/6 : 2N/6]
    m_pool = idx[:, s : 2*s]
    m_samp = m_pool[row, np.random.randint(0, s, (n, k // 2))]
    res[row, m_samp] = 2
    # 3. Far: Sample from 3rd sextile [2N/6 : 3N/6]
    f_pool = idx[:, 2*s : 3*s]
    f_samp = f_pool[row, np.random.randint(0, s, (n, k * 2))]
    res[row, f_samp] = 3
    return res



def mutualNN(mtx):
    mask = mtx != 0
    mask &= mask.T
    return mtx * mask

from collections import Counter
def spanning_tree_neighborgraph(x, neighbors,add_tree=True, neighbors_mutual = True):
    '''
    the plan:
    '''

    model = NearestNeighbors(n_neighbors=neighbors*2).fit(x)
    distances, indices = model.kneighbors(x)
    counts = np.bincount(indices)
    counts_mat = counts[indices]
    cnt_srt = np.argsort(counts_mat, axis = 1)
    indices_new = np.take_along_axis(indices, cnt_srt, axis =1)[:,:neighbors]
    neighborsgraph = np.zeros_like((x.shape[0],x.shape[0]))
    np.put_along_axis(neighborsgraph,indices_new,1,axis=1)

    # distancemat = metrics.euclidean_distances(x)


    # min spanning tree
    # tree = minimum_spanning_tree(distancemat) if add_tree else np.zeros_like(distancemat)
    # tree= ut.zehidense(tree)
    # faster version of neighborsgraph
    # neighborsgraph = fast_neighborgraph(distancemat, neighbors)
    #anti_neighborsgraph = -fast_neighborgraph(-distancemat, neighbors)+100
    # make mutual
    # if neighbors_mutual:
    # neighborsgraph = mutualNN(neighborsgraph)
    # combine and return
    #combinedgraph = np.stack((neighborsgraph,neighborsgraph.T,tree,tree.T, anti_neighborsgraph,anti_neighborsgraph.T), axis =2)
    combinedgraph = np.stack((neighborsgraph,neighborsgraph.T ), axis =2)
    combinedgraph = combinedgraph.max(axis=2)
    np.fill_diagonal(combinedgraph,0)
    check_symmetric(combinedgraph,raise_exception=True) # graph_umap needs a symmetic matrix
    return combinedgraph

def symmetric_spanning_tree_neighborgraph(distancemat, neighbors,add_tree=True, neighbors_mutual = True):

    # distancemat = metrics.euclidean_distances(x)
    # distancemat = skprep.normalize(distancemat, axis =0) # DOTO this should get a flag

    #return fast_neighborgraph(distancemat, neighbors)
    # min spanning tree
    tree = minimum_spanning_tree(distancemat) if add_tree else np.zeros_like(distancemat)
    tree= ut.zehidense(tree)

    # faster version of neighborsgraph
    neighborsgraph = fast_neighborgraph(distancemat, neighbors)
    #anti_neighborsgraph = -fast_neighborgraph(-distancemat, neighbors)+100

    # make mutual
    if neighbors_mutual:
        neighborsgraph = mutualNN(neighborsgraph)

    # combine and return
    #combinedgraph = np.stack((neighborsgraph,neighborsgraph.T,tree,tree.T, anti_neighborsgraph,anti_neighborsgraph.T), axis =2)

    combinedgraph = np.stack((neighborsgraph,neighborsgraph.T,tree,tree.T ), axis =2)
    combinedgraph = combinedgraph.max(axis=2)
    np.fill_diagonal(combinedgraph,0)
    check_symmetric(combinedgraph,raise_exception=True)


    return combinedgraph



def avgdist(a,numneigh = 2):
    #  return mean distance to the first numneigh neighbors for each item in a
    nbrs = NearestNeighbors(n_neighbors=1+numneigh).fit(a)
    distances, indices = nbrs.kneighbors(a)
    return np.mean(distances[:,1:], axis = 1)


# def average_knn_distance(I,J,i_ids,j_ids,numneigh):
#     d1  = avgdist(I,numneigh)[i_ids]
#     d2  = avgdist(J,numneigh)[j_ids]
#     stack = np.vstack((d1,d2))
#     return np.mean(stack, axis=0).T


def linear_assignment_integrate(Xlist, base = 'pca40',
                                neighbors_total = 20,
                                horizonCutoff = 0,
                                neighbors_intra_fraction = .5,
                                intra_neigh=15,
                                inter_neigh = 1,
                                scaling_num_neighbors = 2,
                                distance_metric = 'euclidean',
                                outlier_threshold = .8,
                                dataset_adjacency = False,
                                intra_neighbors_mutual = True,
                                copy_lsa_neighbors = True,
                                outlier_probabilistic_removal = True,
                                add_tree = True,
                                epsilon = 1e-4 ):
    '''
    this is the original version that had way too much going on
    '''
    if 'anndata' in str(type(Xlist[0])):
        Xlist = to_arrays(Xlist, base)



    if len(Xlist) ==1:
                r = symmetric_spanning_tree_neighborgraph(Xlist[0],
                # r = spanning_tree_neighborgraph(Xlist[0],
                                                          int(neighbors_total),
                                                          add_tree = add_tree,
                                                          neighbors_mutual=intra_neighbors_mutual)
                r = sparse.lil_matrix(r)
                return r

    intra_neigh = int(max(1,np.ceil(neighbors_total*neighbors_intra_fraction)))
    inter_neigh_total = neighbors_total - intra_neigh #max(1,np.rint(neighbors_total*(1-neighbors_intra_fraction)))
    inter_neigh_desired = inter_neigh_total / (len(Xlist)-1) # this is how many we want
    inter_neigh = int(np.ceil(inter_neigh_desired)) #int(max(1,np.rint(inter_neigh_total / len(Xlist)))) # this is how many we sample


    def adjacent(i,j):
        if isinstance( dataset_adjacency, np.ndarray):
            return  dataset_adjacency[i][j] == 1
        else:
            return True

    def make_distance_matrix(ij):
            i,j = ij
            if i == j:
                # use the maximum between neighborgraph and min spanning tree to make sure all is connected
                distances = metrics.pairwise_distances(Xlist[i],metric = distance_metric)

                r = symmetric_spanning_tree_neighborgraph(distances, intra_neigh,
                                                          add_tree = add_tree,
                                                          neighbors_mutual=intra_neighbors_mutual)
                r = sparse.lil_matrix(r)

                # cutoffs =  [heapq.nlargest(horizonCutoff,values) for values in r.data] if horizonCutoff else 0
                cutoffs = np.partition(distances, horizonCutoff, axis=1)[:, horizonCutoff] if horizonCutoff else 0
                return r, cutoffs


            elif not adjacent(i,j):
                # just fill with zeros :)
                return  sparse.lil_matrix((Xlist[i].shape[0],Xlist[j].shape[0]), dtype=np.float32) ,0
            else:
                '''
                based = hungdists/= .25
                dm = avg 1nn dist in both
                based*=dm
                '''
                # hungarian
                # ij_euclidian_distances= metrics.euclidean_distances(Xlist[i],Xlist[j])
                ij_euclidian_distances= metrics.pairwise_distances(Xlist[i],Xlist[j],
                                                                   metric=distance_metric)



                res = sparse.lil_matrix(ij_euclidian_distances.shape,dtype=np.float32)
                if inter_neigh==0:
                    return res

                i_ids,j_ids, ij_lsa_distances = lin_asi_thresh(ij_euclidian_distances,
                                                               inter_neigh,outlier_threshold,
                                                               outlier_probabilistic_removal)


                res[i_ids,j_ids] = ij_lsa_distances



                for i,row in enumerate(res):
                    neighbors_have = np.sum(row > 0)
                    if neighbors_have > inter_neigh_desired:
                        assert neighbors_have > inter_neigh_desired > (neighbors_have -1), 'something is wrong with the neighbrs'
                        # we will just remove one, so:
                        if np.random.rand() < ( neighbors_have - inter_neigh_desired): # ok one needs to go
                            targets = np.array(row.rows[0])
                            z = np.random.choice(targets,1)[0]
                            res[i,z] = 0



                if epsilon > 0:
                    res[res > 0] = epsilon

                # zero is the horizon cutoff. which we dont need here
                return res,0

    n_datas = len(Xlist)
    tasks =  [(i,j) for i in range(n_datas) for j in range(i,n_datas)]
    parts = ut.xxmap( make_distance_matrix, tasks)
    # parts = Map( make_distance_matrix, tasks)
    # select p[0] to rm horizon cutoff values
    getpart = dict(zip(tasks,[p[0] for p in parts]))
    getcuts = dict(zip(tasks,[p[1] for p in parts]))


    def steal_neigh(ij):
        i,j = ij
        return i,j,steal_neighbors(getpart[(i,j)], getpart[(j,j)])

    if copy_lsa_neighbors:
        tasks =  [(i,j) for i in range(n_datas) for j in range(i+1,n_datas)]
        parts = ut.xxmap( steal_neigh, tasks)
        for i,j,block in parts:
            getpart[(i,j)]  = block


    # then we built  rows:
    rows = []
    for i in range(n_datas):
        row = []
        for j in range(n_datas):
            if i <= j:
                distance_matrix = getpart[(i,j)]
            else:
                distance_matrix = rows[j][i].T.copy()
            row.append(distance_matrix)

        rows.append(row)


    if horizonCutoff > 0:
        for i in range(n_datas):
            # for c in row:
            #     plt.matshow(c.todense())
            #     plt.roworbar()
            #     plt.show()
            #plt.matshow(np.hstack([c.todense() for c in row]))
            rows[i] = list(removeInstancesFartherThanHorizon(getcuts[(i,i)],rows[i],i,horizonCutoff))

        # now we make it symmetric
        for i in range(n_datas-1):
            for j in range(i+1,n_datas):
                rows[i][j]= rows[i][j].maximum(rows[j][i])
                rows[j][i]= rows[i][j].T
    rowss = [sparse.hstack(row) for row in rows]
    distance_matrix = sparse.vstack(rowss)

    # check_symmetric(distance_matrix,raise_exception=True)
    # so.heatmap(distance_matrix.todense(), dim = (100,100))

    return  distance_matrix

from matplotlib import pyplot as plt
import heapq
def removeInstancesFartherThanHorizon(cutoffs, targets :list,rId:int, nthNeighbor:int):
    '''
    remove instances that are farther than the nth neighbor in the reference in all matrixes contained in targets
    '''
    # cutoffs = np.partition(reference,nthNeighbor,axis=1)[:,nthNeighbor]
    # cutoffs =  [heapq.nlargest(nthNeighbor,values)[-1] for values in reference.data]

    #plt.hist(cutoffs);plt.show()
    #print(f"{cutoffs[0]=} {targets[0][0].data=} {targets[1][0].data=}")
    for j,target in enumerate(targets):
        # values larger than cutoffs (per row) are set to zero
        if j!=rId:
            for i,cut in enumerate(cutoffs):
                badind = [ind for ind,val in zip(target.rows[i], target.data[i]) if val > cut]
                target[i,badind] = 0
        yield target

def testHorizonCutOff():
    random = np.random.random((5,5))
    print(random)
    cutoffs = np.partition(random,3 ,axis=1)[:,3]
    print(cutoffs)
    random= lil_matrix(random)
    for i,cut in enumerate(cutoffs):
        badind = [ind for ind,val in zip(random.rows[i], random.data[i]) if val > cut]
        random[i,badind] = 0
    breakpoint()

# testHorizonCutOff()

def negstuff(Xlist,
            base = 'pca',
            neighbors_total = 20,
             neighbors_intra_fraction = .5, **kwargs):

    if 'anndata' in str(type(Xlist[0])):
        Xlist = to_arrays(Xlist, base)

    # intra_neigh = int(max(1,np.ceil(neighbors_total*neighbors_intra_fraction)))
    neg_used = neighbors_total
    neg_samples= int(Xlist[0].shape[0]/2)
    assert neg_used <= neg_samples, f"{neg_used=}  {neg_samples=}"
    def make_distance_matrix(ij):
        i,j = ij
        if i == j:
            distancemat = metrics.euclidean_distances(Xlist[i])
            part = np.argpartition(-distancemat, neg_samples, axis = 1)[:,:neg_samples]
            # part = np.argsort(-distancemat, axis = 1)[:,neg_samples//2:neg_samples]
            np.random.shuffle(part.T)
            part  =part[:,:neg_used]
            neighborsgraph = np.zeros_like(distancemat)
            np.put_along_axis(neighborsgraph, part, np.take(distancemat,part), axis = 1)
            return sparse.lil_matrix(neighborsgraph)
        return  sparse.lil_matrix((Xlist[i].shape[0],Xlist[j].shape[0]), dtype=np.float32)

    n_datas = len(Xlist)
    tasks =  [(i,j) for i in range(n_datas) for j in range(i,n_datas)]
    #parts = ut.xxmap( make_distance_matrix, tasks)
    parts = Map( make_distance_matrix, tasks)
    getpart = dict(zip(tasks,parts))

    # then we built a row:
    row = []
    for i in range(n_datas):
        col = []
        for j in range(n_datas):
            if i <= j:
                distance_matrix = getpart[(i,j)]
            else:
                distance_matrix = row[j][i].T
            col.append(distance_matrix)
        row.append(col)
    rows = [sparse.hstack(col) for col in row]
    distance_matrix = sparse.vstack(rows)
    return  csr_matrix(distance_matrix)















# def mkblock(matrix, i,j):
#     '''
#     we return a new array, the size of matrix
#     row[i] of the new matrix, has the data of row[j] of the old matrix
#     matrix and res should be lil_matrix
#     '''
#     # res = np.zeros((matrix.shape[0],matrix.shape[0]),dtype=np.float32)
#     # print(f"{ res.shape=}")
#     # print(f"{ matrix.shape=}")
#     # print(f"{ i.shape=}")
#     # print(f"{ j.shape=}")
#     res = sparse.lil_matrix(matrix.shape)
#     # matrix = matrix.T
#     res[i] = matrix[j]
#     return res


def mkblock(matrix, i, j):
    '''
    reorders the rows just like the commented out version above, is uglier, but the speedup is worth it :)
    '''
    if len(i) == 0:
        return sparse.lil_matrix(matrix.shape)
    submat = matrix.tocsr()[j, :].tocoo()
    i_array = np.array(i)
    mapped_rows = i_array[submat.row]
    return sparse.coo_matrix((submat.data, (mapped_rows, submat.col)), shape=matrix.shape).tolil()




from scalp import transform
import sklearn


def stack_blocks(n_datas, getpart):
    '''
    getpart giges us the blocks, and we need to stack them into a matrix
    '''
    rows = []
    for i in range(n_datas):
        row = []
        for j in range(n_datas):
            # if i <= j:
            if (i,j) in getpart:
                distance_matrix = getpart[(i,j)]
            else:
                distance_matrix = rows[j][i].T.copy()
            row.append(distance_matrix)
        rows.append(row)

    return sparse.bmat(rows, format = 'csr')
    # rowss = [sparse.hstack(row) for row in rows]
    # return sparse.vstack(rowss)



integrate_params = '''
hub1_k 3 20 1
hub2_k 3 20 1
hub1_algo 1 5 1
hub2_algo 1 5 1
outlier_threshold .65 .9
k 7 17 1
'''

# metric ['cosine']

def find_duplicate_rows(mat):

    mat = csr_matrix(mat)
    di = {}
    for i, row in enumerate(mat):
        h = hash(tuple(row.indices))
        if h in di:
            print(i, di[h])
        di[h] = i


def integrate(adata,*, base = 'pca40',
              k=12,
              metric = 'cosine',
              dataset_adjacency=False,
              hub1_k = 12,
              hub2_k = 12,
              hub1_algo = 2,
              hub2_algo = 2,
              pac= False, # if true, we will add anti-neighbors to the mix, which is a bit like pacmap
               smartcut = False, # experimenting with looking up if neighbors are close in the projection
              outlier_threshold= .75):

    # make sure the input is in the right format
    # there are 3 options: adata, list of adata and list of np.array
    # case 1:
    if 'anndata' in str(type(adata)):
        adata = transform.split_by_obs(adata)

    # ok now we have a list... it could be in the wrong format
    if 'anndata' in str(type(adata[0])):
        adata = to_arrays(adata, base)

    Xlist =adata

    assert len(Xlist) > 1, 'there should be at least 2 datasets..'
    # if len(Xlist) ==1: assert False, 'why do you only provide 1 dataset?'

    def adjacent(i,j):
        if isinstance( dataset_adjacency, np.ndarray):
            return  dataset_adjacency[i][j] == 1
        else:
            return True

    def make_distance_matrix(ij):
            i,j = ij
            if not adjacent(i,j):
                return [],[]# sparse.lil_matrix((Xlist[i].shape[0],Xlist[j].shape[0]), dtype=np.float32)

            distances= metrics.pairwise_distances(Xlist[i],Xlist[j], metric=metric)

            if i == j:

                assert distances.shape[0] > 50, 'very few cells in dataset, you messed up!'
                # distances = uhub.justtransform(distances, hub1_k, hub1_algo)
                distances = uhub.transform_experiments(distances, hub1_k, hub1_algo)

                f = fast_neighborgraph if not pac else pac_neighborgraph
                distances = f(distances, k)


                np.fill_diagonal(distances,1)
                distances = sparse.lil_matrix(distances)
                # if find_duplicate_rows(distances): breakpoint()# DEBUG, DELETE ME
                return distances


            # this is a case for k-start :) from ug.hubness
            # distances = hubness(distances, hub2_k, hub2_algo)
            distances = uhub.transform_experiments(distances, hub2_k, hub2_algo, startfrom = 0) # changing k-start since i != j

            if not smartcut:
                i_ids,j_ids, ij_lsa_distances = lin_asi_thresh(distances, 1,outlier_threshold, False)
            else:
                i_ids,j_ids, ij_lsa_distances = iterated_linear_sum_assignment(distances,1)
            return i_ids, j_ids



    n_datas = len(Xlist)
    tasks =  [(i,j) for i in range(n_datas) for j in range(i,n_datas)]
    parts = Map( make_distance_matrix, tasks)
    getpart = dict(zip(tasks,parts))


    if smartcut:
        neighDepLinAsiCut(getpart, outlier_threshold,k)


    #
    blockdict = {(i,i): getpart[(i,i)] for i in range(n_datas)}
    tasks =  [(i,j) for i in range(n_datas) for j in range(i+1,n_datas)]
    for i,j in tasks:
        blockdict[(i,j)]  = mkblock( getpart[(j,j)] , *getpart[(i,j)])

    for i,j in tasks:
        # ok so we fill the mirror too... breaking symmetry
        # i,i is the reference now
        imatch, jmatch  = getpart[(i,j)]
        blockdict[(j,i)]  = mkblock( getpart[(i,i)] , jmatch, imatch )

    # if pac: for i in range(n_datas): blockdict[(i,i)][blockdict[(i,i)]==2] = 0
    # MAKE THE MATRIX
    # check_symmetric(distance_matrix,raise_exception=True)
    # so.heatmap(distance_matrix.todense(), dim = (100,100))
    stack =  csr_matrix( stack_blocks(n_datas, blockdict))
    for i in range(stack.shape[0]): stack[i,i] = 0
    return stack

def neighDepLinAsiCut(blockdict, thresh, k):
    # minhits = (1-thresh) * k

    for (i,j) in list(blockdict.keys()):
        if i!=j:
            a = blockdict[(i,i)]
            b = blockdict[(j,j)]
            aaa = blockdict[(i,j)]
            lookup = dict(zip(*aaa))
            # lookup2 = dict(zip(*aaa[::-1])) back and forth should always be the same
            # mask = np.ones(aaa[0].shape)
            hit_score = []
            for x,(k,v) in enumerate(zip(*aaa)): # all the connections
                hits = sum ([lookup.get(e,-1) in b.rows[v]  for e in a.rows[k] ]) -1 # is my neighbor in the partner neighbor list?
                hit_score.append(hits)
                # if hits < (minhits*2): mask[x] = 0

            cut  = np.sort(hit_score)[int(len(hit_score) * thresh)]
            mask = np.array(hit_score) >= cut
            blockdict[(i,j)] = (aaa[0][mask==1], aaa[1][mask==1])


def test_integrate():
    # make 2 random 10x10 matrices
    # and run integrate on them
    # import scalp.data as data
    # a = data.scib(scalp.test_config.scib_datapath, maxdatasets=3, maxcells = 100, datasets = ["Immune_ALL_hum_mou"]).__next__()

    a= (np.random.random((100,200)),np.random.random((100,200)))
    integrate(a, smartcut=True)


# def MP(distance_matrix, k=6):
#     n = distance_matrix.shape[0]
#     k = int(np.sqrt(n))
#     nbrs = NearestNeighbors(n_neighbors=k).fit(distance_matrix)
#     distances, indices = nbrs.kneighbors(distance_matrix)  # Sorted by distance

#     # Initialize rank matrix (high rank = less proximity)
#     ranks = np.zeros((n, n))
#     for i in range(n):
#         ranks[i, indices[i]] = np.arange(1, k + 1)  # Rank 1 is nearest

#     # Step 2: Convert ranks to empirical probabilities (P_i(x_j))
#     P = ranks / n  # P_i(x_j) = rank_i(x_j) / n

#     # Step 3: Compute MP as P_i(x_j) * P_j(x_i)
#     MP = P * P.T
#     return MP

def hubness(d,k,algo):
    return uhub.transform_experiments(d,k,algo)


