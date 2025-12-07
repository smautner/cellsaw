import numpy as np
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


def make_adjacency(similarity, algo=0, connection_percentage=.9):
        neighbors = max(1, int(similarity.shape[0]*connection_percentage))
        assert neighbors <= similarity.shape[0], f'{neighbors=}{similarity.shape = } {connection_percentage=}'
        simm = neighborgraph_p_weird if algo == 1 else neighborgraph_p_real
        return simm(similarity, neighbors)




def merge_adjacency(*args):
    if len(args) == 1:
        return args[0]
    return np.logical_or(args[0],merge_adjacency(*args[1:]))

def make_stairs(num_ds, kList):
    return merge_adjacency(*[np.eye(num_ds,k=k)for k in kList]).astype(np.int32)

def test_mkts():
    print(f"{mkts(4,[-1,0,1])}")



def make_star(size = 5, center =2):
    ret = np.zeros((size,size))
    ret[center] = 1
    ret[:,center] = 1
    return ret

# from ubergauss import tools as ut
#  .,..
# def jaccard_distance(a,b, num_genes):
#     binarized_hvg = np.array([ ut.binarize(d,num_genes) for d in [a,b] ])
#     union = np.sum(np.any(binarized_hvg, axis=0))
#     intersect = np.sum(np.sum(binarized_hvg, axis=0) ==2)
#     return intersect/union
# def similarity(adatas, hvg_name = 'cell_ranger', num_genes = 2000):
#     assert hvg_name in adatas[0].var, 'not annotated..'
#     genescores = [a.var[hvg_name] for a in adatas]
#     # genescores = [a.uns[hvg_name] for a in adatas]
#     res = [[ jaccard_distance(a,b, num_genes = num_genes) for a in genescores] for b in genescores]
#     return np.array(res)


def similarity(adatas, hvg_name = 'cell_ranger', num_genes = 2000):
    assert hvg_name in adatas[0].var, 'not annotated..'
    gs = np.array([a.var[hvg_name] for a in adatas])
    # genescores = [a.uns[hvg_name] for a in adatas]
    return gs @ gs.T








from scipy.sparse.csgraph import connected_components
from ubergauss import hubness
def dynamic_sim(adatas, hvg = 'myvar'):
    matrix = similarity(adatas, hvg_name= hvg)
    fallback= np.ones_like(matrix) # Fallback fully connected
    if matrix.shape[0] <=3 :
        return fallback

    k = int(matrix.shape[0]*.5)
    matrix = hubness.justtransform(matrix,k = k)
    # we see the matrix as an adjacency matrix, find the minimum  threshold such that all nodes are still connected
    # its ok to just try all the tresholds
    thresholds = np.sort(np.unique(matrix.flatten()))
    for t in thresholds:
        adj = (matrix <= t).astype(int)
        # Check connectivity using BFS/DFS or scipy.sparse.csgraph.connected_components
        asd = connected_components(adj, directed=False)
        if asd[0] == 1:
            return matrix <=  t
    return fallback
