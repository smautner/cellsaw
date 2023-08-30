from scalp.data.transform import to_arrays


def align(adatas, base ='pca40'):
    for i in range(len(adatas)-1):
        hung, _ = hungarian(adatas[i], adatas[i+1], base= base)
        adatas[i+1]= adatas[i+1][hung[1]]
    return adatas


def hungarian(adata, adata2, base):
        X = to_arrays([adata, adata2], base=base)
        hung, dist = hung_nparray(*X)
        return hung, dist[hung]

from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
def hung_nparray(X1, X2, debug = False,metric='euclidean'):
    distances = pairwise_distances(X1,X2, metric=metric)
    row_ind, col_ind = linear_sum_assignment(distances)
    if debug:
        x = distances[row_ind, col_ind]
        num_bins = 100
        print("hungarian: debug hist")
        plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
        plt.show()
    return (row_ind, col_ind), distances
