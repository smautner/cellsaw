import lmz
from scalp.data import transform
import numpy as np
from scipy.sparse.csgraph import dijkstra
import ubergauss.tools as ut
from sklearn.semi_supervised import LabelSpreading



def diffuse_label_sklearn(adatas, ids_to_mask = [], base='pca40', new_label = 'sk_diffuse'):

    adatas_stacked = transform.stack(adatas)

    masked_y, sm =mask_y(adatas_stacked,'label',ids_to_mask)


    # now we have everything to run tue model..
    model = LabelSpreading()
    model.fit(np.asarray(adatas_stacked.X.todense()), masked_y)

    adatas = transform.attach_stack(adatas, np.array(sm.decode(model.transduction_)), new_label)
    return adatas


def mask_y(adatas_stacked, label, ids_to_mask):
    newy = list(adatas_stacked.obs[label])
    newy, sm  = ut.labelsToIntList(newy)
    newy = np.array(newy)

    # 2. we mask the batches we want to calculate labels for..
    batchnames = transform.unique_nosort(adatas_stacked.obs[f'batch'])
    for r in [batchnames[i] for i in ids_to_mask]:
        newy[(adatas_stacked.obs[f'batch'] == r).to_numpy()] = -1
    return newy, sm


def diffuse_label(adatas, distance_matrix, use_labels_from_datasets,
                  sigmafac = 1, label = f'label', new_label = f'diffuselabel'):
    '''
    adatas_stacked and lapgraph are the output in wrappers.dolucy

    i want to use the transduct_ thing in sklearn labelpro after fit.
    so i build a kernel that just returns my precomputed lap_graph and give it to fit.
    '''

    # first we build the kernel-similarity-matrix
    distance_matrix = dijkstra(distance_matrix, directed = False)
    sigma = np.mean([ np.min(row[row!=0])  for row in distance_matrix])*sigmafac
    similarity_matrix = np.exp(-distance_matrix/sigma)

    # we want to diffuse the labels, so we ..
    # 1. turn them into integers as required by sklearn
    # 2. we mask the batches we want to calculate labels for..
    adatas_stacked = transform.stack(adatas)
    maskds = [ i for i in lmz.Range(adatas) if i not in use_labels_from_datasets]
    masked_y, sm =mask_y(adatas_stacked, label, maskds)


    # now we have everything to run tue model..
    model = LabelSpreading()
    model.set_params(kernel = lambda x,y: similarity_matrix)
    model.fit(np.asarray(adatas_stacked.X.todense()), masked_y)

    # attach results to our adata object
    # adatas.obs[new_label] = sm.decode(model.transduction_)
    # breakpoint()
    adatas = transform.attach_stack(adatas, np.array(sm.decode(model.transduction_)), new_label)
    return adatas
