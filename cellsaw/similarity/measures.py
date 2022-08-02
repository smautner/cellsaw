from lmz import Map,Zip,Filter,Grouper,Range,Transpose, grouper
import numpy as np
from ubergauss import tools as ut
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn import preprocessing
from sklearn.metrics import precision_score


def cosine(a,b, numgenes = 500):
    scr1, scr2 = a.varm['scores'], b.varm['scores']
    if numgenes:
        mask = scr1+scr2
        mask = ut.binarize(mask,numgenes).astype(np.bool)
        scr1 = scr1[mask]
        scr2 = scr2[mask]
    else:
        assert False
    return cos([scr1],[scr2]).item()



def jaccard(a,b ,ngenes= False):
    # intersect/union
    assert n_genes > 3
    asd = np.array([ ut.binarize(d.varm['scores'],ngenes)  for d in [a,b]])
    union = np.sum(np.any(asd, axis=0))
    intersect = np.sum(np.sum(asd, axis=0) ==2)
    return intersect/union



def precision(matrix,shortlabels, k):
    truth = preprocessing.LabelEncoder().fit_transform(shortlabels)
    srt = np.argsort(matrix, axis=1)
    #print(f'{truth=} {srt=}')
    pred = truth[ [ srt[i,-j]  for j in range(1,k+1) for i in Range(truth)] ]
    true = np.hstack([truth for i in range(k)])
    return  precision_score(true, pred, average='micro')
