from lmz import Map,Zip,Filter,Grouper,Range,Transpose, grouper
import numpy as np
from ubergauss import tools as ut
from sklearn.metrics.pairwise import cosine_similarity as cos
from sklearn import preprocessing
from sklearn.metrics import precision_score


def cosine(a,b, numgenes = 500, scores= 'scores'):
    scr1, scr2 = a.varm[scores], b.varm[scores]
    if numgenes:
        mask = scr1+scr2
        mask = ut.binarize(mask,numgenes).astype(np.bool)
        scr1 = scr1[mask]
        scr2 = scr2[mask]
    else:
        assert False
    return cos([scr1],[scr2]).item()



def jaccard(adata1,adata2 ,numgenes= False,scores='scores'):
    # intersect/union
    assert numgenes > 3
    asd = np.array([ ut.binarize(d.varm[scores],numgenes)  for d in [adata1,adata2]])
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








def mkshortnames(li):
    return [l[:5] for l in li]

def precission_at(df,k):
    return precision(df.to_numpy(),mkshortnames(df.columns),k)







def matrixmap(inst,inst2,meth, scorelabel, genecount):
    l = len(inst)
    l2 = len(inst2)
    res = np.zeros((l,l2))
    f = jaccard if meth == 'jaccard' else cosine
    for i in range(l):
        for j in range(i,l2):
            a,b = inst[i], inst2[j]
            r= f(a,b,numgenes=genecount, scores= scorelabel)
            if r==1.0: # it is 1 if inst and inst2 are the same -> genescore arrays are identical
                r = -1
            res[i,j] = r
            res[j,i] = r
    res[res < 0] = res.max()*1.05
    return res

def adata_to_score(instances,genecount, preprocessing, cosjacc, labels):
    shortlabels = mkshortnames(labels)
    matrix = matrixmap(instances, instances,cosjacc,preprocessing, genecount)
    return [precision(matrix,shortlabels,k) for k in [1,2,3]]
    # make the matrix
    # score the k@k
