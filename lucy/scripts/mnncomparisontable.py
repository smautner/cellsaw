from lmz import Map,Zip,Filter,Grouper,Range,Transpose
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import ubergauss.tools as ut
import wrappers as nu
from lucy import load, adatas





def format_scores(scores, method = 'methodlabel',tests = 'label shilouette batchmix'.split(' ') ):
    '''
    scores is a 2d list of scores,  1 row per dataset
    '''
    return  [{"dataset":data_id , "test":tests[j], 'method':method, 'score': value}
                 for data_id, scores in enumerate(scores) for j,value in enumerate(scores)]

def cprunmnn(ds):
    data = [z.copy() for z in ds]
    data = nu.domnn(data)
    return data

def cprunlucy(ds):
    data = [z.copy() for z in ds]
    data = nu.dolucy(data)
    return data

if __name__ == "__main__":
    datasets = load.load_scib() + load.load_timeseries(path= '/home/ubuntu/repos/cellsaw/notebooks/')
    ssdata = [[adatas.subsample(i,500,31442)  for i in series[:3]]  for series in datasets]
    ssdata = Map(adatas.preprocess, ssdata)

    mnn_scores =Map(nu.scores, ut.xmap(cprunmnn,ssdata))
    result = format_scores(mnn_scores, 'mnn')
    # lucy_scores =Map(nu.scores, ut.xmap(cprunlucy,ssdata))
    # result  += format_scores(lucy_scores, 'lucy')

    print(result)

    import pandas as pd
    print(pd.DataFrame(result))
    print (result)




