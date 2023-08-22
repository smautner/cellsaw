from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import numpy as np
from anndata._core.merge import concat
import ubergauss.tools as ut

def stack(adatas):
    assert 'batch' in adatas[0].obs
    return concat(adatas)


def stack_single_attribute(adatas, attr = ""):
    if not attr:
        data = [a.X for a in adatas]
    else:
        data = [a.obsm[attr] for a in adatas]
    return ut.vstack(data)

def stack_single_obs(adatas, attr = ""):
    data = [a.obs[attr] for a in adatas]
    return ut.vstack(data)

def unique_nosort(items):
    uitems, index = np.unique(items, return_index=True)
    # index shows us the first appearance of an item,
    # thus sort so that we get the first appearing first
    return [items[i] for i in sorted(index)]


def split_by_adatas(adatas, stack):
    # TODO
    batch_ids = np.hstack([ a.obs['batch'] for a in adatas])
    return [ stack [batch_ids == batch] for batch in unique_nosort(batch_ids)]


def split_by_obs(adatas, obs=f'batch'):
    batch_ids = adatas.obs[obs]
    return [ adatas [batch_ids == batch] for batch in unique_nosort(batch_ids)]


def attach_stack(adatas, stack, label):
    '''
    if we generate data for all cells on the stacked-adatas,
    this function can split the data and assign it to the adatas
    '''
    stack_split = split_by_adatas(adatas,stack)
    for a,s in zip(adatas,stack_split):
        if len(stack.shape) == 1:
            a.obs[label]= s
        else:
            a.obsm[label] = s
    return adatas


def to_array(ad,base):
    return ad.X if not base else ad.obsm[base]


def to_arrays(adatas,base):
    return Map(to_array, adatas, base=base)
