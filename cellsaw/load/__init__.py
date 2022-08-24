
from cellsaw.load.loadadata import get41names, get100names, load100
from cellsaw.load.preprocess import annotate_genescores
from scipy.sparse import csr_matrix
import pandas as pd

def easyLoad100(name, path = None, remove_unlabeled = False, mingenes= 200,subsample = None,
                preprocessingmethod = 'natto', donormalize= True,
                plot=False,quiet=True, nattoargs=  {'mean': (0.015, 4), 'bins': (.25, 1)}):
    adata = load100(name, path=path, remove_unlabeled=remove_unlabeled, subsample = subsample)
    gs = annotate_genescores(adata, mingenes=mingenes, selector=preprocessingmethod, donormalize= donormalize, nattoargs= nattoargs, plot=plot, quiet=quiet)
    return gs

import anndata
import glob
import cellsaw.load.loadadata as ldata
from ubergauss import tools as t
import os


def read(dir, suffix = '.gz',datasets=[],delimiter= '\t'):
    '''
    # fun facts
    1. scanpys load function should be able to handle everything,
    but the delimiter is not passed on correctly.,..

    2. it seems not so easy to get rid of unpacked gz files
    '''

    # find the files...
    if dir[-1] != '/':
        dir+='/'
    if not datasets:
        targets = glob.glob(f'{dir}*{suffix}')[:10]
    else:
        targets  = [f'{dir}{e}{suffix}' for e in datasets]


    # scanpy reading has a bug so we use anndatareadcsv in the meantime..
    if not suffix.endswith('h5'):
        t.xmap(lambda x: readcsv(x, delimiter),targets)
        return read(dir,'.h5')

    return  [anndata.read_h5ad(x) for x in targets]


def readcsv(x,delimiter = '\t'):
    # things = pd.read_csv(x, sep='\t').T
    # adata = anndata.AnnData(things)
    # adata.X=csr_matrix(adata.X)
    adata = anndata.read_csv(x,delimiter=delimiter)
    filename = str(x[:x.find('.')])
    print (filename)
    adata.filename = filename
    adata.write(str(filename)+'.h5', compression='gzip')
    adata.file.close()
    os.remove(filename)









def annotate_truth(adatas, path):

    plabels = ldata.loadpangalolabels(path)
    def annotate(adata):
        # add cluster id
        fname = f"{path}/{adata.filename}.cluster.txt"
        lol = open(fname,'r').readlines()
        barcode_cid={}
        for line in lol:
            bc,cl =  line.strip().split()
            barcode_cid[bc]= int(cl)
        adata.obs['true'] = [barcode_cid.get(a,-1)  for a in adata.obs.index]

        # add real label
        # TODO is this ok with the filename?
        #
        ldata.annotatetruecelltype(plabels, adata, adata.filename)

        return adata

    return [annotate(x) for x in adatas]





def saveh5(data):
    for adata in data:
        adata.write(str(adata.filename)+'.h5', compression='gzip')






