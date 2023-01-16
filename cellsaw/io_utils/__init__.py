from lmz import Map,Zip,Filter,Grouper,Range,Transpose
from cellsaw.io_utils.loadadata import get41names, get100names, load100
from cellsaw.preprocess import annotate_genescore_single
from scipy.sparse import csr_matrix
import pandas as pd


def easyLoad100(name, path = None, remove_unlabeled = False, mingenes= 200,subsample = None,
                preprocessingmethod = 'natto', donormalize= True,
                plot=False, nattoargs=  {'mean': (0.015, 4), 'bins': (.25, 1)}):
    '''this would be the sane way to load data, but we dont use this anymore... '''
    adata = load100(name, path=path, remove_unlabeled=remove_unlabeled, subsample = subsample)
    gs = annotate_genescore_single(adata, mingenes=mingenes, selector=preprocessingmethod,
            donormalize= donormalize, nattoargs= nattoargs, plot=plot)
    return gs

import anndata
import glob
import cellsaw.io_utils.loadadata as ldata
from ubergauss import tools as t
import os
import scanpy as sc



def read(dir, suffix = '.gz',
        datasets=[],
         sampleseed = None,
        delimiter= '\t',
        sample_size = 0,
         min_genes = 200,
        remove_cells = {}):

    # targets = list of files to load
    if dir[-1] != '/':
        dir+='/'
    if not datasets:
        targets = glob.glob(f'{dir}*{suffix}')
    else:
        targets  = [f'{dir}{e}{suffix}' for e in datasets]


    # if the files are csv: convert
    if not suffix.endswith('h5'):
        t.xmap(lambda x: readcsv(x, delimiter),targets,n_jobs = 10)
        return read(dir,'.h5')


    # read an h5 file
    def openh5(fname):
        try:
            adata = anndata.read_h5ad(fname, backed = None)
            sc.pp.filter_cells(adata, min_genes=min_genes, inplace=True)
        except:
            print('basic h5 loading failed:', fname)
            return 0

        # reduce cell count
        for k,v in remove_cells.items():
            try:
                for delete_label in v:
                    adata = adata[adata.obs[k]!=delete_label]
            except:
                print('deletelabel failed in:', fname)

        if sample_size:
            try:
                sc.pp.subsample(adata, fraction=None, n_obs=sample_size,
                        random_state=sampleseed,copy=False)
            except Exception as e:
                print(e)
                print  (f"COULD NOT SUBSAMPLE {sample_size} items\
                        from {adata.uns['fname']} cells(labeled)= {adata.X.shape}")
                return adata

        # done
        print(".", end = '')
        return adata

    return Map(openh5, targets)






def readcsv(x,delimiter = '\t', saveas = ''):
    #adata = anndata.read_csv(x,delimiter=delimiter)
    things = pd.read_csv(x, sep='\t').T
    adata = anndata.AnnData(things)
    adata.X=csr_matrix(adata.X)
    path_nosuffix = str(x[:x.find('.')])
    print (path_nosuffix)

    try:
        tis__cellcnt = path_nosuffix[path_nosuffix.rindex('/')+1:].split('_')
        adata.uns['tissue'] = tis__cellcnt[0]
        adata.uns['tissue5'] = tis__cellcnt[0][:5]
        adata.uns['tissue5id'] = f'{tis__cellcnt[0][:5]}_{tis__cellcnt[-1]}'
    except:
        print("something went wrong finding tissue, tissue5 and/or tissue5id to write to 'uns'")


    adata.uns['fname'] = path_nosuffix
    if saveas:
        adata.write(saveas, compression='gzip')
    else:
        adata.write(str(path_nosuffix)+'.h5', compression='gzip')
    adata.file.close()
    #os.remove(filename)
    return 0





def annotate(adatas,annotate_function, **kwargs):
    f = annotate_function(**kwargs)
    return Map(f, adatas)


def annotate_from_barcode_csv(input_field = False, output_field = 'clusterid'):

    def annotate(adata):
        # add cluster id
        #fname = f"{path}/{adata.filename}.cluster.txt"
        fname = f"{adata.uns['fname']}.cluster.txt"
        lol = open(fname,'r').readlines()
        barcode_cid={}
        for line in lol:
            bc,cl =  line.strip().split()
            barcode_cid[bc]= int(cl)
        if not input_field or input_field == 'barcode':
            cells = adata.obs.index
        else:
            cells = adata.obs[input_field]
        adata.obs[output_field] = [barcode_cid.get(a,-1)  for a in cells]
        return adata

    return annotate

def annotate_celltypes( path = '', input_field = '', output_field = ''):
    plabels = ldata.loadpangalolabels(path)

    return lambda x: ldata.annotatetruecelltype(plabels, x, x.uns['fname'],
            f_out = output_field, f_in = input_field)


def save(adatas, format = 'h5'):
    assert format == 'h5', 'there is no generic writer in the libs currently'
    for adata in adatas:
        adata.write(adata.uns['fname']+'.h5', compression='gzip')




def nuread(dir,
        dataset,
        sampleseed = None,
        delimiter= '\t',
        sample_size = 0,
        min_genes = 200,
        remove_cells = {}):
    '''
    the other read version is annoying, we need a version that is just better :D..

    1. if there is no .h5:
        - load, annotate, save

    2. load .h5, filter, return
    '''
    expect = os.path.join(dir,'xmas',dataset+'.h5')
    if not os.path.exists(expect):
        csvpath = os.path.join(dir,dataset+'.counts.gz')
        readcsv(csvpath, delimiter,saveas = expect)
        adata = anndata.read_h5ad(expect, backed = None)

        adata = annotate_from_barcode_csv(input_field='barcode',output_field='clusterid')(adata)
        adata = annotate_celltypes(path = dir,input_field='clusterid',output_field='celltype')(adata)

        adata.write(expect, compression='gzip')

        adata.file.close()

    try:
        adata = anndata.read_h5ad(expect, backed = None)
        sc.pp.filter_cells(adata, min_genes=min_genes, inplace=True)
    except:
        print('basic h5 loading failed:', expect)
        return 0

    # reduce cell count
    for k,v in remove_cells.items():
        try:
            for delete_label in v:
                adata = adata[adata.obs[k]!=delete_label]
        except Exception as e:
            print('deletelabel failed in:', expect, e )

    if sample_size:
        try:
            sc.pp.subsample(adata, fraction=None, n_obs=sample_size,
                    random_state=sampleseed,copy=False)
        except Exception as e:
            print(e)
            print  (f"COULD NOT SUBSAMPLE {sample_size} items\
                    from {adata.uns['fname']} cells(labeled)= {adata.X.shape}")
            return adata

    if 'celltype' not in adata.obs:
        print(f'we should delete {expect}')
    return adata


