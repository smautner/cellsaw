


from cellsaw.load.loadadata import get41names, get100names, load100
from cellsaw.load.preprocess import annotate_genescores

def easyLoad100(name, path = None, remove_unlabeled = False, mingenes= 200,subsample = None,
                preprocessingmethod = 'natto', donormalize= True,
                plot=False,quiet=True, nattoargs=  {'mean': (0.015, 4), 'bins': (.25, 1)}):
    adata = load100(name, path=path, remove_unlabeled=remove_unlabeled, subsample = subsample)
    gs = annotate_genescores(adata, mingenes=mingenes, selector=preprocessingmethod, donormalize= donormalize, nattoargs= nattoargs, plot=plot, quiet=quiet)
    return gs