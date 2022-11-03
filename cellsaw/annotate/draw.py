
import matplotlib.pyplot as plt
from cellsaw.merge import  Merge
from cellsaw import draw
from ubergauss import tools as ut
import numpy as np
import seaborn as sns

def plot_annopair(source,target,source_label = '', target_label ='', pca= 20):
    '''
    annotate will transfer labels between andata objects -> plot these
    '''

    if id(source) == id(target):
        datasets = [target, target.copy()]
    else:
        datasets = [source, target]


    umaps = [2] if pca >2 else []
    merged = Merge(datasets, umaps=umaps, pca = pca, make_even = False, sortfield=-1)

    s,t = merged.projections[2] if pca > 2 else merged.projections[1]


    sns.set_theme(style = 'whitegrid')
    d = draw.tinyUmap(dim=(1,3), lim = [s,t])

    tlab = list(merged.data[1].obs[target_label])
    slab = list(merged.data[0].obs[source_label])
    allthelabels = tlab + slab
    all,sm = ut.labelsToIntList(allthelabels)


    labeld = {k:v[:8] for k,v in sm.getitem.items()}

    size = 10
    d.draw(t,sm.encode(tlab), title = f'T:{target_label}', labeldict = labeld,size= size)

    def plot_cellids(t):
        [plt.text(a, b, str(i),fontsize='xx-small') for i,(a,b) in enumerate(t.tolist())]

    # plot_cellids(t)
    d.draw(s,sm.encode(slab), title = f'S:{source_label}', labeldict = labeld, size=size)
    # plot_cellids(s)
    d.draw(np.vstack((t,s)),all, title = 'Both', labeldict = labeld,size= size)
    #draw.plt.legend()

    plt.legend(markerscale=2,ncol=4,bbox_to_anchor=(1, -.12),fontsize = 'small' )
    return merged
