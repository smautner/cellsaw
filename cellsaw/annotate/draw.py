
import matplotlib.pyplot as plt
from cellsaw.merge import  Merge
from cellsaw.merge import draw as mergedraw
from ubergauss import tools as ut
import numpy as np

def plot(source,target,source_label = '', target_label ='', pca= 20):

    umaps = [2] if pca >2 else []
    if id(source) == id(target):
        merged = Merge([target, target.copy()], umaps=umaps, pca = pca)
    else:
        merged = Merge([source, target], umaps=umaps, pca = pca)

    s,t = merged.projections[2] if pca > 2 else merged.projections[1]



    d = mergedraw.tinyUmap(dim=(1,3), lim = [s,t])

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

    plt.legend(markerscale=4,ncol=3,bbox_to_anchor=(1, -.12) )
    return merged
