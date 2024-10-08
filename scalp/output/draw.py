from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import pandas as pd
from ubergauss import tools
import numpy as np
from scipy.optimize import linear_sum_assignment as lsa

from scalp.data.align import align

# col = plt.cm.get_cmap('tab20').colors
col = matplotlib.colormaps['tab20'].colors
col = col + col + col + ((0, 0, 0),)


# col = plt.cm.get_cmap('prism')
# col = [col(i) for i in np.linspace(0, 1, 35)]
# col = col+[(0,0,0)]


def scatter(X, Y,
             title="No title",
             title_size=10,
             acc: "y:str_description" = {},
             markerscale=4,
             getmarker=lambda cla: {"marker": 'o'},
             col=col,
             label=None,
             alpha=None,
             legend=False,
             labeldict={},
             size=None):
    plt.title(title, size=title_size)
    Y = np.array(Y)
    size = max(int(4000 / len(Y)), 1) if not size else size
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.tick_params(left=False)
    plt.tick_params(bottom=False)

    if labeldict:
        for cla in np.unique(Y):
            plt.scatter(X[Y == cla, 0],
                        X[Y == cla, 1],
                        color=col[cla],
                        s=size,
                        marker=f"${cla}$",
                        edgecolors='none',
                        alpha=alpha,
                        label=labeldict.get(cla, str(cla)))  # str(cla)+" "+acc.get(cla,''),**getmarker(col[cla]))
    else:
            plt.scatter(X[:, 0], X[:, 1], c=Y, s=size)
    # plt.axis('off')
    # plt.xlabel('UMAP 2')
    # plt.ylabel('UMAP 1')
    if legend:
        # plt.legend(markerscale=2, ncol=2, bbox_to_anchor=(1, -.12))
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left", markerscale=1.2, fontsize=3.5)


class tinyUmap():
    def __init__(self, dim=(3, 3), size=2, lim=False):
        figs = (size * dim[1], size * dim[0])
        plt.figure(figsize=figs, dpi=300)
        self.i = 0
        self.dim = dim
        if lim:
            concatX = np.vstack(lim)
            xmin, ymin = concatX.min(axis=0)
            xmax, ymax = concatX.max(axis=0)

            def setlim():
                plt.xlim(xmin, xmax)
                plt.ylim(ymin, ymax)
            lim = setlim()
        #self.lim = lim if lim else lambda: 0
        self.lim = setlim if lim else lambda: 0

    def next(self):
        self.i = self.i + 1
        plt.subplot(*self.dim, self.i)

    def draw(self, *a, **b):
        self.next()
        self.lim()
        scatter(*a, **b)


def plot_X(Xlist, labels, plotsperline=3,
           grad=False, size=3.5, plug = False,
           mkmix = False, mixlabels = [], titles = []):

    if not titles:
        titles = Map(str, Range(Xlist))
    # make a tinyumap with the right dimensions
    itemstodraw = len(Xlist) + mkmix
    rows = ((itemstodraw - 1) // plotsperline) + 1
    columns = plotsperline if itemstodraw > plotsperline else itemstodraw
    d = tinyUmap(dim=(rows, columns), size=size, lim = Xlist)  # default is a row

    alllabels = np.concatenate(labels)
    themap = tools.spacemap(np.unique(alllabels)) if not grad else {}

    for x, y, title in zip(Xlist, labels, titles):
        y = themap.encode(y)
        # d.draw(x, y, labeldict=themap.getitem, legend = True)
        d.draw(x, y, title=title, labeldict=themap.getitem, legend = True)


    if mkmix:
        #mixlabels = [i*2 for i,stack in enumerate(Xlist) for item in stack]
        d.draw(np.vstack(Xlist),themap.encode(alllabels), labeldict= themap.getitem, legend = True)

    plt.show()




from scalp.data.transform import stack, to_arrays


def batchplot(adatas, from_obsm = 'embedding'):
    # stackedadatas
    #sm= tools.spacemap(np.unique(stacked.obs['batch']))
    #plt.scatter(*Transpose(stacked.obsm['lsa']), c= sm.encode(stacked.obs['batch']) )
    adatas = stack(adatas)
    assert adatas.obsm[from_obsm].shape[1] == 2, 'from_obsm not 2d'
    df = pd.DataFrame({a:b for a,b in zip('x y batch label'.split(),
                                          [*Transpose(adatas.obsm[from_obsm]),
                                           adatas.obs['batch'], adatas.obs['label'] ] )})
    # return df
    ax = sns.scatterplot(data = df, x = 'x', y= 'y', hue = 'label', style = 'batch', s   = 10)
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.show()


class drawMovingCenters:
    def __init__(self,XX,yy):
        # we build something like:
        # clusterid -> center1 False center3
        self.clustercenters ={}
        self.numframes = len(XX)
        self.frameid = 0

        def getacenter(x,y,label):
            y= y.values
            if label not in y:
                return False
            return x[y==label].mean(axis=0)

        for label in np.unique(np.concatenate(yy)):
            self.clustercenters[label] = [getacenter(x,y,label) for x,y in zip(XX,yy)]

    def draw(self, labelmap):
        # now we draw a frame
        alphas = 1/2**np.abs(np.array(Range(self.numframes))-self.frameid)

        for label in self.clustercenters:
            for i,(coordinate, alpha) in enumerate(zip(self.clustercenters[label][:-1],alphas)):
                if not isinstance( coordinate, bool) and not isinstance(self.clustercenters[label][i+1],bool):
                    intlabel = labelmap.getint.get(label)
                    plot((coordinate[0], self.clustercenters[label][i + 1][0]),
                         (coordinate[1], self.clustercenters[label][i+1][1]), alpha = alpha, color = col[intlabel])

        for label in self.clustercenters:
            for coordinate, alpha in zip(self.clustercenters[label],alphas):
                if not isinstance( coordinate, bool):
                    intlabel = labelmap.getint.get(label)
                    plt.text(x=coordinate[0], y= coordinate[1],s= intlabel, alpha = alpha, color = col[intlabel])
        self.frameid += 1



def mkconfusion(y1, y2):
    assert len(y1) == len(y2)
    sm2 = tools.spacemap(np.unique(y2))
    sm1 = tools.spacemap(np.unique(y1))
    res = np.zeros((sm1.len, sm2.len))
    for a, b in zip(y1, y2):
        res[sm1.getint[a], sm2.getint[b]] += 1
    return res, sm1, sm2


def confusion_matrix(y1, y2, norm=True, draw=False):
    '''
    the task here is to draw a confision matrix
    returns translated labels for y2
    '''
    cm, sm1, sm2 = mkconfusion(y1, y2)


    itemCount2 = dict(zip(*np.unique(y2, return_counts=True)))
    itemCount1 = dict(zip(*np.unique(y1, return_counts=True)))

    if norm:
        cm = cm.astype(float)
        for x in Range(itemCount2):
            for y in Range(itemCount1):
                res = (2 * cm[y, x]) / (itemCount2[sm2.getitem[x]] + itemCount1[sm1.getitem[y]])
                # print(cm[y,x])
                # print(xcounts[x],ycounts[y])
                cm[y, x] = res


    names1, new_order2 = lsa(cm, maximize=True)
    # print("hung matches:",names1,new_order2, itemCount2.keys())

    if draw:
        # new_order = np.argsort(new_order2)
        # cm[names1] = cm[new_order2]

        ylab = np.unique(y1)
        for n1, n2 in zip(names1, new_order2):
            cm[[n1, n2]] = cm[[n2, n1]]
            ylab[[n1, n2]] = ylab[[n2, n1]]

        sns.heatmap(cm, xticklabels=np.unique(y2), yticklabels=ylab, annot=False, linewidths=.5, cmap="YlGnBu",
                    square=True)
        plt.xlabel('Clusters data set 2')
        plt.ylabel('Clusters data set 1')

    g = dict(zip(new_order2, names1))
    return [sm1.getitem[g[sm2.getint[y]]] for y in y2]


def plot_confusion_matrix_twice(labels):
    '''
    does the confusionmatrix twice, normalized and unnormalized, and plots!
    '''
    y1, y2 = labels
    f = plt.figure(figsize=(16, 8))
    ax = plt.subplot(121, title='absolute hits')
    confusion_matrix(y1, y2, norm=False, draw=True)
    ax = plt.subplot(122, title='relative hits (2x hits / sumOfLabels)')
    confusion_matrix(y1, y2, norm=True, draw=True)
    plt.tight_layout()


def plot_confusion_matrix_normalized_raw(adatas, label = 'label', alignmentbase = 'pca40'):
    assert len(adatas) == 2
    adatas = align(adatas, base= alignmentbase)
    draw.plot_confusion_matrix_twice(*[x.obs['label'] for x in adatas])


def plot(adatas, projection = 'umap2', label= 'label', **kwargs):
    X = to_arrays(adatas, base=projection)
    labels = [a.obs[label] for a in adatas]
    batch_labels = [a.obs['batch'][0] for a in adatas]
    draw.plot_X(X, labels,titles = batch_labels,**kwargs)





# we just leave this here for now, seems superfluous though..
def get_hue_order_to_color_mapping(g):
  """Creates a dictionary that maps the hue order to a color value in a FacetGrid.
  Args:
    g: A FacetGrid object.
  Returns:
    A dictionary that maps the hue order to a color value in the FacetGrid.
  """
  hue_order = g.hue_order
  cmap = g.cmap
  hue_to_color_mapping = {}
  for hue in hue_order:
    color = cmap(hue)
    hue_to_color_mapping[hue] = color
  return hue_to_color_mapping



def snsplot(adatas, coordinate_label = 'pca2',label = 'label',
            splitby = 'batch', compare_label_to = None):
    '''
     an attempt to redo draw via seaborn
    '''

    # for sns we need pandas dataframes,,,
    col_order =[ a.obs[splitby].iloc[0] for a in adatas]
    adatas = stack(adatas)

    data = {a:b for a,b in zip('x y batch label'.split(),
                                          [*Transpose(adatas.obsm[coordinate_label]),
                                           adatas.obs[splitby], adatas.obs[label] ] )}
    if compare_label_to:
        data['edgecolors'] = ['True' if z else 'False' for z in adatas.obs[compare_label_to]  == adatas.obs[label]]

    data = pd.DataFrame(data)

    g = sns.FacetGrid( data, col=splitby, col_wrap=3, col_order = col_order) # , height=2, ylim=(0, 10))
    def myscatterplot(data = None,*args,**kwargs):
        edgecolors = kwargs.pop('edgecolor')
        if edgecolors:
            # kwargs['edgecolor'] =  data['edgecolors'].tolist()
            kwargs['style'] = 'edgecolors'
            kwargs['markers'] = {'True': 'o', "False": 'X'}
            kwargs['sizes'] = { "False": 4 , f"True": 10 }
            kwargs['size'] = 'edgecolors'

        sns.scatterplot(data = data, *args,**kwargs)
    g.map_dataframe( myscatterplot , x = 'x', y= 'y', hue = 'label', s   = 10,
                     edgecolor = 'edgecolors' if compare_label_to else None)



    g.add_legend()
    g.set( yticks=[])
    g.set( xticks=[])
    g.set_axis_labels("", "")
    plt.show()
