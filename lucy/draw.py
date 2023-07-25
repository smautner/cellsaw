
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy as hira
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import precision_score
import pandas as pd
import matplotlib.pyplot as plt
from ubergauss import tools
import numpy as np
from umap import UMAP
import seaborn as sns
from scipy.optimize import linear_sum_assignment as lsa
from lmz import *

col = plt.cm.get_cmap('tab20').colors
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




def batchplot(adatas):
    # stackedadatas

    #sm= tools.spacemap(np.unique(stacked.obs['batch']))
    #plt.scatter(*Transpose(stacked.obsm['lsa']), c= sm.encode(stacked.obs['batch']) )

    df = pd.DataFrame({a:b for a,b in zip('x y batch label'.split(),
                                          [*Transpose(adatas.obsm['lsa']),
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
                    plt.plot((coordinate[0],self.clustercenters[label][i+1][0]),
                             (coordinate[1], self.clustercenters[label][i+1][1]) , alpha = alpha, color = col[intlabel])

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




import pprint
from collections import Counter, defaultdict
from ubergauss.tools import spacemap
import matplotlib as mpl

def mkcolors(label):
    colorsm = spacemap(np.unique(label))
    cmap = plt.cm.get_cmap('turbo', len(colorsm.integerlist))
    myrgb = Map(cmap, colorsm.encode(label))
    return Map(mpl.colors.rgb2hex, myrgb)



def  add_by_leftk(cnt, leftk, support_ab, support_ba):
    a_outcount = defaultdict(list)
    for (a,b),count in cnt.items():
        a_outcount[a].append((count,a,b))

    clean_count = {}
    for a in a_outcount.keys():
        asd = sorted(a_outcount[a], key = lambda x: x[0], reverse= True)[:leftk]
        for count, a, b in asd:
            clean_count[(a,b)] = count
            support_ba.pop(b, None)
            support_ab.pop(a, None)
    return clean_count


def  add_by_rightk(cnt, rightk, support_ab, support_ba):

    b_outcount = defaultdict(list)
    for (a,b),count in cnt.items():
        b_outcount[b].append((count,a,b))

    clean_count = {}
    for bkey in b_outcount.keys():
        asd = sorted(b_outcount[bkey], key = lambda x: x[0], reverse= True)[:rightk]
        for count, a, b in asd:
            clean_count[(a,b)] = count
            support_ba.pop(b, None)
            support_ab.pop(a, None)
    return clean_count


def  add_by_thresh(cnt, thresh, support_ab, support_ba):
    # sum outgoing for each source
    a_outcount = defaultdict(int)
    for (a,b),count in cnt.items():
        a_outcount[a] += count

    clean_count = {}
    # add what is passing the threshold
    for (a,b),count in cnt.items():
        if count > a_outcount[a] * thresh:
            clean_count[(a,b)] = count
            # popping b so we will be left with the unaccounted
            support_ba.pop(b, None)
            support_ab.pop(a, None)

    return clean_count

def clean_counter(cnt, thresh=1, leftk = 0, rightk = 0):
    '''
    we remove connectins from the counter to remove noise
    - threshold is for the outflow of a..
        if a connection a->b has below threshold connections, we remove it
        however we can not remove all such connections as lone
        instances in b need to be preserverd
    '''
    # we cant drop targets so we keep a list and remove the covered ones later
    support_ba = defaultdict(list)
    for (a,b),count in cnt.items():
        support_ba[b].append([count, a])

    support_ab = defaultdict(list)
    for (a,b),count in cnt.items():
        support_ab[a].append([count, b])


    clean_count = add_by_thresh(cnt, thresh, support_ab, support_ba)
    clean_count.update(add_by_leftk(cnt, leftk, support_ab, support_ba))
    clean_count.update(add_by_rightk(cnt, rightk, support_ab, support_ba))


    # add the remaining b instances
    # adding all adds too much junk
    # so we sort and only add one
    for b,aa in support_ba.items():
        aa = sorted(aa, key = lambda x:x[0], reverse = True)
        a = aa[0][1]
        clean_count[(a,b)] = cnt[(a,b)]
    for a,bb in support_ab.items():
        bb = sorted(bb, key = lambda x:x[0], reverse = True)
        b = bb[0][1]
        clean_count[(a,b)] = cnt[(a,b)]


    return clean_count




def adatas_to_sankey(adatas, thresh = .1, leftk = 0, rightk = 0, labelfield = f'label'):
    source,target ,value = [],[],[]

    node_groups = []

    for i in range(len(adatas)-1):
        a1 = adatas[i]
        a2 = adatas[i+1]
        c = Counter(zip(a1.obs[labelfield],a2.obs[labelfield]))
        c = clean_counter(c, thresh = thresh, leftk=leftk, rightk= rightk)

        s,t = Transpose(list(c.keys()))
        source+=[ss+str(i) for ss in s]
        target+=[tt+str(i+1) for tt in t]

        value += list(c.values())
        node_groups.append( [ss+str(i) for ss in s])


    sm = spacemap(np.unique(source+target))

    label = [s[:-1] for s in sm.itemlist ]

    # node_groups = Map(sm.encode, node_groups) doesnt work //

    return {'label':label, 'color':mkcolors(label)}, {'source':sm.encode(source), 'target':sm.encode(target), 'value':value}

def adatas_to_sankey_fig(adatas, align = False, thresh = .15, leftk= 0, rightk = 0, label ='label'):
    import plotly.graph_objects as go
    if align:
        import lucy.adatas as ada
        ada.align(adatas, base = align)

    node,link = adatas_to_sankey(adatas, thresh = thresh, leftk=leftk, rightk= rightk, labelfield = label)
    fig = go.Figure(data=[go.Sankey(
        node = node,
        link = link
        )])

    fig.update_layout( hovermode = 'x', title="title",
    font=dict(size = 10, color = 'black'),
    plot_bgcolor='white',
    paper_bgcolor='white')

    return fig
