
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


def plot_merge(merge, labels, plotsperline=3, grad=False, size=3.5, plug = False, mkmix = False, mixlabels = []):
    '''scatterplots for merge.d2'''

    # make a tinyumap with the right dimensions

    X = merge.d2

    itemstodraw = len(X) + mkmix
    rows = ((itemstodraw - 1) // plotsperline) + 1
    columns = plotsperline if itemstodraw > plotsperline else itemstodraw

    d = tinyUmap(dim=(rows, columns), size=size)  # default is a row

    # set same limit for all the plots
    concatX = np.vstack(X)
    xmin, ymin = concatX.min(axis=0)
    xmax, ymax = concatX.max(axis=0)

    # xdiff = np.abs(xmax - xmin)
    # ydiff = np.abs(ymax - ymin)
    # plt.xlim(xmin - 0.1 * xdiff, xmax + 0.1 * xdiff)
    # plt.ylim(ymin - 0.1 * ydiff, ymax + 0.1 * ydiff)
    themap = tools.spacemap(np.unique(np.concatenate(labels)))
    for x, y in zip(X, labels):
        #y,sm = tools.labelsToIntList(y)
        y = themap.encode(y)
        if not grad:
            d.draw(x, y, title=None, labeldict=themap.getitem)
            # plt.legend(markerscale=1.5, fontsize=4, ncol=int(len(X) * 2.5), bbox_to_anchor=(1.1, -.01))
            # plt.legend(markerscale=1.5, fontsize=4, ncol=int(len(X) * 2.5))
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left", markerscale=1.2, fontsize=3.5)
        else:
            d.next()
            plt.gca().axes.yaxis.set_ticklabels([])
            plt.gca().axes.xaxis.set_ticklabels([])
            plt.scatter(x[:, 0], x[:, 1], c=y, s=1)
            # plt.colorbar(shrink=.5)
            # plt.tick_params(labelsize=4)
        if plug:
            plug.draw(themap)
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)


    if mkmix:
        if not mixlabels:
            mixlabelslabels = [i*2 for i,stack in enumerate(X) for item in stack]
        else:
            mixlabels = themap.encode(mixlabels)
        d.draw(np.vstack(merge.d2),mixlabels)
        plt.legend(bbox_to_anchor=(1, 1), loc="upper left", markerscale=1.2, fontsize=3.5)

    plt.show()


def tinyumap(X, Y,
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
    size = max(int(4000 / Y.shape[0]), 1) if not size else size
    embed = X
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.xaxis.set_ticklabels([])
    plt.tick_params(left=False)
    plt.tick_params(bottom=False)
    for cla in np.unique(Y):
        plt.scatter(embed[Y == cla, 0],
                    embed[Y == cla, 1],
                    color=col[cla],
                    s=size,
                    marker=f"${cla}$",
                    edgecolors='none',
                    alpha=alpha,
                    # label= labeldict.get(cla,str(cla)), **getmarker(cla)) #str(cla)+" "+acc.get(cla,''),**getmarker(col[cla]))
                    label=labeldict.get(cla, str(cla)))  # str(cla)+" "+acc.get(cla,''),**getmarker(col[cla]))
    # plt.axis('off')
    # plt.xlabel('UMAP 2')
    # plt.ylabel('UMAP 1')
    if legend:
        plt.legend(markerscale=2, ncol=2, bbox_to_anchor=(1, -.12))


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
        self.lim = lim if lim else lambda: 0

    def next(self):
        self.i = self.i + 1
        plt.subplot(*self.dim, self.i)

    def draw(self, *a, **b):
        self.next()
        self.lim()
        tinyumap(*a, **b)


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


def confuse(y1, y2, norm=True, draw=False):
    '''
    the task here is to draw a confision matrix
    returns translated labels for y2
    '''
    cm, sm1, sm2 = mkconfusion(y1, y2)

    # print(cm.shape, len(np.unique(y1)),  len(np.unique(y2)))
    # print(np.unique(y1),  np.unique(y2))
    # sns.heatmap(cm,xticklabels=np.unique(y2),yticklabels=np.unique(y1), annot=True, linewidths=.5,cmap="YlGnBu" , square=True)
    # plt.show()
    # plt.close()

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

    # sns.heatmap(cm,xticklabels=np.unique(y2),yticklabels=np.unique(y1), annot=False, linewidths=.5,cmap="YlGnBu" , square=True)
    # plt.title('raw data')
    # plt.show()
    # plt.close()

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


def confuse2(labels):
    '''
    does the confusionmatrix twice, normalized and unnormalized, and plots!
    '''
    y1, y2 = labels
    f = plt.figure(figsize=(16, 8))
    ax = plt.subplot(121, title='absolute hits')
    confuse(y1, y2, norm=False, draw=True)
    ax = plt.subplot(122, title='relative hits (2x hits / sumOfLabels)')
    confuse(y1, y2, norm=True, draw=True)
    plt.tight_layout()


################
# similarity related
################
def dendro(simiarity, labels, distancecut = 1):
    print ('deprecated?')
    '''
    # HEATMAP
    f=plt.figure(figsize=(16,8))
    ax=plt.subplot(121)
    ax.set_title('complete similarity matrix', fontsize=20)
    sns.heatmap(similarity, xticklabels = False,
            yticklabels = labels,  cmap="YlGnBu")
    locs, labels = plt.yticks()
    plt.setp(labels,size = 8)

    # DENDROGRAM
    ax=plt.subplot(122)
    ax.set_title('induced dendrogram (ward)', fontsize=20)
    Z = squareform(similarity)
    Z = hira.linkage(1-Z,'ward')
    hira.dendrogram(Z, labels = labels, color_threshold=distancecut, orientation='right')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90,size = 8)
    plt.subplots_adjust(wspace = .4)
    return hira.fcluster(Z, t=7, criterion = 'maxclust')
    '''


def dendro_degen(similarity, xlabels,ylabels, distancecut = 1):
    '''
    so for the new mode the matrix is not symmetrical anymore..
    '''

    f=plt.figure(figsize=(16,8))
    ax=plt.subplot(122)

    # HEATMAP
    ax.set_title('complete similarity matrix', fontsize=20)
    sns.heatmap(similarity, xticklabels = xlabels,
            yticklabels = ylabels,  cmap="hot")
    locs, labels = plt.yticks()
    plt.setp(labels,size = 8)


    # DENDROGRAM
    ax=plt.subplot(121)

    ax.set_title('induced dendrogram (ward)', fontsize=20)
    #Z = squareform(similarity)
    Z = similarity
    Z = hira.linkage(1-Z,'ward')
    hira.dendrogram(Z, labels = labels, color_threshold=distancecut, orientation='right')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90,size = 8)
    plt.subplots_adjust(wspace = .4)


    return hira.fcluster(Z, t=7, criterion = 'maxclust')


def plotPrecision(data, wg = True, method='cosine similarity'):
    # plor precision at k :)
    # [val,rep,p@k,n_genes]
    df = pd.DataFrame(data)
    df.columns = 'precision rep neighbors±σ genes sim preproc'.split()
    title = f'Searching for similar datasets via {method}'

    if wg:
        sns.set_theme(style='whitegrid')
    sns.lineplot(data = df,x='genes', y = 'precision',style= 'neighbors±σ',
            hue = 'neighbors±σ',palette="flare", ci=68)
    plt.title(title, y=1.06, fontsize = 16)
    plt.ylim([.75,1])
    plt.ylabel('precision of neighbors (40 datasets)')
    plt.xlabel('number of genes')


import gif
def makegif(merge, labels, plug):
	gif.options.matplotlib["dpi"] = 200
	def plot_merge(merge, labels, plotsperline=3, grad=False, size=3.5, plug = False):
		X = merge.d2
		#rows = ((len(X) - 1) // plotsperline) + 1
		#columns = plotsperline if len(X) > plotsperline else len(X)
		d = tinyUmap(dim=(1, len(merge.d2)), size=size)  # default is a row
		plt.close()
		# set same limit for all the plots
		concatX = np.vstack(X)
		xmin, ymin = concatX.min(axis=0)
		xmax, ymax = concatX.max(axis=0)

		themap = tools.spacemap(np.unique(np.concatenate(labels)))
		@gif.frame
		def myplotter(x,y):
			#y,sm = tools.labelsToIntList(y)
			y = themap.encode(y)
			if not grad:
				tinyumap(x, y, title=None, labeldict=themap.getitem)
				# plt.legend(markerscale=1.5, fontsize=4, ncol=int(len(X) * 2.5), bbox_to_anchor=(1.1, -.01))
				# plt.legend(markerscale=1.5, fontsize=4, ncol=int(len(X) * 2.5))
				plt.legend(bbox_to_anchor=(1, 1), loc="upper left", markerscale=1.2, fontsize=3.5)
			else:
				plt.gca().axes.yaxis.set_ticklabels([])
				plt.gca().axes.xaxis.set_ticklabels([])
				plt.scatter(x[:, 0], x[:, 1], c=y, s=1)
				# plt.colorbar(shrink=.5)
				# plt.tick_params(labelsize=4)
			if plug:
				plug.draw(themap)
			plt.xlim(xmin, xmax)
			plt.ylim(ymin, ymax)

		frames = []
		for x, y in zip(X, labels):
			frames.append(myplotter(x,y))
		gif.save(frames, 'meganice.gif', duration=1000)
	plot_merge(merge,labels,plug=plug)
