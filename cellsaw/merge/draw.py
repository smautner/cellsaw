import matplotlib.pyplot as plt
from ubergauss import tools
import numpy as np
from umap import UMAP
import seaborn as sns
from scipy.optimize import linear_sum_assignment as lsa
from lmz import *


col = plt.cm.get_cmap('tab20').colors
col = col+col+col+ ((0,0,0),)



def plot(merge, labels, plotsperline =3, grad=False):


    # make a tinyumap with the right dimensions
    X = merge.d2
    rows = ((len(X)-1)//plotsperline)+1
    columns  = plotsperline if  len(X) > plotsperline else len(X)
    d = tinyUmap(dim = (rows, columns))  # default is a row



    # set same limit for all the plots
    concatX  = np.vstack(X)
    xmin,ymin = concatX.min(axis = 0)
    xmax,ymax = concatX.max(axis = 0)
    xdiff = np.abs(xmax - xmin)
    ydiff = np.abs(ymax - ymin)

    for x,y in zip(X,labels):

        if not grad:
            d.draw(x,y, title=None)

            plt.xlim(xmin - 0.1 * xdiff, xmax + 0.1 * xdiff)
            plt.ylim(ymin - 0.1 * ydiff, ymax + 0.1 * ydiff)

            #plt.legend(markerscale=1.5, fontsize='small', ncol=int(len(X) * 2.5), bbox_to_anchor=(1, -.12))
        else:
            plt.scatter(x[:,0], x[:,1], c=y, s=1)

    plt.show()



def tinyumap(X,Y,
        title="No title",
        title_size=10,
        acc : "y:str_description"={},
        markerscale=4,
        getmarker = lambda cla: {"marker":'o'},
        col=col,
        label=None,
        alpha = None,
        legend = False,
        size=None):
    plt.title(title, size=title_size)
    Y=np.array(Y)
    size=  max( int(4000/Y.shape[0]), 1) if not size else size
    embed = X
    plt.gca().axes.yaxis.set_ticklabels([])
    plt.gca().axes.xaxis.set_ticklabels([])
    for cla in np.unique(Y):
        plt.scatter(embed[Y==cla, 0],
                    embed[Y==cla, 1],
                    color= col[cla],
                    s=size,
                    edgecolors = 'none',
                    alpha = alpha,
                    label=str(cla), **getmarker(cla)) #str(cla)+" "+acc.get(cla,''),**getmarker(col[cla]))
    #plt.axis('off')
    #plt.xlabel('UMAP 2')
    #plt.ylabel('UMAP 1')
    if legend:
        plt.legend(markerscale=2,ncol=2,bbox_to_anchor=(1, -.12) )


class tinyUmap():

    def __init__(self, dim=(3,3), size= 2):
        figs = (size*dim[1], size*dim[0])

        plt.figure( figsize=figs, dpi=300)
        self.i =0
        self.dim = dim

    def next(self):
        self.i= self.i+1
        plt.subplot(*self.dim,self.i)

    def draw(self, *a, **b):
        self.next()
        tinyumap(*a,**b)


def mkconfusion(y1,y2):
    assert len(y1)== len(y2)
    sm2 = tools.spacemap(np.unique(y2))
    sm1 = tools.spacemap(np.unique(y1))
    res = np.zeros((sm1.len, sm2.len))
    for a,b in zip(y1,y2):
        res[sm1.getint[a], sm2.getint[b]]+=1
    return res, sm1, sm2


def confuse(y1,y2, norm = True, draw= False):
    '''
    the task here is to draw a confision matrix
    returns translated labels for y2
    '''
    cm,sm1,sm2 = mkconfusion(y1,y2)


    #print(cm.shape, len(np.unique(y1)),  len(np.unique(y2)))
    #print(np.unique(y1),  np.unique(y2))
    # sns.heatmap(cm,xticklabels=np.unique(y2),yticklabels=np.unique(y1), annot=True, linewidths=.5,cmap="YlGnBu" , square=True)
    # plt.show()
    # plt.close()

    itemCount2 = dict(zip(*np.unique(y2,return_counts = True)))
    itemCount1 = dict(zip(*np.unique(y1,return_counts = True)))


    if norm:
        cm = cm.astype(float)
        for x in Range(itemCount2):
            for y in Range(itemCount1):
                res = (2*cm[y,x]) / (itemCount2[sm2.getitem[x]]+itemCount1[sm1.getitem[y]])
                #print(cm[y,x])
                #print(xcounts[x],ycounts[y])
                cm[y,x]  = res

    #sns.heatmap(cm,xticklabels=np.unique(y2),yticklabels=np.unique(y1), annot=False, linewidths=.5,cmap="YlGnBu" , square=True)
    # plt.title('raw data')
    #plt.show()
    # plt.close()

    names1, new_order2 = lsa(cm, maximize= True)
    #print("hung matches:",names1,new_order2, itemCount2.keys())

    if draw:
        #new_order = np.argsort(new_order2)
        #cm[names1] = cm[new_order2]

        ylab = np.unique(y1)
        for n1,n2 in zip(names1, new_order2):
            cm[[n1, n2]] = cm[[n2, n1]]
            ylab[[n1,n2]]=ylab[[n2,n1]]


        sns.heatmap(cm,xticklabels=np.unique(y2),yticklabels=ylab, annot=False, linewidths=.5,cmap="YlGnBu" , square=True)
        plt.xlabel('Clusters data set 2')
        plt.ylabel('Clusters data set 1')

    g = dict(zip(new_order2, names1))
    return [ sm1.getitem[g[sm2.getint[y]]]  for y in y2]


def confuse2(labels):
    '''
    does the confusionmatrix twice, normalized and unnormalized, and plots!
    '''
    y1,y2 =  labels
    f=plt.figure(figsize=(16,8))
    ax=plt.subplot(121,title ='absolute hits')
    confuse(y1,y2,norm=False, draw= True)
    ax=plt.subplot(122,title ='relative hits (2x hits / sumOfLabels)')
    confuse(y1,y2,norm=True, draw= True)
    plt.tight_layout()
