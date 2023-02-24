import cellsaw.merge.mergehelpers  as mergehelpers
import numpy as np
from cellsaw.merge.diffusion import Diffusion, stringdiffuse
from cellsaw.draw import plot_merge, confuse2
import logging
class mergeutils:
    def getlabels(self,masked = [], label = 'true', fill = -1):
        f = lambda i,e: np.full_like(e.obs[label],fill)  if i in masked else e.obs[label]
        return [ f(i,e) for i,e in enumerate(self.data)]

    def confuse2(self, labels):
        assert self.sorted
        assert len(self.data) == 2, 'i could just assume that you mean the first 2 datasets, but i refuse :D'
        assert len(labels[0])== self.data[0].shape[0]
        assert len(labels[1])== self.data[1].shape[0]

        confuse2(labels)

    def size(self, pid=-1):
        return [x.shape for x in self.projections[pid]]

    def plot(self, labels, **kwargs):
        plot_merge(self, labels,**kwargs)



class Merge(mergeutils):
    def __init__(self, adatas, selectgenes = 2000,
            make_even = True, pca = 40, umaps = [2],
            joint_space = True,
            sortfield = 0,
            oldcut = False,
            genescoresid = '',
            titles = "ABCDEFGHIJKIJ"):
        assert isinstance(adatas, list), f'merge wants a list, not {type(adatas)}'
        shapesbevorepp= [a.X.shape for a in adatas]
        assert all([a.X.shape[1] == adatas[0].X.shape[1] for a in adatas])
        #self.genescores = [a.varm['scores'] for a in adatas]

        scorename = genescoresid or adatas[0].uns['lastscores']
        self.genescores = [a.varm[scorename] for a in adatas]
        self.similarity = calc_similarity(self.genescores)
        #self.geneab = [a.varm['genes'] for a in adatas]
        if oldcut:
            self.data  = mergehelpers.unioncut(self.genescores, selectgenes, adatas)
        else:
            self.data  = mergehelpers.equal_contrib_gene_cut(self.genescores, selectgenes, adatas)
        self.sorted = False
        self.jointSpace = joint_space

        # geneab = np.all(np.array(self.geneab), axis=0)
        # for i, d in enumerate(self.data):
        #     data[i] = d[:, geneab]

        if make_even:
            self.data = mergehelpers.make_even(self.data)



        logging.info("preprocess:")
        logging.info(f"{make_even=}")
        for a,b in zip(shapesbevorepp, self.data):
            logging.info(f"{a} -> {b.shape}")

        ######################
        self.titles = titles

         # do dimred
        self.projections = [[ d.X for d in self.data]]+mergehelpers.dimension_reduction(self.data,
                                                                False, # scale (will still scale if pca)
                                                                False, # will be ignored anyway
                                                                PCA=pca,
                                                                umaps=umaps,
                                                                joint_space=joint_space)

        if pca:
            self.PCA = self.projections[1]

        if sortfield >=0:
            assert make_even==True, 'sorting uneven datasets will result in an even dataset :)'
            self.sort_cells(sortfield)

        if umaps:
            for x,d in zip(umaps,self.projections[int(pca>0)+1:]):
                self.__dict__[f"d{x}"] = d


    def sort_cells(self,projection_id = 0):
        # loop over data sets
        self.hungdist =[]
        for i in range(len(self.data)-1):
            hung, dist = self.hungarian(projection_id,i,i+1)
            self.hungdist.append(dist)
            #self.data[i+1] = self.data[i+1][hung[1]]
            #self.data[i+1].X = self.data[i+1].X[hung[1]]
            for x in range(len(self.projections)):
                self.projections[x][i+1] = self.projections[x][i+1][hung[1]]
                if x == 0:
                    self.data[i+1]= self.data[i+1][hung[1]]

        self.sorted = True

    #
    def hungarian(self,data_fld,data_id, data_id2):
            hung, dist = mergehelpers.hungarian(self.projections[data_fld][data_id],self.projections[data_fld][data_id2])
            return hung, dist[hung]


from sklearn.metrics import accuracy_score as acc
def accuracy_evaluation(target, true = '', predicted = ''):
    t = target.obs[true]
    p = target.obs[predicted]
    t =list(t)
    p=list(p)
    #for a in zip(t,p): print (a)
    return acc(t,p)



import ubergauss.tools as ut
def calc_similarity(scorelist):
    def dist(a,b):
        asd = np.array([ ut.binarize(d,2000)  for d in [a,b] ])
        union = np.sum(np.any(asd, axis=0))
        intersect = np.sum(np.sum(asd, axis=0) ==2)
        return intersect/union
    res = [[ dist(a,b) for a in scorelist] for b in scorelist]
    return np.array(res)
