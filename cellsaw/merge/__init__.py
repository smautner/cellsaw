import cellsaw.merge.mergehelpers  as mergehelpers
import numpy as np
from cellsaw.merge.diffusion import diffuse
from cellsaw.merge import draw

class mergeutils:
    def getlabels(self,masked = []):
        f = lambda i,e: np.full_like(e.obs['true'],-1)  if i in masked else e.obs['true']
        return [ f(i,e) for i,e in enumerate(self.data)]


class merge(mergeutils):
    def __init__(self, adatas, selectgenes = 800, make_even = True, pca = 20, umaps = [2],
            joint_space = False,
            sortfield = 0,
            titles = "ABCDEFGHIJKIJ"):

        shapesbevorepp= [a.X.shape for a in adatas]
        self.genescores = [a.varm['scores'] for a in adatas]
        self.geneab = [a.varm['genes'] for a in adatas]
        self.data  = mergehelpers.unioncut(self.genescores, selectgenes, adatas)

        # geneab = np.all(np.array(self.geneab), axis=0)
        # for i, d in enumerate(self.data):
        #     data[i] = d[:, geneab]

        if make_even:
            self.data = mergehelpers.make_even(self.data)


        print("preprocess:", end= '')
        for a,b in zip(shapesbevorepp, self.data):
            print(f"{a} -> {b.shape}")

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
    #
    def hungarian(self,data_fld,data_id, data_id2):
            hung, dist = mergehelpers.hungarian(self.projections[data_fld][data_id],self.projections[data_fld][data_id2])
            return hung, dist[hung]


