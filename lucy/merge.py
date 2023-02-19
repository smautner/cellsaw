import numpy as np
import logging
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import pairwise_distances


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
        #self.geneab = [a.varm['genes'] for a in adatas]
        if oldcut:
            self.data  = unioncut(self.genescores, selectgenes, adatas)
        else:
            self.data  = equal_contrib_gene_cut(self.genescores, selectgenes, adatas)
        self.sorted = False
        self.jointSpace = joint_space

        # geneab = np.all(np.array(self.geneab), axis=0)
        # for i, d in enumerate(self.data):
        #     data[i] = d[:, geneab]

        if make_even:
            self.data = make_even(self.data)


        logging.info("preprocess:")
        logging.info(f"{make_even=}")
        for a,b in zip(shapesbevorepp, self.data):
            logging.info(f"{a} -> {b.shape}")

        ######################
        self.titles = titles

        # do dimred
        self.projections = [[ d.X for d in self.data]]+dimension_reduction(self.data,
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
        hung, dist = hungarian(self.projections[data_fld][data_id],
                                            self.projections[data_fld][data_id2])
        return hung, dist[hung]




def hungarian(X1, X2, debug = False):
    distances = pairwise_distances(X1,X2, metric=metric)
    row_ind, col_ind = linear_sum_assignment(distances)
    if debug:
        x = distances[row_ind, col_ind]
        num_bins = 100
        print("hungarian: debug hist")
        plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
        plt.show()
    return (row_ind, col_ind), distances


def unioncut(scores, numGenes, data):
    indices = np.argpartition(scores, -numGenes)[:,-numGenes:]
    indices = np.unique(indices.flatten())
    [d._inplace_subset_var(indices) for d in data]
    return data

def equal_contrib_gene_cut(scores,numgenes,data):
    ar = np.array(scores)
    ind = np.argsort(ar)
    def calcoverlap(scores,number):
        indices = scores[:,-number:]
        indices = np.unique(indices.flatten())
        return indices
    def findcutoff(low,high, lp = -1):
        probe = int((low+high)/2)
        y = calcoverlap(ind,probe)
        if probe == lp:
            return y
        if len(y) > numgenes:
            return findcutoff(low,probe,probe)
        else:
            return findcutoff(probe,high,probe)
    indices = findcutoff(0,numgenes)
    [d._inplace_subset_var(indices) for d in data]
    return data


def umapify(dx, dimensions,n_neighbors=10):
    mymap = umap.UMAP(n_components=dimensions,
                      n_neighbors=n_neighbors,
                      random_state=1337).fit(np.vstack(dx))
    return [mymap.transform(a) for a in dx]



def dimension_reduction(adatas, scale, zero_center, PCA, umaps):
    # get a (scaled) dx
    if scale or PCA:
        adatas= [sc.pp.scale(adata, zero_center=False, copy=True,max_value=10) for adata in adatas]
    dx = [adata.to_df().to_numpy() for adata in adatas]

    res = []
    if PCA:
        pca = decomposition.PCA(n_components=PCA)
        pca.fit(np.vstack(dx))
        #print('printing explained_variance\n',list(pca.explained_variance_ratio_))# rm this:)
        dx = [ pca.transform(e) for e in dx  ]
        res.append(dx)

    for dim in umaps:
        assert 0 < dim < PCA, f'{dim=} {PCA=}'
        res.append(umapify(dx,dim))

    return res

    def make_even(data):
        # assert all equal
        size = data[0].X.shape[1]
        assert all([size == other.X.shape[1] for other in data])

        # find smallest
        counts = [e.X.shape[0] for e in data]
        smallest = min(counts)

        for a in data:
            if a.X.shape[0] > smallest:
                sc.pp.subsample(a,
                                fraction=None,
                                n_obs=smallest,
                                random_state=0,
                                copy=False)
        return data
