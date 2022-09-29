import cellsaw.merge.mergehelpers  as mergehelpers
import numpy as np
from cellsaw.merge.diffusion import Diffusion, stringdiffuse
from cellsaw.merge import draw

class mergeutils:
    def getlabels(self,masked = [], label = 'true', fill = -1):
        f = lambda i,e: np.full_like(e.obs[label],fill)  if i in masked else e.obs[label]
        return [ f(i,e) for i,e in enumerate(self.data)]

    def confuse2(self, labels):
        assert self.sorted
        assert len(self.data) == 2, 'i could just assume that you mean the first 2 datasets, but i refuse :D'
        assert len(labels[0])== self.data[0].shape[0]
        assert len(labels[1])== self.data[1].shape[0]

        draw.confuse2(labels)


    def plot(self, labels, **kwargs):
        draw.plot(self, labels,**kwargs)



class merge(mergeutils):
    def __init__(self, adatas, selectgenes = 800,
            make_even = True, pca = 20, umaps = [2],
            joint_space = True,
            sortfield = 0,
            titles = "ABCDEFGHIJKIJ"):

        shapesbevorepp= [a.X.shape for a in adatas]
        self.genescores = [a.varm['scores'] for a in adatas]
        self.geneab = [a.varm['genes'] for a in adatas]
        self.data  = mergehelpers.unioncut(self.genescores, selectgenes, adatas)
        self.sorted = False
        self.jointSpace = joint_space

        # geneab = np.all(np.array(self.geneab), axis=0)
        # for i, d in enumerate(self.data):
        #     data[i] = d[:, geneab]

        if make_even:
            self.data = mergehelpers.make_even(self.data)


        print("preprocess:", end= '')
        print(f"{make_even=}")
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

        self.sorted = True

    #
    def hungarian(self,data_fld,data_id, data_id2):
            hung, dist = mergehelpers.hungarian(self.projections[data_fld][data_id],self.projections[data_fld][data_id2])
            return hung, dist[hung]










def mergewrap(a,b,umap_dim,**kwargs):
    assert  isinstance(umap_dim, int), 'umap_dim must be an integer'
    umaps = [] if umap_dim == 0 else [umap_dim]
    m =  merge([a,b],umaps=umaps,**kwargs)

    return m





def annotate_label(target,
                   source,
                   source_label = 'celltype',
                   target_label='predicted_celltype',
                   pca_dim = 20,
                   umap_dim = 0,
                   make_even = True,
                   sigmafac = 1,
                   n_intra_neighbors = 7,
                   n_inter_neighbors = 2,
                   premerged = False,
                   linear_assignment_factor= 1,
                   similarity_scale_factor = 1.0):

    assert similarity_scale_factor == 1.0, 'not implemented'

    pid = (pca_dim>0)+ (umap_dim>0)
    merged =premerged or  mergewrap(target,source,umap_dim,pca = pca_dim, make_even=make_even)
    newlabels = stringdiffuse(merged,merged.data[1].obs[source_label],sigmafac=sigmafac,
            pid = pid,
            neighbors_inter=n_inter_neighbors,
            neighbors_intra=n_intra_neighbors, linear_assignment_factor=linear_assignment_factor)
    target.obs[target_label] = newlabels
    return target

from collections import Counter
import scanpy as sc
def multi_annotate(target,
                   sources,
                   annotator = lambda x:x,
                   source_label = 'celltype',
                   target_label='multisrc',
                   **kwargs):

    # we should not be loosing cells:
    sc.pp.subsample(target,
            fraction=None,
            n_obs=min([a.shape[0] for a in sources+[target]]),
            random_state=1227, copy=False)


    # annotate
    allobs = []
    for i,source in enumerate(sources):
        # we send a copy in as merge will delete genes :)
        target_tmp = annotator(target.copy(),source.copy(),
                source_label = source_label,
                target_label = f'{target_label}_{i}',
                **kwargs)
        allobs.append(target_tmp.obs[f'{target_label}_{i}'])

    def calclabel(a):
        c = Counter(a)
        maxcount = max(c.values())
        # so we return the majority,
        # problem might be that the cardinality might show up twice
        # this is no problem since counter seems to list keys in order of reading
        # so we get the 'left' most value in case of a collision
        for k,v in c.items():
            if maxcount ==v:
                return k

    target.obs[target_label] = [ calclabel(a) for a in zip(*allobs)  ]
    return target




from cellsaw.merge.mergehelpers import hungarian

def annotate_label_linsum_copylabel(
                                target,
                                source,
                               source_label = 'celltype',
                               target_label= 'diffuseknn',
                               premerged = False,
                               pca_dim = 20, umap_dim = None):

    pid = (pca_dim>0)+ (umap_dim>0)
    merged =   premerged or            mergewrap(target,source,umap_dim,pca=pca_dim, sortfield = pid)
    target.obs[target_label] = list(merged.data[1].obs[source_label])
    return target


from sklearn.neighbors import KNeighborsClassifier as knn
def annotate_label_knn(target,source,
                              source_label = 'celltype',
                              target_label='knn',
                              premerged = False,
                              pca_dim = 20, umap_dim = None, k = 5):


    pid = (pca_dim>0)+ (umap_dim>0)
    merged = premerged or  mergewrap(target,source,umap_dim,pca=pca_dim)
    a,b = merged.projections[pid]
    y = merged.data[1].obs['celltype']
    model = knn(n_neighbors=k).fit(a,y)
    target.obs[target_label] = model.predict(b)
    return target



from sklearn.semi_supervised import LabelPropagation as lapro
from sklearn.semi_supervised import LabelSpreading as laspre
def annotate_label_raw_diffusion(target,source,source_label = 'celltype',
                               target_label='raw_diffusion',
                                 premerged = False,
                                 n_neighbors = 5,gamma = 10,
                               pca_dim = 20, umap_dim = None):
    pid = (pca_dim>0)+ (umap_dim>0)
    print(f"{pid=}")
    merged = premerged or mergewrap(target,source,umap_dim,pca=pca_dim)
    a,b = merged.projections[pid]
    y = merged.data[1].obs[source_label]
    #diffusor = laspre( gamma = .1, n_neighbors = 5, alpha = .4).fit(b,y)
    diffusor = lapro( gamma = gamma, n_neighbors = n_neighbors).fit(b,y)
    target.obs[target_label] = diffusor.predict(a)
    return target






from MarkerCount.marker_count import MarkerCount_Ref, MarkerCount
import warnings


def markercount(target,source,source_label = 'celltype',
                               target_label='markercount_celltype',premerged = False,
                               pca_dim = 20, umap_dim = None):


    pid = (pca_dim>0)+ (umap_dim>0)
    merged = premerged or mergewrap(target,source,umap_dim,pca=pca_dim)

    X_ref=merged.data[1].to_df()
    X_test=merged.data[0].to_df()
    reflabels = merged.data[1].obs[source_label]
    #print(f"{X_ref.shape=}{X_test.shape=}{reflabels=}")
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df_res = MarkerCount_Ref( X_ref, reflabels, X_test,
                      cell_types_to_excl = ['Unknown'],
                      log_transformed = True,
                      file_to_save_marker = 'my_markers',
                      verbose = False )
        # get results :D
        predict = df_res['cell_type_pred']
        target.obs[target_label] = predict
    return target

from ubergauss import tools as ut

import matplotlib.pyplot as plt
def plot(source,target,source_label = '', target_label ='', pca= 20):




    umaps = [2] if pca >2 else []
    if id(source) == id(target):
        merged = merge([target, target.copy()], umaps=umaps,pca = pca)
    else:
        merged = merge([source,target], umaps=umaps, pca = pca)

    s,t = merged.projections[2] if pca > 2 else merged.projections[1]



    d = draw.tinyUmap(dim=(1,3), lim = [s,t])

    tlab = list(merged.data[1].obs[target_label])
    slab = list(merged.data[0].obs[source_label])
    allthelabels = tlab + slab
    all,sm = ut.labelsToIntList(allthelabels)


    labeld = {k:v[:8] for k,v in sm.getitem.items()}

    size = 10
    d.draw(t,sm.encode(tlab), title = f'T:{target_label}', labeldict = labeld,size= size)
    setlim()

    def plot_cellids(t):
        [plt.text(a, b, str(i),fontsize='xx-small') for i,(a,b) in enumerate(t.tolist())]

    # plot_cellids(t)
    d.draw(s,sm.encode(slab), title = f'S:{source_label}', labeldict = labeld, size=size)
    setlim()
    # plot_cellids(s)
    d.draw(np.vstack((t,s)),all, title = 'Both', labeldict = labeld,size= size)
    setlim()
    #draw.plt.legend()

    draw.plt.legend(markerscale=4,ncol=3,bbox_to_anchor=(1, -.12) )
    return merged


from sklearn.metrics import accuracy_score as acc
def accuracy_evaluation(target, true = '', predicted = ''):
    t = target.obs[true]
    p = target.obs[predicted]
    t =list(t)
    p=list(p)
    #for a in zip(t,p): print (a)
    return acc(t,p)


