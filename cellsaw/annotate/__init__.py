from cellsaw.merge import Merge, stringdiffuse, accuracy_evaluation
from collections import Counter
import scanpy as sc
from cellsaw.annotate.draw import plot_annopair
from cellsaw.annotate.annotators import label_knn, linsum_copylabel, raw_diffusion, markercount, mergewrap, raw_diffusion_combat, tunnelclust, scanorama_integrate_diffusion
import cellsaw.preprocess as preprocess
import numpy as np


def predict_celltype(target,
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
                   n_genes = 800,
                   pp = False,
                   linear_assignment_factor= 1,
                   similarity_scale_factor = 1.0):

    assert similarity_scale_factor == 1.0, 'not implemented'

    pid = (pca_dim>0)+ (umap_dim>0)
    #merged =premerged or  mergewrap(target,source,umap_dim,pca = pca_dim, make_even=make_even)
    if pp and not premerged:
        def prep(x):
            # x.X = np.expm1(x.X)
            # x._uns.pop("log1p")
            # x.uns
            target = preprocess.annotate_genescore_single(x,selector = pp)
            # sc.pp.log1p(x)

        prep(target)
        prep(source)
    merged = premerged or mergewrap(target,
                                   source,
                                   umap_dim,
                                   pca = pca_dim,
                                    selectgenes = n_genes,
                                   make_even=make_even, sortfield = -1)
    newlabels = stringdiffuse(merged,merged.data[1].obs[source_label],sigmafac=sigmafac,
            pid = pid,
            neighbors_inter=n_inter_neighbors,
            neighbors_intra=n_intra_neighbors, linear_assignment_factor=linear_assignment_factor)
    target.obs[target_label] = newlabels
    return target


def multi_annotate(target,
                   sources,
                   annotator = lambda x:x,
                   source_label = 'celltype',
                   target_label='multisrc',
                   annotatorargs = None):

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
                **annotatorargs)
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

