from lmz import Map,Zip,Filter,Grouper,Range,Transpose
from ubergauss import tools as t
import cellsaw.preprocess as preprocess
import numpy as np
path = '/home/ubuntu/data/scdata/'
from cellsaw import io_utils as loader
import pandas as pd
from cellsaw import similarity
from cellsaw import merge
from cellsaw import similarity
import seaborn as sns
import os
import cellsaw.annotate as annotate





import notebookhelper
dataset_list = notebookhelper.filenames44

def getsamples(seed = 31337):
    cache = f'cache{seed}.oc'
    if os.path.exists(cache):
        return t.loadfile(cache)
    datasets = [loader.nuread(dir = path,
                           dataset = dataset,
                           sampleseed=seed,
                           sample_size = 1000,
                           remove_cells = {'celltype': ["no pangalo", "Unknown"]}) for dataset in dataset_list]
    t.dumpfile(datasets,cache)
    return datasets





def evalparams(params, f=annotate.predict_celltype, firsthalf = True):

    gt_method = params.pop('pair_pp')
    gt_numgenes = params.pop('pair_genes')
    gt_cmp = params.pop('pair_cmp')

    def gettasks(seed):
        datasets = getsamples(seed)
        ranked_datasets_list = similarity.rank_by_sim_splitbyname(datasets,dataset_list,
                                            method = gt_method,
                                            similarity = gt_cmp,
                                            numgenes = gt_numgenes,
                                            return_similarity = False)
        # ranked_datasets_list = [li[:2] for li in ranked_datasets_list ]

        if firsthalf:
            re =  ranked_datasets_list[:len(ranked_datasets_list)//2]
        else:
            re =  ranked_datasets_list[len(ranked_datasets_list)//2:]

        # cleaning, or reloading.. in case we do log bla on the data :)
        datasets = getsamples(seed)
        ds_dict = {d.uns['fname']:d for d in datasets}
        return [  [ds_dict[r[0].uns['fname']],ds_dict[r[1].uns['fname']]]   for r in ranked_datasets_list]


    seeds =  [43,421,31338] # full
    tasklists = t.xmap( gettasks, seeds ,n_jobs = 5)
    #tasklists = [ gettasks(seed) for seed in seeds]

    def eva(li):
        target = li[0].copy()
        source = li[1].copy()
        target = f(target,source,source_label = 'celltype',  target_label='PREDI',**params)
        return merge.accuracy_evaluation(target,true='celltype', predicted = 'PREDI')

    #seeds =  [42,420,31337,1337] # only 5 datasets
    #res = [ item for seed in seeds for item in t.xmap(eva,getsamples(seed)) ]
    #res = t.xmap(eva,( task for seed in seeds for task in gettasks(seed)),tasksperchild=5,n_jobs = 25)
    res = t.xmap(eva,(task for tasklist  in tasklists for task in tasklist),tasksperchild=5,n_jobs = 32)
    return res



from hyperopt import fmin, tpe, hp, Trials
from hyperopt.pyll import scope

##############
# optimize mymethod
#################

def evalparams_optimize(params, f=annotate.predict_celltype):
    return -np.mean(evalparams(params,f=annotate.predict_celltype))

def optimize_mymethod():
    #z.pos_args[0].pos_args[0]._obj <- this is nice in case i want to write a searchspace maker function
    space = { 'pca_dim' :           scope.int(hp.quniform('pca_dim',25,48,1)),
              'umap_dim' :          scope.int(hp.quniform('umap_dim',8,14,1)),
       'n_intra_neighbors' :        scope.int(hp.quniform('n_intra_neighbors',1,7,1)),
       'n_inter_neighbors' :        scope.int(hp.quniform('n_inter_neighbors',1,15,1)),
       'sigmafac' :                 hp.uniform('sigmafac',40,70),
       'linear_assignment_factor':  hp.uniform('linear_assignment_factor',3,8),
        'pp' :                      hp.choice('pp',['natto', 'seurat', 'cell_ranger', 'seurat_v3']),
       'n_genes' :                  scope.int(hp.quniform('n_genes',500,4000,25)),
        'pair_pp' :                      hp.choice('pair_pp',['seurat_v3','meanexpression','cell_ranger','seurat', 'natto','meanexpressionnolog']),
       'pair_genes' :                  scope.int(hp.quniform('pair_genes',500,4000,25)),
        'pair_cmp' :                      hp.choice('pair_cmp',['jaccard','cosine'])
    } #hp.uniform('linear_assignment_factor',4,12)}

    #trials = Trials()
    trials = t.loadfile('400.trials')
    best = fmin(fn=evalparams_optimize,
                space = space,
        algo=tpe.suggest,
        trials = trials,
        max_evals=600)


    # best is weird because the scopes are not int and choices are just indexes
    # {'linear_assignment_factor': 4.62261789365175, 'n_genes': 1049.0, 'n_inter_neighbors': 3.0, 'n_intra_neighbors': 4.0, 'pair_cmp': 'jaccard', 'pair_genes': 2449.0, 'pair_pp': 'cell_ranger', 'pca_dim': 36.0, 'pp': 'cell_ranger', 'sigmafac': 63.4539639069615, 'umap_dim': 8.0}

    print(best)
    #print('cv score:', evalparams(best,firsthalf=False))
    breakpoint()


##############
# optimize raw diffusion
#################

def evalparams_rawdiff(params):

    def f(a,b,**params):
        pp = params.pop('pp')
        preprocess.annotate_genescore([a,b], selector = pp)
        return annotate.raw_diffusion(a,b,**params)

    return -np.mean(evalparams(params,f=f))

def optimize_rawdiff():


    # n_neighbors = 5,gamma = .1, pca_dim = 40, umap_dim = 10, ngenes = 800

    #z.pos_args[0].pos_args[0]._obj <- this is nice in case i want to write a searchspace maker function
    space = { 'pca_dim' :           scope.int(hp.quniform('pca_dim',30,55,1)),
              'umap_dim' :          scope.int(hp.quniform('umap_dim',5,15,1)),
       'n_neighbors' :        scope.int(hp.quniform('n_neighbors',3,14,1)),
       'gamma' :                 hp.uniform('gamma',.001,1),
        'pp' :    'cell_ranger',  #                hp.choice('pp',['natto', 'seurat', 'cell_ranger', 'seurat_v3']),
       'n_genes' :   scope.int(hp.quniform('n_genes',1000,1600,50)),
        'pair_pp' :  'cell_ranger',#                    hp.choice('pair_pp',['seurat_v3','meanexpression','cell_ranger','seurat', 'natto','meanexpressionnolog']),
       'pair_genes' : 2450, #                 scope.int(hp.quniform('pair_genes',500,4000,25)),
        'pair_cmp' :  'jaccard'     #               hp.choice('pair_cmp',['jaccard','cosine'])
    } #hp.uniform('linear_assignment_factor',4,12)}

    trials = Trials()
    best = fmin(fn=evalparams_rawdiff,
                space = space,
        algo=tpe.suggest,
        trials = trials,
        max_evals=100)

    # best is weird because the scopes are not int and choices are just indexes
    # {'linear_assignment_factor': 4.62261789365175, 'n_genes': 1049.0, 'n_inter_neighbors': 3.0, 'n_intra_neighbors': 4.0, 'pair_cmp': 'jaccard', 'pair_genes': 2449.0, 'pair_pp': 'cell_ranger', 'pca_dim': 36.0, 'pp': 'cell_ranger', 'sigmafac': 63.4539639069615, 'umap_dim': 8.0}
    print(best)
    #print('cv score:', evalparams(best,firsthalf=False))
    breakpoint()



##############
# optimize raw diffusion COMBAT
#################

def evalparams_rawdiff_combat(params):

    def f(a,b,**params):
        pp = params.pop('pp')
        preprocess.annotate_genescore([a,b], selector = pp)
        return annotate.raw_diffusion_combat(a,b,**params)

    return -np.mean(evalparams(params,f=f))

def optimize_rawdiff_combat():

    # n_neighbors = 5,gamma = .1, pca_dim = 40, umap_dim = 10, ngenes = 800
    #z.pos_args[0].pos_args[0]._obj <- this is nice in case i want to write a searchspace maker function
    space = { 'pca_dim' :           scope.int(hp.quniform('pca_dim',30,55,1)),
              'umap_dim' :          scope.int(hp.quniform('umap_dim',5,15,1)),
       'n_neighbors' :        scope.int(hp.quniform('n_neighbors',3,14,1)),
       'gamma' :                 hp.uniform('gamma',.001,1),
        'pp' :    'cell_ranger',  #                hp.choice('pp',['natto', 'seurat', 'cell_ranger', 'seurat_v3']),
       'n_genes' :   scope.int(hp.quniform('n_genes',1000,1600,50)),
        'pair_pp' :  'cell_ranger',#                    hp.choice('pair_pp',['seurat_v3','meanexpression','cell_ranger','seurat', 'natto','meanexpressionnolog']),
       'pair_genes' : 2450, #                 scope.int(hp.quniform('pair_genes',500,4000,25)),
        'pair_cmp' :  'jaccard'     #               hp.choice('pair_cmp',['jaccard','cosine'])
    } #hp.uniform('linear_assignment_factor',4,12)}

    trials = Trials()
    best = fmin(fn=evalparams_rawdiff_combat,
                space = space,
        algo=tpe.suggest,
        trials = trials,
        max_evals=100)

    # best is weird because the scopes are not int and choices are just indexes
    # {'linear_assignment_factor': 4.62261789365175, 'n_genes': 1049.0, 'n_inter_neighbors': 3.0, 'n_intra_neighbors': 4.0, 'pair_cmp': 'jaccard', 'pair_genes': 2449.0, 'pair_pp': 'cell_ranger', 'pca_dim': 36.0, 'pp': 'cell_ranger', 'sigmafac': 63.4539639069615, 'umap_dim': 8.0}
    print(best)
    #print('cv score:', evalparams(best,firsthalf=False))
    breakpoint()



##############
# optimize linsum
#################

def evalparams_linsum(params):

    def f(a,b,**params):
        pp = params.pop('pp')
        preprocess.annotate_genescore([a,b], selector = pp)
        return annotate.linsum_copylabel(a,b,**params)
    return -np.mean(evalparams(params,f=f))

def optimize_linsum():
    # n_neighbors = 5,gamma = .1, pca_dim = 40, umap_dim = 10, ngenes = 800
    #z.pos_args[0].pos_args[0]._obj <- this is nice in case i want to write a searchspace maker function
    space = { 'pca_dim' :           scope.int(hp.quniform('pca_dim',30,55,1)),
              'umap_dim' :          scope.int(hp.quniform('umap_dim',5,15,1)),
       'n_genes' :   scope.int(hp.quniform('n_genes',1000,1600,50)),
        'pp' :    'cell_ranger',  #                hp.choice('pp',['natto', 'seurat', 'cell_ranger', 'seurat_v3']),
        'pair_pp' :  'cell_ranger',#                    hp.choice('pair_pp',['seurat_v3','meanexpression','cell_ranger','seurat', 'natto','meanexpressionnolog']),
       'pair_genes' : 2450, #                 scope.int(hp.quniform('pair_genes',500,4000,25)),
        'pair_cmp' :  'jaccard'     #               hp.choice('pair_cmp',['jaccard','cosine'])
    } #hp.uniform('linear_assignment_factor',4,12)}

    trials = Trials()
    best = fmin(fn=evalparams_linsum,
                space = space,
        algo=tpe.suggest,
        trials = trials,
        max_evals=100)

    # best is weird because the scopes are not int and choices are just indexes
    # {'linear_assignment_factor': 4.62261789365175, 'n_genes': 1049.0, 'n_inter_neighbors': 3.0, 'n_intra_neighbors': 4.0, 'pair_cmp': 'jaccard', 'pair_genes': 2449.0, 'pair_pp': 'cell_ranger', 'pca_dim': 36.0, 'pp': 'cell_ranger', 'sigmafac': 63.4539639069615, 'umap_dim': 8.0}
    print(best)
    #print('cv score:', evalparams(best,firsthalf=False))
    breakpoint()



##############
# optimize knn
#################

def evalparams_knn(params):

    def f(a,b,**params):
        pp = params.pop('pp')
        preprocess.annotate_genescore([a,b], selector = pp)
        return annotate.label_knn(a,b,**params)
    return -np.mean(evalparams(params,f=f))

def optimize_knn():
    # n_neighbors = 5,gamma = .1, pca_dim = 40, umap_dim = 10, ngenes = 800
    #z.pos_args[0].pos_args[0]._obj <- this is nice in case i want to write a searchspace maker function
    space = { 'pca_dim' :           scope.int(hp.quniform('pca_dim',30,55,1)),
              'umap_dim' :          scope.int(hp.quniform('umap_dim',5,15,1)),
       'n_genes' :   scope.int(hp.quniform('n_genes',1000,1600,50)),
            'k' :          scope.int(hp.quniform('k',5,20,1)),
        'pp' :    'cell_ranger',  #                hp.choice('pp',['natto', 'seurat', 'cell_ranger', 'seurat_v3']),
        'pair_pp' :  'cell_ranger',#                    hp.choice('pair_pp',['seurat_v3','meanexpression','cell_ranger','seurat', 'natto','meanexpressionnolog']),
       'pair_genes' : 2450, #                 scope.int(hp.quniform('pair_genes',500,4000,25)),
        'pair_cmp' :  'jaccard'     #               hp.choice('pair_cmp',['jaccard','cosine'])
    } #hp.uniform('linear_assignment_factor',4,12)}

    trials = Trials()
    best = fmin(fn=evalparams_knn,
                space = space,
        algo=tpe.suggest,
        trials = trials,
        max_evals=100)

    # best is weird because the scopes are not int and choices are just indexes
    # {'linear_assignment_factor': 4.62261789365175, 'n_genes': 1049.0, 'n_inter_neighbors': 3.0, 'n_intra_neighbors': 4.0, 'pair_cmp': 'jaccard', 'pair_genes': 2449.0, 'pair_pp': 'cell_ranger', 'pca_dim': 36.0, 'pp': 'cell_ranger', 'sigmafac': 63.4539639069615, 'umap_dim': 8.0}
    print(best)
    #print('cv score:', evalparams(best,firsthalf=False))
    breakpoint()


##############
# optimize
#################

def evalparams_tunnel(params):

    def f(a,b,**params):
        pp = params.pop('pp')
        preprocess.annotate_genescore([a,b], selector = pp)
        return annotate.tunnelclust(a,b,**params)
    return -np.mean(evalparams(params,f=f))

def optimize_tunnel():
    # n_neighbors = 5,gamma = .1, pca_dim = 40, umap_dim = 10, ngenes = 800
    #z.pos_args[0].pos_args[0]._obj <- this is nice in case i want to write a searchspace maker function
    space = { 'pca_dim' :           scope.int(hp.quniform('pca_dim',30,55,1)),
              'umap_dim' :          scope.int(hp.quniform('umap_dim',5,15,1)),
       'n_genes' :   scope.int(hp.quniform('n_genes',1000,1600,50)),
            #'umap_neigh' :          scope.int(hp.quniform('umap_neigh',5,20,1)),
        'pp' :    'cell_ranger',  #                hp.choice('pp',['natto', 'seurat', 'cell_ranger', 'seurat_v3']),
        'pair_pp' :  'cell_ranger',#                    hp.choice('pair_pp',['seurat_v3','meanexpression','cell_ranger','seurat', 'natto','meanexpressionnolog']),
       'pair_genes' : 2450, #                 scope.int(hp.quniform('pair_genes',500,4000,25)),
        'pair_cmp' :  'jaccard'     #               hp.choice('pair_cmp',['jaccard','cosine'])
    } #hp.uniform('linear_assignment_factor',4,12)}

    trials = Trials()
    best = fmin(fn=evalparams_tunnel,
        algo=tpe.suggest,
        trials = trials,
                space = space,
        max_evals=100)

    # best is weird because the scopes are not int and choices are just indexes
    # {'linear_assignment_factor': 4.62261789365175, 'n_genes': 1049.0, 'n_inter_neighbors': 3.0, 'n_intra_neighbors': 4.0, 'pair_cmp': 'jaccard', 'pair_genes': 2449.0, 'pair_pp': 'cell_ranger', 'pca_dim': 36.0, 'pp': 'cell_ranger', 'sigmafac': 63.4539639069615, 'umap_dim': 8.0}
    print(best)
    #print('cv score:', evalparams(best,firsthalf=False))
    breakpoint()




if __name__ == '__main__':
    # best = {'linear_assignment_factor': 5.62261789365175,
    #     'n_genes': 1050, 'n_inter_neighbors': 4, 'n_intra_neighbors': 5,
    #     'pair_cmp': 'jaccard', 'pair_genes': 2450, 'pair_pp': "cell_ranger",
    #     'pca_dim': 37, 'pp': "cell_ranger",
    #     'sigmafac': 64.4539639069615, 'umap_dim': 8}
    # evalparams(best,firsthalf = False)
    # optimize_mymethod()
    optimize_linsum()








'''

results = []
for id, target in enumerate(target_datasets):
    target = target_datasets[id].copy()
    source = ranked_datasets_list[id][1].copy()

    premerged = merge.mergewrap(target, source, umap_dim = 5, pca = 20, make_even=True, sortfield = 2)

    target = merge.annotate_label(target,source,source_label = 'celltype',
                                   target_label='predicted_celltype',
                                   premerged = premerged,
                                   pca_dim = 20, umap_dim = 5,
                                   n_intra_neighbors = 5,
                                   n_inter_neighbors = 1,
                                   make_even= False,
                                   sigmafac = 1,
                                   linear_assignment_factor = 1,
                                   similarity_scale_factor = 1.0)


    target = merge.annotate_label_linsum_copylabel(target,source,source_label = 'celltype',
                                                   target_label= 'linsumprediction', premerged = premerged,
                                                   pca_dim = 20, umap_dim = 0)



    # just stack the data and do knn
    target = merge.annotate_label_knn(target,source,source_label = 'celltype',
                                       target_label='knn',  premerged = premerged,
                                       pca_dim = 20, umap_dim = 0,k=5)

    # diffusion on stacked data
    target = merge.annotate_label_raw_diffusion(target,source,source_label = 'celltype',
                                               target_label='rawdiffusion',
                                              premerged = premerged,
                                                n_neighbors = 5,gamma = 10,
                                               pca_dim = 20, umap_dim = 10)


    # ! carefull i exclude unknowns here
    target = merge.markercount(target,source,source_label = 'celltype',
                                   target_label='markercount',  premerged = premerged,
                                   pca_dim = 20, umap_dim = 0)

    targetlabels = ['predicted_celltype','linsumprediction','knn','rawdiffusion','markercount']

    acc = { k: merge.accuracy_evaluation(target,true='celltype', predicted = k)  for k in  targetlabels}


    results.append([{k:acc[k]} for k in targetlabels])



results = [  {'method':k,'value':v} for li in results for d in  li for k,v in d.items()]
df = pd.DataFrame(results)


'''
