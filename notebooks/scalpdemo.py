from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
'''
so we exported scalp demo to a script... lets see what we can do
'''


import matplotlib as mpl


if __name__ == "__main__":
    # mpl.use('Agg')  # Use a non-interactive backend for script execution
    mpl.use('module://matplotlib-backend-sixel')

from matplotlib import pyplot as plt
import warnings
from collections import defaultdict
import scalp
from scalp.output import draw
import lmz
import numpy as np
import pandas as pd
import scanpy as sc
import functools
import time
import seaborn as sns # Added for plotting
import better_exceptions

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", module="anndata")



'''
HOWTO:
import scalpdemo as demo
ds,d = demo.get_data()
demo.Scalp(ds)
'''


# In[3]:


conf = {'maxdatasets':10, 'maxcells':1000,'filter_clusters': 10, 'slow':0}
conf = {'maxdatasets':4, 'maxcells':500,'filter_clusters': 10, 'slow':0}

def get_data():
    if True:
        datasets = scalp.data.scmark(scalp.test_config.scmark_datapath,  **conf)
        print(f"{len(datasets)=}")
        datasets += scalp.data.timeseries(scalp.test_config.timeseries_datapath,**conf)
        print(f"{len(datasets)=}")
        datasets +=scalp.data.scib(scalp.test_config.scib_datapath,**conf)
        print(f"{len(datasets)=}")
        dataset = datasets[12]
    else:
        datasets = list(scalp.data.timeseries(scalp.test_config.timeseries_datapath,datasets = ['s5'],**conf))
        dataset = datasets[0]

    return datasets, dataset



import scanpy as sc


# # COMPARISON + SCORING

import scanpy as sc
from scalp import graph as sgraph
from scipy.sparse import csr_matrix
import scalp.data.similarity as sim
import umap
import pandas as pd

def setup_grid(ax, dataset):
    counts = pd.Series(dataset.obs['batch']).value_counts()
    ticks = np.cumsum([0] + counts).to_list()
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.grid(True, color='white', linestyle='--', linewidth=0.5)

def Scalp(dataset, dim = 2, ot= .97):
    # if find_duplicate_rows(dataset.X): print("duplicates!"); return
    parm = {'neighbors_total': 60, 'intra_neighbors_mutual': False,
            'neighbors_intra_fraction': .33, 'add_tree': False, "epsilon":-1,
                  'copy_lsa_neighbors': False, 'horizonCutoff':0, # this hz cutoff is what we look at
            'inter_outlier_threshold': -1, 'distance_metric':'euclidean', 'standardize':0,
            'inter_outlier_probabilistic_removal': False}


    bestparm = {'neighbors_total': 27,
         'neighbors_intra_fraction': 0.2204608669516461,
         'inter_outlier_threshold': 0.7495085614413425,
         'inter_outlier_probabilistic_removal': 0,
         'intra_neighbors_mutual': 0,
         'copy_lsa_neighbors': 1,
         'add_tree': 0,
         'horizonCutoff': 60,
         'distance_metric': 'euclidean',
         'standardize': 0}
    # parm = {'add_tree': True, 'copy_lsa_neighbors': False, 'inter_outlier_probabilistic_removal': False,  'inter_outlier_threshold': 0.72, 'intra_neighbors_mutual': False, 'neighbors_intra_fraction': 0, 'neighbors_total': 1}
    # grap = scalp.mkgraph(dataset,**bestparm)
    # stair =  sim.make_stairs(3,[0,1])
    # grap = scalp.graph.integrate(dataset, k=10, dataset_adjacency=stair, ls=False)

    # hub1_algo  hub1_k  hub2_algo  hub2_k   k  outlier_threshold  config_id     score       time
    #   0       9          3       9          19           0.970536    30  2.188789  15.822385
    # 0      10          3       6  15
    # grap = scalp.graph.integrate(dataset,hub1_algo = 0, hub1_k = 10,  hub2_algo=3, hub2_k=6,  k=15,  dataset_adjacency=None, outlier_threshold=ot)




    grap = scalp.graph.integrate(dataset,hub1_algo = 2, hub1_k = 12,  hub2_algo=2, hub2_k=12,  k=12,  dataset_adjacency=None, outlier_threshold=ot)
    grap = grap != 0


    # graph= csr_matrix(grap)
    #  there are many options to expose and some hubness correctors to be implemented
    # grap = grap.astype(bool)
    # neggraph = sgraph.negstuff(dataset,**parm)
    # scalp.umapwrap.graph_jax(dataset,(csr_matrix(grap), csr_matrix(neggraph)),label = 'umap', n_components = 2)


    if  False:
        dataset.obsp['connectivities'] = grap
        dataset.obsp['distances'] = grap
        dataset.uns['neighbors'] =  {'params': {'n_neighbors': 15,
      'method': 'umap',
      'metric': 'euclidean',
      'n_pcs': 40,
      #'bbknn': {'trim': 150, 'computation': 'annoy', 'batch_key': 'batch'},
      'use_rep': 'pca40'},
     'distances_key': 'distances',
     'connectivities_key': 'connectivities'}
        sc.tl.umap(dataset,n_components=2)
        grap = dataset.obsm.pop('X_umap')

    if False:
        grap = (grap.T+grap) != 0 # mk symmetric
        if find_duplicate_rows(grap): breakpoint()
        mapr = umap.UMAP()
        mapr.graph_ = grap
        grap = mapr.fit_transform(grap)

    # plt.matshow(grap.todense())
    # setup_grid(plt.gca(), dataset)
    # jplt.colorbar()
    # projection = umap.UMAP(n_neighbors = 10).fit_transform(grap)
    # projection = umap.UMAP(metric='precomputed', n_neighbors = 60).fit_transform(grap > 0)
    # projection  = scalp.umapwrap.umap_last_experiment(dataset, grap ,label = 'umap', n_components = dim)
    # projection = scalp.umapwrap.graph_pacmap2(False,csr_matrix(grap),label = 'umap', n_components = dim)
    # projection = scalp.umapwrap.graph_umap(False,grap,label = 'umap', n_components = dim) # THIS IS WHAT WE WANT TO USE I GUESS
    # print("umap done")
    dataset.obsm['scalp']= grap
    dataset.uns.setdefault('integrated',[])
    dataset.uns['integrated'].append('scalp')
    return dataset



# for e in ds: print(demo.find_duplicate_rows(e.X))
def find_duplicate_rows(mat):
    di = {}
    mat=csr_matrix(mat)
    for i, row in enumerate(mat):
        h = hash( tuple(row.indices))
        if h in di:
            print(i, di[h])
            return True
        di[h] = i
    return False

def do_the_data(ds):
    for d in ds:
        if d.uns['timeseries']:
            print(d.uns['name'])
            stack = Scalp(d)
            if stack:
                scalp.plot(stack,'scalp', color=['batch','label'])



# In[10]:


##################
##   RUN EXPERIMENTS
###############
import ubergauss.tools as ut
import functools
import time


#def run_all(datasets, scalpvalues = [.15, .25, .35, .45, .55, .65, .75, .85, .95]):
def run_all(datasets, scalpvalues = [.15,.2, .25, .3, .35, .45,  .55, .7, .9]):
    funcs = [scalp.mnn.harmony, scalp.mnn.scanorama, scalp.mnn.bbknnwrap, scalp.mnn.combat]
    for ot in scalpvalues:
        funcs.append( functools.partial(Scalp, ot=ot))
    fuid = Range(funcs)
    dataid = Range(datasets)
    tasks = [(f,d) for f in fuid for d in dataid]

    def run(fd):
        starttime = time.time()
        f,d = fd
        fun = funcs[f]
        dat = datasets[d]
        stack = fun(dat)
        return stack, time.time()-starttime

    mydata = ut.xxmap(run, tasks)
    mydata, runtimes = Transpose(mydata)

    fnames = 'Harmony Scanorama BBKNN ComBat'.split()
    fnames+=[f'Scalp: {s}' for s in scalpvalues]

    times = defaultdict(int)
    for (fi, di), t in zip(tasks, runtimes):
        times[fnames[fi]] += t
    # datasets_stack = Map(scalp.transform.stack, datasets)
    for (fu,da), result in zip(tasks, mydata):
        method = fnames[fu]

        # the method name is correct as i am having custom names for the methods
        # but we dont pass the label to runner, all the scalp runs are called scalp
        rmeth = method
        if "Scalp" in method:
            rmeth = 'scalp'
        try:
            datasets[da].obsm[method] = result.obsm[rmeth]
        except:
            breakpoint()
        datasets[da].uns.setdefault('methods', []).append(method)
        datasets[da].uns.setdefault('integrated', []).append(method)
    return datasets, fnames, times




def scalpscore(datasets):
    scr = lambda i: scalp.score.scalp_scores(datasets[i],projection = 'methods', label_batch_split=False)
    res  = ut.xxmap(scr, Range(datasets))
    return dict(zip(Range(datasets),res))



def scib_score(fanmes, datasets, saveas= 'scib.csv'):
    tasks = [(f,d) for f in fanmes for d in Range(datasets)]
    def f(item):
        fn,ds = item
        r = scalp.score.scib_scores(datasets[ds],fn)
        r.update({'method':fn})
        r.update({'dataset':ds})
        return r

    df =  pd.DataFrame(ut.xxmap(f,tasks))
    df.to_csv(saveas)
    return df



def mkscib_table(SCIB):

    def geomean_row(row):
        return np.sqrt(row['batch'] * row['label'])

    SCIB['geomean'] = SCIB.apply(geomean_row, axis=1)

    # rank within each group d
    SCIB[['a_rank','b_rank','geomean_rank']] = ( SCIB.groupby('dataset')[['batch','label','geomean']]
          .rank(method='average', ascending=False))

    # average ranks by c
    result = ( SCIB.groupby('method')[['a_rank','b_rank','geomean_rank']]
          .mean()
          .reset_index()
          .rename(columns={
              'a_rank': 'rank batch',
              'b_rank': 'rank label',
              'geomean_rank': 'rank geomean'
          }))

    return result.to_latex(index=False)

## In[ ]:


#######################
## SCORE INSTANCES
#######################
#scores = { str(i): scalp.score.scalp_scores(datasets[i],projection = 'methods', label_batch_split=False) for i in lmz.Range(datasets) }


## In[ ]:


#if False: # this is super slow
#    scores_kni = { str(i): scalp.score.kni_scores(datasets[i], projection='methods') for i in lmz.Range(datasets) }
#    dff = pd.DataFrame.from_dict(scores_kni, orient='index')
#    average_scores = dff.mean()
#    print(average_scores)
#'''
#scanorama      0.616067
#bbknn          0.646266
#combat         0.458397
#Scalp: 0.15    0.691557
#Scalp: 0.25    0.667454
#Scalp: 0.35    0.642752
#Scalp: 0.45    0.615580
#Scalp: 0.55    0.593257
#Scalp: 0.75    0.527588
#Scalp: 0.85    0.499085
#Scalp: 0.95    0.482638
#'''


## In[ ]:





## In[ ]:


#import pandas as pd
#import numpy as np
#from scipy.stats import gmean

#def compute_method_ranks(data, methods):
#    if not data:
#        return 0
#    rows = []
#    for ds, methods_dict in data.items():
#        for meth, scores in methods_dict.items():
#            if meth in methods:
#                label = scores['label_mean']
#                batch = scores['batch_mean']
#                geo = gmean([label, batch])
#                rows.append({'dataset': ds, 'method': meth,
#                             'label': label, 'batch': batch, 'geomean': geo})
#    df = pd.DataFrame(rows)

#    # Rank within each dataset (lower rank = better)
#    ranks = df.groupby('dataset')[['label', 'batch', 'geomean']].rank(ascending=False)
#    df[['label_rank', 'batch_rank', 'geo_rank']] = ranks

#    # Average ranks per method
#    avg_ranks = df.groupby('method')[['label_rank', 'batch_rank', 'geo_rank']].mean()
#    return avg_ranks

#base = fnames[:3]
#for e in fnames[3:]:
#    ffnames = base.copy()+[e]
#    zzzz = compute_method_ranks(scores, ffnames)
#    print(zzzz)

## all the methods
#print('########### ALL #######')
#print(compute_method_ranks(scores, fnames))
## ts = false
#print('############## batch #############')
#scores_tsf = [ str(i) for i,e in enumerate(datasets)
#                if not datasets[i].uns.get('timeseries',False) ]
#scores_tsf = {k:scores[k] for k in scores_tsf}
#print(compute_method_ranks(scores_tsf, fnames))
## ts = true
#print('########## time series ##########')
#scores_ts = [ str(i) for i,e in enumerate(datasets) if datasets[i].uns.get('timeseries',False)    ]
#scores_ts = {k:scores[k] for k in scores_ts}
#print(compute_method_ranks(scores_ts, fnames))


## In[ ]:


#scores


## In[ ]:


#ranktable, dom = scalp.output.score.pareto_avg(scores)
#ranktable


## In[ ]:


#dom


## In[ ]:


#import pandas as pd


## In[ ]:





## In[ ]:


#import pandas as pd
#scores2 = []
#for dsid, methdicts in scores.items():
#    for method,  metrics in methdicts.items():
#        #print(scores)
#        f = dict(metrics)
#        f['dataset'] = dsid
#        f['method'] = method
#        scores2.append(f)

#import seaborn as sns
#sc2 = pd.DataFrame(scores2)
#sc21 = sc2.pivot(index="method", columns="dataset", values="label_mean")
#sns.heatmap(sc21)



## In[ ]:

fixed_color_methods = {
    "Scanorama": "red",
    "BBKNN": "green",
    "ComBat": "purple",
    "Harmony": "blue",
}



def bluestar(scores2):
    """
    Generates and displays scatter plots of 'label_mean' vs 'batch_mean' scores
    from the provided DataFrame, showing individual dataset points and
    mean points with error bars.

    Args:
        scores2_df (pd.DataFrame): DataFrame containing 'method', 'label_mean',
                                  'batch_mean' (and optionally 'label_std', 'batch_std' for error bars).
        palette (dict): Dictionary mapping method names to colors for plotting.
    """
    # Filter out 'scalp' method if it exists (assuming it's a generic name and specific Scalp: 0.XX are preferred)
    scores2_df = pd.DataFrame(scores2)
    z = scores2_df[scores2_df.method != 'scalp'].copy()



    palette = {
            "Scanorama": "red",
        "BBKNN": "green",
        "ComBat": "purple",
        "Harmony" : "orange",

        # "Scalp: 0.15": "#f0f8ff",   # AliceBlue, almost white
        # "Scalp: 0.25": "#dceefb",  # Original very light blue
        # "Scalp: 0.35": "#a7cce5",  # Original lighter blue
        # "Scalp: 0.45": "#73b0da",  # Mid-point light blue
        # "Scalp: 0.55": "#4892c7",
        # "Scalp: 0.65": "#1f77b4",  # Original medium blue
        # "Scalp: 0.75": "#165a87",  # Original dark blue
        # "Scalp: 0.85": "#0f3e5a",  # Original dark blue
        # "Scalp: 0.95": "#0b2b40"    # Darkest blue

    }

    # Define methods that should appear at the end with specific colors
    # Get all unique methods present in the dataframe
    all_methods = sorted(z['method'].unique())

    # Separate fixed-color methods from others
    other_methods = [m for m in all_methods if m not in fixed_color_methods]

    # Create a Viridis colormap for the 'other_methods'
    viridis_cmap = plt.get_cmap('copper')
    # Generate colors, ensuring enough steps if many Scalp values exist
    num_other_methods = len(other_methods)
    viridis_colors = [viridis_cmap(i/max(1, num_other_methods -1)) for i in range(num_other_methods)]
    viridis_colors.reverse()

    # Sort Scalp methods if they are in 'other_methods' to ensure consistent color gradient
    scalp_other_methods = sorted([m for m in other_methods if "Scalp:" in m],
                                 key=lambda x: float(x.split(': ')[1]))
    non_scalp_other_methods = [m for m in other_methods if "Scalp:" not in m]

    # Combine them for consistent Viridis ordering (non-scalp first, then sorted scalp)
    ordered_other_methods = non_scalp_other_methods + scalp_other_methods

    # Map these ordered methods to viridis colors
    viridis_palette = {method: viridis_colors[i] for i, method in enumerate(ordered_other_methods)}


    # Combine the custom palette with the Viridis palette
    palette = {**viridis_palette, **fixed_color_methods}

    # Ensure the legend order shows fixed_color_methods last
    legend_order = ordered_other_methods + list(fixed_color_methods.keys())

    # Plot individual dataset points
    if False:# plot real data:
        plt.figure(figsize=(10, 8))
        ax = sns.scatterplot(data=z, x='label_mean', y='batch_mean', hue="method",
                             palette=palette, legend='full', hue_order=legend_order)
        ax.set_title('Individual Dataset Scores (Label vs Batch Mean)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    if False: # plot without decoration
        # Calculate mean scores per method
        z_avg = z.groupby("method", as_index=False)[["label_mean", "batch_mean"]].mean()

        # Plot mean scores per method
        plt.figure(figsize=(10, 8))
        ax = sns.scatterplot(data=z_avg, x='label_mean', y='batch_mean', hue="method",
                             palette=palette, legend='full', s=100, marker='o', hue_order=legend_order)
        ax.set_title('Mean Scores per Method (Label vs Batch Mean)')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # Calculate mean and standard deviation for error bars
    z_stats = z.groupby("method").agg(
        label_mean=("label_mean", "mean"),
        label_std=("label_mean", "std"),
        batch_mean=("batch_mean", "mean"),
        batch_std=("batch_mean", "std")
    ).reset_index()

    # Plot mean scores with error bars
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title('Mean Scores with Standard Deviation Error Bars')

    # Scatter plot with error bars for each method, respecting the legend_order
    for method_name in legend_order:
        row = z_stats[z_stats['method'] == method_name]
        if not row.empty:
            row = row.iloc[0]
            color = palette.get(method_name)

            ax.errorbar(
                x=row["label_mean"],
                y=row["batch_mean"],
                xerr=row["label_std"],
                yerr=row["batch_std"],
                fmt='o',  # format for the markers
                label=method_name,
                color=color,
                capsize=5,  # size of the caps on the error bars
                markersize=8
            )

    # Customize legend and labels
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    ax.set_xlabel("Label Mean Score")
    ax.set_ylabel("Batch Mean Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return

#     # Plot individual dataset points
#     plt.figure(figsize=(10, 8))
#     ax = sns.scatterplot(data=z, x='label_mean', y='batch_mean', hue="method", palette=palette, legend=True)
#     ax.set_title('Individual Dataset Scores (Label vs Batch Mean)')
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.show()

#     # Calculate mean scores per method
#     z_avg = z.groupby("method", as_index=False)[["label_mean", "batch_mean"]].mean()

#     # Plot mean scores per method
#     plt.figure(figsize=(10, 8))
#     ax = sns.scatterplot(data=z_avg, x='label_mean', y='batch_mean', hue="method", palette=palette, legend=True, s=100, marker='o')
#     ax.set_title('Mean Scores per Method (Label vs Batch Mean)')
#     ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.show()

#     # Calculate mean and standard deviation for error bars
#     z_stats = z.groupby("method").agg(
#         label_mean=("label_mean", "mean"),
#         label_std=("label_mean", "std"),
#         batch_mean=("batch_mean", "mean"),
#         batch_std=("batch_mean", "std")
#     ).reset_index()

#     # Plot mean scores with error bars
#     fig, ax = plt.subplots(figsize=(10, 8))
#     ax.set_title('Mean Scores with Standard Deviation Error Bars')

#     # Scatter plot with error bars for each method
#     for _, row in z_stats.iterrows():
#         method_name = row["method"]
#         color = palette.get(method_name)

#         ax.errorbar(
#             x=row["label_mean"],
#             y=row["batch_mean"],
#             xerr=row["label_std"],
#             yerr=row["batch_std"],
#             fmt='o',  # format for the markers
#             label=method_name,
#             color=color,
#             capsize=5,  # size of the caps on the error bars
#             markersize=8
#         )

#     # Customize legend and labels
#     ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
#     ax.set_xlabel("Label Mean Score")
#     ax.set_ylabel("Batch Mean Score")
#     plt.grid(True, linestyle='--', alpha=0.6)
#     plt.tight_layout()
#     plt.show()

'''

palette = {
        "scanorama": "red",
    "bbknn": "green",
    "combat": "purple",
    "Scalp: 0.15": "#0b2b40",
    "Scalp: 0.25": "#0f3e5a",  # Original dark blue
    "Scalp: 0.35": "#165a87",  # Original dark blue
    "Scalp: 0.45": "#1f77b4",  # Original medium blue
    "Scalp: 0.55": "#4892c7",
    "Scalp: 0.65": "#73b0da",  # Mid-point light blue
    "Scalp: 0.75": "#a7cce5",  # Original lighter blue
    "Scalp: 0.85": "#dceefb",  # Original very light blue
    "Scalp: 0.95": "#f0f8ff"   # AliceBlue, almost white
}
z = pd.DataFrame(scores2)
z = z[z.method != 'scalp']

ax = sns.scatterplot(z,x='label_mean', y= 'batch_mean', hue = "method",palette = palette, legend=True)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()
z_avg = z.groupby("method", as_index=False)[["label_mean", "batch_mean"]].mean()
ax = sns.scatterplot(z_avg,x='label_mean', y= 'batch_mean', hue = "method",palette = palette, legend=True)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()

z_stats = z.groupby("method").agg({
    "label_mean": ["mean", "std"],
    "batch_mean": ["mean", "std"]
}).reset_index()

# Flatten MultiIndex columns
z_stats.columns = ["method", "label_mean", "label_std", "batch_mean", "batch_std"]

# Step 2: Plot with error bars
fig, ax = plt.subplots()

# Scatter plot with error bars
for _, row in z_stats.iterrows():
    ax.errorbar(
        x=row["label_mean"],
        y=row["batch_mean"],
        xerr=row["label_std"],
        yerr=row["batch_std"],
        fmt='o',
        label=row["method"],
        color=palette[row["method"]] if row["method"] in palette else None
    )

# Customize legend
ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
ax.set_xlabel("label_mean")
ax.set_ylabel("batch_mean")

'''

## In[ ]:


#sc22 = sc2.pivot(index="method", columns="dataset", values="batch_mean")
#sns.heatmap(sc22)




#geomean = np.sqrt(sc21*sc22)
#sns.heatmap(geomean)


## In[ ]:


#geomean.index.name = 'method'
#df_melted = geomean.reset_index().melt(id_vars='method', var_name='dataset', value_name='score')


##df_melted['group'] = np.where(df_melted.dataset.astype(int) < 13 , 'timeseries', 'batch')
#df_melted['group'] = [ 'timeseries' if datasets[int(i)].uns['timeseries'] else 'batch'  for i in  df_melted.dataset] # np.where(df_melted.dataset.astype(int) < 13 , 'timeseries', 'batch')
#df_melted['size'] = [ len(np.unique(datasets[int(i)].obs['batch']) ) for i in  df_melted.dataset] # np.where(df_melted.dataset.astype(int) < 13 , 'timeseries', 'batch')
## df_melted['group']=df_melted.dataset.astype(int) % 16




#g = sns.catplot( data=df_melted, x="method", y="score", hue = 'group')# native_scale=True, zorder=1 )
## g = sns.catplot( data=df_melted, x="method", y="score", hue = 'dataset')# native_scale=True, zorder=1 )
##

## Calculate mean per category
#means = df_melted.groupby("method")["score"].mean()

## Add markers for each category's mean
#for i, day in enumerate(means.index):
#    plt.scatter(i, means[day], color='black', marker='x', s=50, label='Mean' if i == 0 else "")

#g.set_xticklabels(rotation=45, labels = np.unique(df_melted.method)) # ha='right' is useful here too
#plt.show()
#g = sns.boxplot( data=df_melted, x="method", y="score", hue ="group")# native_scale=True, zorder=1 )
#g.set_xticklabels(rotation=45, labels = np.unique(df_melted.method)) # ha='right' is useful here too
#plt.show()


import matplotlib.transforms as mtrans




def barplot(geomean, datasets):

    geomean.index.name = 'method'
    df_melted = geomean.reset_index().melt(id_vars='method', var_name='dataset', value_name='score')
    #df_melted['group'] = np.where(df_melted.dataset.astype(int) < 13 , 'timeseries', 'batch')
    df_melted['group'] = [ 'timeseries' if datasets[int(i)].uns['timeseries'] else 'batch'  for i in  df_melted.dataset] # np.where(df_melted.dataset.astype(int) < 13 , 'timeseries', 'batch')
    df_melted['size'] = [ len(np.unique(datasets[int(i)].obs['batch']) ) for i in  df_melted.dataset] # np.where(df_melted.dataset.astype(int) < 13 , 'timeseries', 'batch')
    # df_melted['group']=df_melted.dataset.astype(int) % 16

    if False: # old plot
        g = sns.catplot( data=df_melted, x="method", y="score", hue = 'group')# native_scale=True, zorder=1 )
        # g = sns.catplot( data=df_melted, x="method", y="score", hue = 'dataset')# native_scale=True, zorder=1 )
        #
        # Calculate mean per category
        means = df_melted.groupby("method")["score"].mean()
        # Add markers for each category's mean
        for i, day in enumerate(means.index):
            plt.scatter(i, means[day], color='black', marker='x', s=50, label='Mean' if i == 0 else "")
        g.set_xticklabels(rotation=45, labels = np.unique(df_melted.method)) # ha='right' is useful here too
        plt.show()

    palette = fixed_color_methods

    # custom colors like this:
    # group_palette = {"timeseries": sns.color_palette("Blues")[2], "batch": sns.color_palette("Blues")[0]}

    order = sorted([m for m in df_melted['method'].unique() if 'Scalp' in m], key=lambda x: float(x.split(': ')[1])) + sorted([m for m in df_melted['method'].unique() if 'Scalp' not in m])
    g = sns.boxplot( data=df_melted, x="method", y="score", hue ="group", palette= "Blues", order = order)

    # Define a custom palette for the 'group' hue, e.g., using blues

    g.set_xticklabels(rotation=45, labels = order, ha = 'right', x= 4)
    #g.set_xticklabels(rotation=45, labels = np.unique(df_melted.method), ha = 'right', x= 4)

    for i, label in enumerate(g.get_xticklabels()):
        label.set_x(i+ 0.6)

    # Apply a translation to all x-tick labels
    offset = mtrans.ScaledTranslation(.1, .90, g.figure.dpi_scale_trans)
    for label in g.get_xticklabels():
        label.set_transform(g.transData + offset)
        #g.set_xticklabels(rotation=45, labels=np.unique(df_melted.method), ha='right')

    # colors = [palette.get(label, 'gray') for label in np.unique(df_melted.method)]
    # [t.set_color(i) for (i,t) in zip(colors,g.get_xticklabels())]
    #print(g.get_xticklabels()[0].__dict__)
    [i.set_color( palette.get(i._text,'gray')) for  i in g.get_xticklabels()]

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_axisbelow(True)

    legend = plt.legend(bbox_to_anchor=(.2, .25), loc='upper left', borderaxespad=0.)

    plt.show()



#pd.set_option('display.max_rows', 1000)
#df_melted

#groupby = 'group' # 'dataset'
#groupby = 'dataset' # 'dataset'
#df_melted  = df_melted[df_melted.group=='batch']
#avg_rank = df_melted.groupby('dataset').score.rank(pct=True).groupby(df_melted.method).mean()
## avg_rank = df_melted.groupby(groupby).score.rank(pct=True).groupby(df_melted.method).mean()
#import scikit_posthocs as sp
#df_melted['block_id_col'] = range(len(df_melted))
#test_results = sp.posthoc_conover_friedman(
#    df_melted,
#    melted=True,
#    block_col=groupby,
#    block_id_col='block_id_col',
#    group_col='method',
#    y_col='score',
#)
#plt.figure(figsize=(10, 2), dpi=100)
#plt.title('Critical difference diagram of geometric mean')
#sp.critical_difference_diagram(avg_rank, test_results)



#grp = 'dataset'

##dom['group']=  dom.dataset.astype(int) % 16
#avg_rank = dom.groupby(grp).domcount.rank(pct=True).groupby(dom.method).mean()
#import scikit_posthocs as sp
#dom['block_id_col'] = range(len(dom))
#test_results = sp.posthoc_conover_friedman(
#    dom,
#    melted=True,
#    block_col=grp,
#    block_id_col='block_id_col',
#    group_col='method',
#    y_col='domcount',
#)
##sp.sign_plot(test_results)
#plt.figure(figsize=(10, 2), dpi=100)
#plt.title('Critical difference diagram of pareto dominance')
#sp.critical_difference_diagram(avg_rank, test_results)


## # more eval metrics, we stick to the ones above

## In[ ]:


#from scalp.output.score import score_lin, score_lin_batch, score_scib_metrics
#scoredics_lb = ut.xxmap(score_lin_batch, mydata)
#scoredics_scib = ut.xxmap(score_scib_metrics, mydata)
#scoredics_l = ut.xxmap(score_lin, mydata)

## score_lin_batch and score_lin -> pareto comparison


## In[ ]:


#import pandas as pd
#import seaborn as sns
## this only applies if we dont use the scib score fucntion
#funcs = 'scanorama, umaponly, bbknn, combat, Scalp'.split(', ')

#results = [ {"method":funcs[f], 'score':s, 'dataset':d, 'target':'label' } for s,(f,d) in zip(scoredics_l,tasks)]
#results += [ {"method":funcs[f], 'score':s, 'dataset':d, 'target':'batch'} for s,(f,d) in zip(scoredics_lb,tasks)]
#df = pd.DataFrame(results)
#sns.barplot(data=df, y = 'score', x = 'method', errorbar = 'sd', hue='target')
#plt.show()
#ours = df.pivot_table(index='method', columns='target', values='score')


## In[ ]:


#import lmz
## this is for SCIB scoring


## turn scores into a dataframe
#funcs = 'scanorama, umaponly, bbknn, combat, Scalp'.split(', ')
#results = [ [{"method":funcs[f], 'score':ss, 'dataset':d, 'metric':scrmeth }
#             for (scrmeth,ss) in s.items() ]for s,(f,d) in zip(scoredics_scib,tasks)]
#results = lmz.Flatten(results)
#df = pd.DataFrame(results)



#sns.barplot(data=df, y = 'score', x = 'method', errorbar = 'sd', hue = 'metric')
#plt.legend(loc='right', bbox_to_anchor=(1.85, 0.5), ncol=1)
#plt.show()


## In[ ]:


#def split_scib_scores(dicts):
#    '''splits scores in batch and label scores'''
#    batchwords = 'PCR_batch ASW_label/batch graph_conn'.split()

#    def split(d):
#        b = np.mean([v for k,v in d.items() if k in batchwords ])
#        a = np.mean([v for k,v in d.items() if not k in batchwords ])
#        return a,b

#    scores = lmz.Map(split, dicts)
#    return lmz.Transpose(scores)


#scr_l, scr_b = split_scib_scores(scoredics_scib)
#results = [ {"method":funcs[f], 'score':s, 'dataset':d, 'target':'bioconservation_scib_avg' } for s,(f,d) in zip(scr_l,tasks)]
#results += [ {"method":funcs[f], 'score':s, 'dataset':d, 'target':'batch_scib_avg'} for s,(f,d) in zip(scr_b,tasks)]
#df = pd.DataFrame(results)
#sns.barplot(data=df, y = 'score', x = 'method', errorbar = 'sd', hue='target')
#plt.show()
#theirs = df.pivot_table(index='method', columns='target', values='score')


## In[ ]:


#our_score = [ {"method":funcs[f], 'score':s, 'dataset':d, 'target':'label' } for s,(f,d) in zip(scoredics_l,tasks) ]
#our_score += [ {"method":funcs[f], 'score':s, 'dataset':d, 'target':'batch'} for s,(f,d) in zip(scoredics_lb,tasks)  ]
#df2 = pd.DataFrame(our_score)
#df2

#from ubergauss.optimization import pareto_scores
#pareto_scores(df)
#pareto_scores(df2) # this is pancreatic only  btw



#np.corrcoef(ours.batch.values, theirs.batch_scib_avg.values)[0,1], np.corrcoef(ours.label.values, theirs.bioconservation_scib_avg.values)[0,1]

#avg_rank = dom.groupby('dataset').domcount.rank(pct=True).groupby(dom.method).mean()
#import scikit_posthocs as sp
#test_results = sp.posthoc_conover_friedman(
#    dom,
#    melted=True,
#    block_col='dataset',
#    group_col='method',
#    y_col='domcount',
#)
##sp.sign_plot(test_results)
#plt.figure(figsize=(10, 2), dpi=100)
#plt.title('Critical difference diagram of average score ranks')
#sp.critical_difference_diagram(avg_rank, test_results)


## In[ ]:


#d={4:99}
#d.setdefault(4,5)
#print(d)


## In[ ]:


## horizoncut test
#d = np.random.rand(10,10)
#co = 2
#print(np.partition(d, co, axis=1)[:, co])
#print(np.sort(d,axis=1))


## In[ ]:


#from scipy.sparse import csr_matrix
#import numpy as np
#random_matrix = np.random.rand(10, 10)
#mask = np.random.rand(10, 10) > 0.5
#testmat = random_matrix * mask

#testmat


## In[ ]:


#n_samples = 10
##S = np.random.rand(n_samples, n_samples)
#np.fill_diagonal(S, 0)  # self-similarity = 1

#plt.matshow(S)
#plt.colorbar()

## Apply CSLS

#def csls_distance_matrix(D, k=10):
#    """
#    Applies CSLS hubness reduction to a distance matrix.

#    Args:
#        D: (n_samples, n_samples) distance matrix (lower = more similar)
#        k: Number of nearest neighbors to consider for local scaling

#    Returns:
#        D_csls: Hubness-reduced distance matrix
#    """
#    n = D.shape[0]

#    # Find k-nearest neighbors for each point (excluding self)
#    knn = np.argpartition(D, k+1, axis=1)[:, :k+1]  # +1 to account for self
#    knn = np.array([row[row != i][:k]for i, row in enumerate(knn)])  # remove self

#    # Compute mean distance of each point's neighborhood r(x_i)
#    r = np.array([D[i, knn[i]].mean() for i in range(n)])

#    print(r)

#    # Symmetric CSLS adjustment (note the + instead of - for distances)
#    D_csls = 2 * D + r[:, None] + r[None, :]

#    return D_csls

#plt.show()
#plt.matshow(csls_distance_matrix(S,3))
#plt.colorbar()


## In[ ]:


#array = np.array
#z = [array([405,  61, 220, 231, 145,  58,  81, 442, 466, 281, 199]), array([110, 453, 224, 115, 405,  61, 134, 160, 345, 218, 231]), array([ 61, 164, 453, 220, 231, 307, 466, 281, 110, 145, 442]), array([112, 405, 110, 307, 453, 115, 469, 224, 289,  81,  28]), array([115, 405, 383, 466,  30, 190, 134, 453, 307, 305, 442]), array([220, 453, 218, 405, 268, 222, 164, 115,  58, 307, 110]), array([281, 110, 442, 134, 453, 466, 307,  61, 115, 231, 141]), array([138, 298, 110, 453, 268, 115, 209,  61, 466, 442, 281]), array([466, 268, 405, 112, 115,  81, 110, 453, 220, 231,  61]), array([160, 307, 115, 466, 442, 405, 164,  81, 190, 110, 498]), array([164, 218, 110, 307, 133,  81, 442, 405, 160,  23,  61]), array([268, 220, 466, 229,  85,  61, 231, 453, 102, 442, 164]), array([110, 218, 405, 442, 468, 281, 220, 115, 271, 453, 231]), array([281, 405, 209, 218, 466, 271,  28, 115, 453, 468, 268]), array([453, 281, 134,  61,  81, 160, 110, 231, 466, 442, 115]), array([218, 307, 405, 110,  81, 115, 134, 453, 133, 442,  61]), array([115,  61, 468, 466, 442, 405, 281, 453, 110, 218, 231]), array([ 23, 164, 466, 160,  81, 218, 133, 405, 442, 307, 268]), array([405, 307, 453,  58,  61,  81, 220, 218, 209, 164, 133]), array([209, 110, 281,  61, 442, 466, 453, 218, 231, 298, 405]), array([281, 453, 307, 115, 209, 110, 298, 100, 134,  61, 444]), array([164, 405, 268, 442, 110, 184, 307, 133, 130, 453, 231]), array([115, 453, 405, 468, 110, 307, 218, 134, 305, 281, 231]), array([453, 110, 115, 468, 224, 405,  61, 307, 305, 134, 281]), array([110, 339, 231, 115,  81, 271, 112, 307, 116, 442, 160]), array([100,  61, 468, 134, 281, 160, 224, 110, 453, 444, 218]), array([307, 453, 405, 222, 220, 218, 466, 164, 110, 231,  58]), array([ 61, 218, 209,  81, 453, 307, 405, 110, 160, 134, 281]), array([ 61, 115, 453, 405, 110,  69, 468, 100, 442, 231, 220]), array([110, 466, 453,  81, 160, 400, 231,  61, 307, 134, 115]), array([218, 453, 134, 405,  61, 209, 466, 307, 110,  81, 220]), array([112, 442, 453, 164, 281, 209, 229,  61, 110, 307,  81]), array([156, 110, 281,  81, 442,  89, 305,  61, 453, 231, 405]), array([453,  61, 100, 110, 115, 231, 468, 209, 444, 220, 293]), array([115,  61, 453, 160, 110, 220, 218, 442, 134,  81, 405]), array([307, 190, 396, 164,  81, 434, 115, 110, 231, 466, 268]), array([466, 298, 110, 209,  61, 115, 405, 281, 138, 231, 134]), array([453, 209, 134, 268, 160, 110, 442, 307, 231,  61, 220]), array([ 28, 115, 110, 466,  61, 453, 281, 335, 405,  81, 307]), array([ 61, 466, 453, 110, 115, 164, 400, 134,  85, 268, 190]), array([483, 218,  87, 453,  58, 417, 220, 405,  81, 164, 231]), array([396, 100, 218,  61, 281, 282, 110, 453, 115, 444, 434]), array([442, 444, 138, 110,  81, 100, 281, 453, 134,  61, 115]), array([281, 298, 220, 231,  61, 453, 466, 110, 307, 389, 415]), array([110, 100, 268, 281, 453, 115, 442, 307,  61, 220, 112]), array([110, 442,  61, 307, 231, 453, 160, 209, 115, 400, 134]), array([209,  61, 160, 307, 164, 453, 110, 231, 405, 442, 220]), array([466, 268, 307, 442, 216, 453, 293, 405, 112,  28,  61]), array([115, 307, 383, 339, 498, 110, 231, 434,  61,  52, 190]), array([405,  23, 231, 164, 110, 442,  81, 261, 453, 220, 145]), array([134, 307, 110, 442, 220,  61, 405, 218, 209, 115, 224]), array([209, 453,  61, 110, 134, 218, 298, 115, 231, 281, 405]), array([307, 218, 115, 405,  81, 190,  28, 110, 281, 453, 498]), array([220, 164, 231,  85, 145,  61, 253, 442, 453, 345, 466]), array([453, 110, 134, 466,  61, 220, 229, 160, 400, 442, 231]), array([405,  81, 307,  28, 466, 110,  61, 112, 115, 268, 453]), array([281, 444, 115, 110,  61, 453, 209, 405, 442, 466, 100]), array([231, 110, 466,  61, 405,  81, 442, 281, 220, 216, 293]), array([470, 110, 453,  61, 307,  81, 405, 466, 281, 134, 115]), array([231, 442, 281, 453,  61, 466, 110, 218, 134, 160, 115]), array([307, 400, 160, 442, 115, 110, 190, 453, 218,  81, 466]), array([466, 281, 307, 453, 110, 115, 218, 209, 271, 405, 134]), array([ 52, 110, 307, 115, 268, 442, 190, 128, 231,  81, 466]), array([218, 231, 164, 220,  58, 466, 453, 307,  61, 442, 417]), array([442, 405, 160, 231,  61, 307, 466, 134, 453, 110, 133]), array([ 61, 218, 453, 405, 164, 268, 307, 110, 220, 442, 231]), array([405, 307, 110, 453, 134, 115, 396, 141, 442, 218, 209]), array([268, 307,  28, 115, 112, 466, 442,  61, 453, 231,  81]), array([ 81, 453, 442, 218,  61, 110, 307, 396, 115, 209, 405]), array([434,  81, 307, 115, 383, 453,  28,  52, 190, 498, 268]), array([281, 110,  61, 231, 405, 218, 220, 453, 466, 442, 271]), array([164, 231, 209,  85, 405, 442, 466, 229, 453,  61, 145]), array([466, 110, 442,  69, 405, 307, 281, 218,  61, 115, 209]), array([160, 134, 115, 110,  81, 453,  61, 466, 442, 307, 231]), array([453,  61, 231, 442, 307, 220, 281, 100, 110, 444, 209]), array([453, 110, 130, 307, 466, 184, 442, 268, 164, 220, 405]), array([405, 115, 110, 209, 100, 466,  61, 444, 442, 281, 268]), array([160, 209, 281,  81, 218,  28, 442, 110, 115,  61, 307]), array([400, 307, 115, 434, 112,  61,  81, 190, 396, 164, 466]), array([110, 281, 115, 468, 405, 453,  69, 100, 209, 307, 218]), array([164, 253,  81, 231, 145, 442,  85,  61, 405, 220, 110]), array([466, 307, 110,  28, 453, 206, 268, 115, 305, 218,  61]), array([110, 218, 405, 281, 271, 307, 442, 115, 184, 231, 134]), array([466, 231,  81, 405, 442, 110, 453, 307, 305, 220, 281]), array([400, 405, 466, 115, 160,  61, 453, 209, 134, 110, 442]), array([453, 281, 218,  81, 298, 209, 442, 110, 231, 405, 220]), array([ 61, 110, 444, 100, 405, 442, 453, 307, 115, 209,  81]), array([466, 110, 282,  81, 453,  97, 100, 444, 442, 405, 115]), array([307, 110,  81, 220, 268, 293, 141, 231, 216, 442, 261]), array([115,  81, 442, 110,  61, 307, 190, 400, 453, 160, 405]), array([115, 405, 281, 224, 307, 453, 110, 442, 134, 271, 209]), array([307, 160, 110, 442, 115, 453, 405, 281,  61, 128, 466]), array([268,  28, 466, 307, 115, 110, 453, 206, 218, 133, 112]), array([442, 110, 220, 261, 141, 466, 281, 268, 453, 395,  81]), array([405, 115, 434, 112,  85, 110,  61, 268,  81, 160, 453]), array([442, 216, 293, 268,  81, 220, 453, 231, 110, 395, 405]), array([ 61, 115, 100, 209, 442,  81, 110, 453, 468, 444, 396]), array([466,  61, 231,  81, 420, 442, 110, 281, 164, 102,  28]), array([281, 453, 220, 466,  61, 442, 218, 110, 115, 231, 405]), array([115,  61, 405, 453, 468, 281,  69, 110, 218, 209, 466]), array([110,  28,  81, 307, 134, 405, 453, 442, 133, 470, 231]), array([442, 453,  69,  61, 220, 110, 444, 115, 100, 405, 466]), array([453,  61, 110, 442, 405,  28,  81, 160, 209, 444, 307]), array([339, 220, 231, 268, 190, 112, 494, 307, 453, 383, 115]), array([110, 307, 281, 405,  61, 115, 442, 218, 268, 209, 231]), array([ 52, 164, 434, 453, 400,  85, 231,  61, 442, 212,  23]), array([307,  28, 268,  81, 112, 442, 466, 305,  61, 231, 110]), array([466,  81, 209,  61, 134, 444, 453, 115, 110, 281, 160]), array([ 81, 115, 133, 498, 116, 307, 164, 190, 339,  28, 160]), array([405, 218,  81, 268, 231, 466, 307, 453, 115, 281, 110]), array([453, 405, 218,  81, 281, 115, 209, 466, 307, 231]), array([110, 307, 405, 453, 134, 442,  81, 133, 468, 218, 115]), array([442, 218, 453, 164, 307, 466, 220, 190, 405,  81, 268]), array([110, 444, 453, 115, 466,  61, 100, 442, 134, 209, 298]), array([420, 307, 442, 466, 231, 115,  81, 110, 160,  61, 209]), array([453, 209, 442, 405, 110,  61, 468, 218, 298, 268, 271]), array([115, 453,  81, 218, 307, 281,  61, 305, 110, 405, 134]), array([417, 164, 453, 218, 220, 405, 307,  61, 222,  58, 110]), array([453, 209, 110, 466,  61, 281, 405, 160, 231, 134,  85]), array([110, 405, 453, 209, 307,  81, 281, 115, 218, 466, 133]), array([405, 209, 218, 281,  61, 110, 468, 100,  69, 453, 134]), array([218, 453,  81, 405, 110, 115, 281, 307, 442, 134, 468]), array([442, 293, 216, 220, 231, 160,  61,  23,  81, 212, 164]), array([ 81, 442, 466, 110,  23, 268, 231, 453, 281, 117, 307]), array([ 81,  61, 268, 231, 216, 293, 220, 110, 442, 453, 281]), array([184, 453, 199, 442, 307, 405, 218, 281, 164, 305, 209]), array([293, 434,  81, 216, 220, 231, 115, 453, 218, 110, 405]), array([405,  58,  87, 417, 442, 453,  81, 220, 218, 231,  61]), array([115, 481, 218, 453, 110,  61, 307, 209, 396, 405, 442]), array([209, 307, 110, 115, 218,  61, 224,  81, 453, 442, 405]), array([160, 400, 307, 110, 442, 383, 434, 453,  61, 134, 466]), array([218, 134, 231,  81, 307, 442, 405,  28, 110, 453, 133]), array([231, 396, 453,  28,  81, 164, 110, 405, 442, 293, 216]), array([384,  81, 307, 190, 339, 160, 115, 218, 224, 498, 405]), array([218, 417, 466, 405, 110,  58, 453, 483, 220,  81, 231]), array([110, 218, 115, 281, 453, 442,  81, 466, 307,  61, 405]), array([307, 268, 115,  28, 110, 453, 134, 405, 218, 442, 206]), array([405,  61, 115, 268, 110, 444, 160, 453, 442, 134, 293]), array([395,  81, 442, 110, 453, 268, 307, 115, 216, 293, 231]), array([442, 453, 115, 218, 229, 110, 134, 220,  85, 231,  61]), array([277, 442,  81, 395, 293, 141, 231, 110, 216, 268, 466]), array([ 69,  81, 110, 115, 281, 209, 468, 405, 224, 218,  61]), array([453, 224, 405, 307, 134, 218, 110,  61, 115,  81, 209]), array([231, 190,  81, 307,  28, 115, 442, 133, 110, 383, 405]), array([231, 307, 115, 128, 442, 141, 281, 268, 110, 405,  81]), array([231,  81, 220, 453,  61, 268, 115, 218, 110, 442, 209]), array([453, 442, 160, 307, 110, 396, 115, 134, 405, 224, 164]), array([405, 164, 442, 231, 453, 134, 298, 110, 307, 115, 468]), array([ 81,  28, 466, 307, 218, 116, 115, 110,  61, 268, 405]), array([ 81, 231,  85, 110, 466, 442, 453, 405,  61,  28, 218]), array([442, 281, 110,  61,  69, 453, 468, 307, 231, 405, 268]), array([307,  81, 115,  28, 190, 164, 405, 281, 231, 133, 160]), array([405, 110, 218, 268, 453, 164, 307,  81,  61, 220, 133]), array([220, 453, 417, 405,  58,  87, 218,  61, 231, 483, 442]), array([281, 110, 405, 453,  85, 115, 100, 442, 466, 218, 231]), array([164, 405, 218, 307, 110, 133, 442, 466, 268, 453, 134]), array([307, 110,  81, 115, 405, 209, 220, 442, 466, 453, 396]), array([442,  61, 110, 307, 220, 453, 405, 209, 231, 298,  58]), array([298, 218, 110, 115,  61, 134, 405, 453,  81, 442,  85]), array([160, 453, 115, 442,  81,  61, 110, 307, 268, 134, 231]), array([ 81, 216, 220, 293, 231, 268, 442, 115, 110,  28, 396]), array([442, 405, 110, 453, 281, 100, 115, 209,  61, 134, 231]), array([281,  61,  81, 110, 466, 134, 453, 220, 209, 160, 115]), array([453, 134, 160,  61, 110, 345, 100, 218, 466, 112, 442]), array([110, 453, 442, 115,  61, 209, 307, 405, 100,  28, 281]), array([ 58,  87, 405, 218, 220, 231, 453,  81, 466,  61, 417]), array([281, 220, 453, 231,  61, 466, 442, 110,  85, 268, 218]), array([ 81, 268, 134, 307, 231,  61, 442, 110, 466, 115, 190]), array([218, 224, 466, 453,  69,  81, 281, 110, 405, 115, 468]), array([405, 218, 110, 112, 442, 466, 115,  81, 453,  61, 307]), array([100,  61, 115, 405, 231, 110, 468, 453, 444, 307, 160]), array([231,  23, 164, 453, 212, 442, 216, 293,  61, 161,  81]), array([383, 298, 268, 453, 141, 466, 190, 307, 231, 112, 110]), array([220, 442,  81, 293, 216,  23, 395, 466, 268, 231, 110]), array([164, 218,  58,  61, 220, 405, 453, 307, 442, 110,  81]), array([115, 128, 190,  81, 307,  52, 339, 405, 268, 442, 110]), array([468, 138, 281,  81, 110, 218, 405, 307, 224, 209, 115]), array([466, 420, 453, 442, 281, 110,  61, 231, 496, 220,  81]), array([281,  81, 110, 453, 466, 405, 218, 209, 231, 160, 115]), array([145, 231,  61,  81, 164,  85, 420, 253, 405,  57, 442]), array([307, 115, 110, 442, 160, 400, 190, 466,  61, 453,  81]), array([405, 110, 442, 115, 444, 453, 281,  69, 209, 468, 307]), array([405, 271, 453, 160, 307, 442, 110, 115, 466, 231, 218]), array([444, 453, 100, 110, 115, 442, 405,  61, 307, 206, 434]), array([218, 453, 307, 229, 231, 271, 405, 164, 442, 110, 281]), array([110, 307,  81, 442, 268, 133, 164, 115, 218, 222, 405]), array([453, 307, 164, 190,  81, 434, 396, 281, 115, 220,  28]), array([444, 110, 453,  61, 405, 281, 231, 442, 218, 115, 220]), array([134,  61, 100, 444, 453, 110, 115, 281, 307, 442, 218]), array([307, 268,  81, 405,  28, 442, 466, 218, 231, 133, 110]), array([231,  61,  85, 466, 453, 442, 220, 405,  81, 281, 110]), array([442,  23,  81, 164, 438, 110, 115, 307, 293, 405, 216]), array([216,  81, 261,  23, 164, 231, 293, 442, 453,  61, 405]), array([442, 218, 220,  58, 405, 199, 417,  81,  87, 453, 483]), array([110, 442, 220, 453,  61, 405, 400, 231, 100, 444, 115]), array([160, 442,  81, 110, 115, 438, 405, 307, 218,  28, 112]), array([466, 307,  28, 268, 442, 110, 115, 453,  81, 112,  61]), array([110, 453, 405, 281, 115, 218, 224, 468,  61, 100,  69]), array([395, 442, 466, 231, 220, 307, 115,  81, 134, 110, 281]), array([268, 231,  87, 220,  61, 453, 145, 222, 115, 400, 218]), array([281, 134, 453, 110, 115, 307,  89, 405, 218, 442, 268]), array([405, 307, 110, 442, 453, 218, 115,  81, 134, 281, 209]), array([405, 307, 218, 134, 110, 220, 442, 453, 466, 281, 115]), array([134, 281, 494, 434, 160, 268, 307, 453, 115, 159,  61]), array([453, 307, 110, 115, 442,  61, 134, 209, 305, 405,  85]), array([281, 218, 100, 115,  69, 468, 110, 405,  61, 453, 444]), array([453, 160, 115, 110, 134, 442, 444,  61, 282, 405, 100]), array([ 28, 466, 110,  81, 134, 281, 405, 218, 115, 268,  61]), array([388,  28, 307, 466, 453, 190, 268, 494, 434, 218, 115]), array([115,  61, 453, 110, 468, 405, 281,  69, 100, 442, 218]), array([110, 400, 209, 115, 466, 231, 453, 220, 298,  61, 138]), array([164, 190, 222, 498,  81, 307, 160, 466, 110, 453, 442]), array([ 28, 112, 115, 466, 206, 453, 220,  61, 268, 307, 110]), array([115, 307, 110, 442, 268, 134,  61, 112,  28, 466, 453]), array([307, 110, 220, 453, 231, 134, 466,  61, 400, 442, 160]), array([281, 307, 468, 405, 164, 466, 271, 110,  89, 442, 453]), array([ 23, 133, 405,  81, 442, 110, 164, 307, 218,  61, 209]), array([281, 218, 110, 453, 405, 307, 305, 231, 115, 164,  81]), array([466, 307,  28, 268, 115, 405,  81, 133, 112, 110, 442]), array([ 81, 110, 271, 307, 224, 281, 453, 442, 405, 218, 115]), array([231,  28, 268,  81, 110,  58, 453,  61, 466, 442]), array([468, 405, 281, 442, 115, 110,  69, 218, 271, 453, 134]), array([453,  81, 307, 218, 110, 405, 442, 134, 115, 434, 160]), array([442, 220, 110, 268, 159, 358, 141, 164, 115, 231, 305]), array([164, 318, 405, 268,  61, 110, 115, 218, 466, 307,  81]), array([115, 110,  81, 442, 307, 134, 405, 224, 453,  61, 281]), array([134, 453, 160, 466,  61, 405, 110,  81, 218,  85, 281]), array([453, 100, 110, 281, 231,  85, 442, 209, 218,  81,  61]), array([ 61, 115, 400, 160, 466, 307,  81, 190, 231, 110, 268]), array([ 81, 305, 115, 218, 307, 453, 184, 405, 134, 466, 442]), array([405, 442,  81, 466,  23, 453, 293, 231, 216, 268, 220]), array([110,  52, 268, 115, 453, 466, 383, 112, 116, 339]), array([ 85,  87, 220, 231,  61, 110, 229, 417,  58, 453, 483]), array([442,  61, 220, 231,  81,  85, 466, 164, 405, 110, 281]), array([466, 307,  28, 268, 115, 405,  81, 133, 112, 110, 442]), array([281, 115, 218, 405, 307, 468, 160, 442, 110, 231, 209]), array([110, 218,  81, 281, 453, 442, 115, 405, 112, 307, 164]), array([115, 405, 110, 468, 281,  89, 453, 268, 218, 422, 112]), array([ 58, 231, 405,  87, 220, 417, 453,  61, 483, 268, 218]), array([400, 110, 160, 442, 339, 307,  81, 190, 115, 164,  61]), array([164,  28,  81, 405, 110, 466, 261, 133, 281, 307, 218]), array([216, 293, 442,  61,  81, 220,  23,  85, 231, 212, 466]), array([224,  61, 488, 115, 110, 453, 281, 156, 405,  89, 307]), array([ 28, 453, 466, 307, 115, 110,  61, 231, 268,  81, 335]), array([442, 281, 209, 307, 133, 470, 405,  81, 218,  74, 110]), array([ 61,  58, 483, 466,  85, 220,  87, 405, 258, 453, 231]), array([453, 231,  58, 466, 110, 220, 268, 129, 133,  28, 442]), array([466,  28, 220,  81, 112, 115, 268, 442, 453, 218, 281]), array([110, 307, 133, 466, 115, 218, 231, 220, 268,  61,  28]), array([ 28, 281, 405, 268,  81, 231, 133, 466, 112,  61, 442]), array([307, 190, 339, 383, 115, 498,  81, 110, 160, 268, 466]), array([442, 453, 110, 400, 115,  81, 160, 307, 218, 134, 190]), array([ 61, 110, 442, 231, 466, 117, 293, 216,  23, 268, 220]), array([231, 268, 466,  61,  81, 220, 218, 115, 112, 110, 358]), array([281,  81, 110, 164, 466, 307, 405, 134,  61, 442, 112]), array([115, 453, 218, 466, 405, 281, 209, 110, 156, 468, 307]), array([307, 442, 110, 405, 453, 133, 466, 134, 112, 209,  81]), array([231,  81, 442, 466, 216, 110, 453, 438,  61, 268, 293]), array([466, 134, 453,  61, 400, 110, 231, 209, 160, 281,  85]), array([115, 453, 444, 281, 134, 100,  61, 110, 209, 160,  81]), array([ 81, 307, 133, 160, 405,  28, 134, 218, 115, 190, 110]), array([281, 110, 453, 405, 134, 307, 115, 224, 218, 466, 442]), array([405, 115, 218,  61, 468, 281, 110, 307, 453, 466,  69]), array([405, 466, 307, 281, 422, 115,  89, 110, 231, 468, 453]), array([281, 110, 453, 218, 307,  89, 271, 405, 468, 115, 224]), array([307,  61, 405, 110, 164, 218, 222, 453, 442, 220, 268]), array([ 81,  28, 307, 453, 405, 112, 164, 466, 268, 110,  61]), array([115, 209, 453, 220, 307, 442, 110, 281,  61,  89, 405]), array([220, 466, 281, 110, 442, 453,  81,  28, 405, 307]), array([ 81, 110, 466, 222, 134, 405, 218, 307, 453, 442, 115]), array([231, 466, 293, 216, 212, 268, 442,  23, 110,  81, 229]), array([405, 307, 442, 453, 134, 468,  81, 110, 115, 112, 305]), array([218, 298, 453, 281, 224, 466,  61, 110, 134, 442, 307]), array([190, 231, 307, 498, 116, 339, 268, 383, 160,  85, 453]), array([110, 444, 453,  61, 442, 231, 268, 405, 218, 100, 281]), array([405, 134, 110, 220, 442, 307,  58, 453, 218,  81, 268]), array([405, 231, 110, 453, 281, 218, 115, 442, 468, 466, 160]), array([110, 115, 405, 218, 134, 453, 307, 305, 442, 268, 281]), array([253, 164,  58, 110, 220, 483, 453, 405, 190, 442,  87]), array([293,  81, 442,  61, 164, 216, 231,  23, 220, 453, 466]), array([405, 453, 110,  89, 281, 271, 307, 224, 468, 466, 218]), array([466, 218, 453,  58, 220, 405, 112, 417, 231, 164, 268]), array([164, 442, 307, 190,  61, 229, 110, 268, 358, 220, 400]), array([442,  81, 293,  23, 453, 261, 115, 110, 220, 216, 307]), array([ 81, 442, 110, 231,  61, 281, 115, 220, 307, 358, 134]), array([405, 466, 134, 164, 218, 453, 444, 115,  61, 110, 307]), array([405, 307, 164, 453, 115, 190, 218, 110, 134, 469, 289]), array([ 81, 307, 453, 466, 110, 405, 209, 133, 442, 134, 281]), array([442, 405,  81, 115, 231, 453, 466, 268,  61,  58, 117]), array([453, 405, 307, 134, 442, 110, 218, 133, 281,  81, 115]), array([453,  81, 405, 133, 218, 307, 164, 222, 209, 110, 231]), array([164,  81, 466,  23, 442, 231, 281,  61, 261, 293, 216]), array([405,  81, 281, 184, 218, 110, 115, 453, 307, 134, 442]), array([ 81, 115, 133, 307, 110, 466, 281, 453, 405, 442, 209]), array([453, 231, 468, 307, 218, 466, 115,  89, 405, 281,  61]), array([442, 231,  23, 268,  61,  81, 220, 261, 110, 466, 216]), array([405, 110, 100, 160,  61, 453, 466, 231, 218, 115,  28]), array([110, 281, 466, 115, 444, 453,  61, 100, 218, 231, 405]), array([115, 405, 307, 110, 470, 281, 134, 218, 442,  81, 224]), array([115, 281, 405, 442, 160,  81, 218, 110, 307, 209,  61]), array([466,  28, 112, 307, 268, 115, 388, 133,  81, 405, 231]), array([133, 405, 164, 307, 110, 268, 442, 300,  81, 466, 218]), array([453, 115, 405, 218, 442, 307,  81, 164, 110, 220, 466]), array([ 81, 164,  61, 110, 160, 453, 115, 307, 405, 466, 442]), array([307,  28, 466, 133,  81, 442, 110, 112, 220, 268,  61]), array([218, 453, 307,  81, 110, 115, 134, 466, 405, 222, 220]), array([307, 442, 115, 128, 160, 434, 110, 453, 164, 229, 218]), array([229,  23,  61, 453,  85, 220, 483,  28, 442, 281, 268]), array([466, 268, 442, 453,  61, 110, 220, 231, 307, 160, 405]), array([ 81, 442, 293, 216, 231, 115, 405, 110, 466,  23, 261]), array([218, 405, 110, 115, 307, 133, 134,  81, 453, 224, 281]), array([405,  69,  61, 110, 468, 453, 100, 231, 209, 115, 442]), array([115, 307,  81, 498, 110, 339, 190, 160, 453, 442, 383]), array([ 28, 453, 115, 133, 307, 494, 466, 416,  61,  81, 405]), array([ 28, 307, 453, 115, 110, 268, 442, 305,  52, 335, 339]), array([164, 442,  23,  81, 261, 231,  58, 220,  61, 405, 466]), array([ 81, 466, 268, 112, 442, 110, 453, 115, 231, 405, 220]), array([307, 115, 453, 206, 305, 405, 268, 110, 231, 190, 134]), array([405, 164, 271, 281, 307, 442,  61, 134, 110, 218, 466]), array([218, 405, 115, 453, 110, 134,  69, 231, 220,  61, 442]), array([218,  61, 307, 110, 115, 453, 442, 134,  81, 405, 224]), array([220, 229, 466, 442, 231,  61,  85, 261, 298,  23, 405]), array([115, 307, 222, 134, 220, 218, 405, 110, 453,  61,  81]), array([231, 307, 405, 190,  61, 112, 268, 224, 218, 453,  81]), array([231, 453, 268, 110, 442, 307,  85,  61, 116, 134, 466]), array([231,  61,  81, 110, 405, 442, 220, 453, 466, 438, 164]), array([222, 164,  61, 281, 218, 405, 220, 466,  28,  81, 453]), array([231, 307, 160, 110, 442, 268, 220, 112, 405, 134, 466]), array([298, 453,  61, 110,  85, 307, 466, 231, 434, 268, 281]), array([453, 220, 271, 405,  89, 110, 468, 281, 442, 218, 115]), array([ 28, 453, 466, 307, 115, 110,  61, 231, 268,  81, 335]), array([400,  81, 307, 128, 268, 160, 453, 115, 383, 110, 190]), array([453, 160, 307, 115, 110, 134, 218, 231, 405,  61, 268]), array([453, 405, 468,  69, 218, 110, 466, 115, 281, 307, 318]), array([110, 442, 444,  61, 405, 100, 115, 453, 231, 281, 466]), array([307, 115, 405,  61, 442, 281, 110, 134, 453, 318, 434]), array([466, 268,  28, 453, 231, 110, 115, 307,  61, 112,  81]), array([281, 100, 110, 115, 134,  61, 209, 453, 444, 442, 231]), array([117, 442, 110, 220,  61, 395, 231, 453, 298, 234, 134]), array([115, 307, 466, 218,  81, 442, 268,  28, 110, 134, 160]), array([442, 218, 405,  81, 110, 220, 307, 133, 466,  61, 164]), array([115, 268, 307, 466, 453, 112, 442,  81, 110,  28, 218]), array([ 69, 281, 110, 405, 442, 468, 307, 115,  81,  28, 271]), array([115, 405, 110, 218, 231,  61,  81, 281, 466, 453, 307]), array([231, 110, 160, 466, 453,  23,  61, 268, 218, 220, 442]), array([400, 218, 231, 110, 134, 160, 453,  81,  58, 220, 405]), array([218,  61, 405, 453, 466, 231,  81, 115, 220,  58, 307]), array([405, 453, 442, 110, 115, 218, 281, 468, 307, 271, 268]), array([218, 405, 307, 115, 442,  61, 134, 110,  28, 434,  81]), array([444,  61, 442, 453, 307, 434, 110, 220, 268, 281, 115]), array([466, 231,  61, 442, 453, 220, 110,  85, 396, 268, 307]), array([110, 281, 405, 468, 453, 115,  69, 209, 444, 218, 100]), array([110, 405, 281, 115,  61, 468, 218, 224, 453, 134,  89]), array([442, 110, 231, 468, 209, 466, 453, 115, 405, 218, 281]), array([ 61, 231, 405,  81, 218, 466, 220,  28, 307, 268, 133]), array([405, 133, 110,  81, 307, 442, 134,  28, 281, 218, 470]), array([281, 442, 110, 405, 307, 218,  61, 115, 231, 453, 134]), array([307, 218, 442, 405, 281, 271, 220, 453, 468, 466, 115]), array([281, 453, 442, 220, 307, 298,  61, 110, 218, 115, 268]), array([ 61, 220, 293, 442, 216, 110, 113,  81, 231, 453, 466]), array([453, 307, 442, 110, 218,  28, 134,  81, 115, 405, 160]), array([405, 466, 268, 281, 218, 453, 307, 209, 160, 442, 110]), array([271, 453, 281, 307, 115, 405, 110, 112, 468, 442, 231]), array([442,  81, 110, 453,  61, 115, 220, 209, 470, 134, 420]), array([ 81, 231, 110, 268,  28, 307, 442, 115, 466, 164, 206]), array([453, 164, 209, 115, 110, 466, 281, 307, 405,  81, 318]), array([ 61, 110, 115, 218, 405, 453,  81, 268, 281, 466, 231]), array([405, 115, 110, 281,  61, 209, 396, 466, 468, 218, 453]), array([453, 160, 218, 110, 305, 307, 115, 224, 231, 206, 466]), array([442,  23,  81, 164, 438, 110, 115, 307, 293, 405, 216]), array([307, 453, 442, 281, 405, 110,  81, 466, 218, 134, 133]), array([453, 307, 110, 134, 442, 396,  81, 115,  61, 218, 405]), array([453, 231, 268, 110, 218, 442, 160, 281, 405, 307, 209]), array([ 81, 442, 231, 216, 293,  23, 261, 281,  28, 405, 161]), array([293, 216,  81, 110, 115, 117,  61, 164, 442, 231,  23]), array([405, 307, 281, 110, 453, 164, 442, 468,  89,  70, 271]), array([110, 453, 220, 231,  61, 466, 268, 405, 116, 281, 307]), array([160, 453, 115, 307, 110, 231, 442, 134, 190,  61, 400]), array([231, 115,  61, 405, 453, 110, 134,  81, 160, 307, 281]), array([442,  81, 164,  23, 110, 220, 231, 405, 307, 268, 261]), array([110,  61, 160, 400, 466, 209, 218, 164, 405, 231, 134]), array([453, 220,  61, 466,  81, 231, 307, 442, 110, 133, 405]), array([307, 115, 405,  69, 110, 468, 453,  61, 231, 218,  81]), array([218, 231, 110, 112, 466, 405, 453, 268, 442,  28, 307]), array([231, 420, 110,  28, 453, 442, 281, 220, 466,  61,  81]), array([220, 405,  89, 281, 110, 453, 442, 231,  61, 307, 268]), array([268, 307, 453,  28, 115, 206,  81, 335, 466, 128, 164]), array([115, 405,  89, 218, 110, 468, 442, 281, 453, 224, 307]), array([164,  81, 231, 268, 160, 442, 307, 110, 112, 339, 434]), array([281, 164,  61,  81, 466, 110, 453, 134, 160, 442, 405]), array([405, 307, 115, 281, 466, 468, 453, 218, 110, 209,  61]), array([218, 307, 405, 110,  81, 115, 134, 453, 133, 442,  61]), array([268, 220, 405, 110, 442, 453, 307, 160, 164, 218, 231]), array([216, 220, 231, 293,  81, 442, 438, 261, 453, 405, 141]), array([453, 218, 220, 307, 405, 466, 164, 268, 110, 222,  81]), array([110, 442, 444,  61,  81, 453, 405, 100, 218, 115, 281]), array([231, 281, 466, 220,  61, 420, 453, 110, 496,  85, 442]), array([453, 112, 160, 442, 134, 218, 405, 110, 307, 281, 466]), array([209, 453, 405, 218, 281, 110,  81, 307, 466, 442, 115]), array([218, 216, 164, 405,  81, 281, 466, 453, 110, 442, 293]), array([220, 466, 396,  81, 110,  28, 442, 231, 218, 405, 268]), array([468, 110, 405, 115, 281,  69, 307, 453, 209, 442,  61]), array([405, 466, 134, 164, 218, 453, 444, 115,  61, 110, 307]), array([268,  61, 453, 110, 466,  81, 218, 112, 231, 405, 115]), array([442, 231, 405, 110, 112,  28, 453, 268, 307,  61, 160]), array([ 28, 133, 442, 112, 110, 115, 218,  81, 466, 307, 268]), array([405, 307, 133,  81, 110, 199,  28, 281, 498, 218, 442]), array([400, 134, 110, 160, 453,  61, 434, 231, 442, 115, 307]), array([ 85, 231, 442, 453,  61, 405, 466, 229, 220, 102, 145]), array([268, 466, 110,  61, 307, 405, 112, 442, 453,  28, 281]), array([ 81, 231,  61,  23, 115, 453, 293, 216, 442, 466, 307]), array([405, 164, 281, 307, 133, 231, 110, 218,  81,  61, 466]), array([ 28, 442, 466, 307, 453, 115,  61, 110, 268, 206, 305]), array([164, 231,  23,  81, 442, 133, 261, 438, 216, 405, 293]), array([ 61, 231, 405,  81,  85, 229, 466, 453, 454, 190, 220]), array([110,  81, 405, 261, 231,  23, 216, 442, 293, 268, 453]), array([231, 113, 453, 293, 216, 442, 220,  61,  85, 115, 110]), array([164, 298, 231, 110, 209, 307, 405, 218, 224,  61, 138]), array([218, 405,  81, 468, 100, 281, 110, 307, 318, 115, 453]), array([231, 218, 307, 115, 442, 405, 110, 164, 453, 220, 466]), array([453, 466, 281, 110, 307, 405, 442,  81, 115, 268,  61]), array([115, 218, 405,  81, 110, 307,  28, 133, 268, 466, 453]), array([160, 442, 218, 498, 115, 405,  81, 133, 164, 307, 190]), array([199, 220,  81, 405, 466,  28, 453, 218,  58,  87,  61]), array([281, 453, 110, 307,  89, 206, 115, 231, 468, 112, 405]), array([298, 468, 218, 307,  69, 405, 281, 156, 110, 442,  61]), array([115, 405, 110,  61,  69, 453, 281, 218, 468, 307, 224]), array([110, 209, 298, 453, 281, 134, 138, 218, 115, 466, 307]), array([115, 405, 110, 218, 231,  61,  81, 281, 466, 453, 307]), array([231, 453, 307,  28, 494, 268, 466, 115, 383,  81,  52]), array([442,  61, 110, 231, 220, 293, 216, 115, 466, 453, 268]), array([110,  61, 293, 453, 216, 444, 442, 231, 115,  81, 100]), array([281,  61, 209, 466, 110, 115, 405, 453, 134, 231, 160]), array([281, 220,  61, 218, 231, 110, 453, 444, 442, 209, 100]), array([405,  69, 281, 110, 466,  22, 307, 468, 298, 453, 156]), array([453, 218,  58, 222, 466,  81, 112, 164, 405, 231, 307]), array([160, 145, 220,  81,  85, 231, 420,  61, 164, 218,  28]), array([110,  69, 453, 405, 442, 281, 209, 231,  61, 468,  81]), array([ 28, 110, 442, 133, 307,  81, 268, 115, 231, 466,  61]), array([271, 307, 468, 218, 405, 115, 281,  61, 110, 305, 229]), array([160, 442, 115, 453, 405, 110, 281,  61, 444, 231, 100]), array([134, 218, 222, 307, 220, 453,  61, 115, 405, 281, 466]), array([405, 466, 453, 307, 222, 220, 218, 268, 110, 281]), array([466, 442, 160, 453, 110, 209, 218, 405, 115,  81, 134]), array([307, 138, 112, 434, 110, 218, 115, 134, 305, 468, 453]), array([453, 444, 115, 110,  61, 307, 442, 405, 281, 268,  85]), array([442, 141, 466, 453, 363, 110, 220, 405, 231, 395, 396]), array([ 58, 220,  87, 453, 417, 405, 218, 466, 483, 231, 442]), array([220, 466, 110,  61, 231, 229,  85, 208, 281, 454, 405]), array([229,  61, 454, 231, 164,  85, 145, 453, 466, 405, 281]), array([453, 400, 110, 307, 160, 466,  81, 209, 420, 498, 115]), array([339, 145,  85, 453, 268, 164, 231, 442,  23, 190, 466]), array([442, 110, 115, 453, 281, 405, 218, 466, 134, 468,  61]), array([468, 307,  89, 110, 405, 281, 231, 268,  61, 229, 218]), array([468, 442,  69, 405, 453, 110, 209, 444,  81, 115, 160]), array([444, 115, 110, 100, 453,  61, 281, 307, 134, 442, 209]), array([442, 110, 405, 466, 115, 218, 281, 307, 209, 133,  28]), array([466, 444, 115, 112, 110, 405, 453, 100, 268,  81, 396]), array([405, 468, 110, 453,  69, 442, 115, 281,  61, 100, 209]), array([445, 218,  81, 307, 405, 224, 110, 134, 133, 281, 442]), array([110,  61, 293, 453, 216, 444, 442, 231, 115,  81, 100]), array([434, 466, 268, 231, 116, 115, 112, 110, 383, 453, 220]), array([ 81,  28, 115, 442, 110, 281, 405, 466, 268, 218, 453]), array([ 28,  61, 231, 164,  81, 216, 442,  23, 293, 220, 307]), array([466,  81, 220, 405,  58, 218, 222, 453, 268, 231, 307]), array([220, 110,  85, 442, 231, 268, 417, 115,  61, 453, 164]), array([159, 442, 110, 112, 115, 441, 453, 268, 307,  61]), array([470, 281, 115,  81, 405, 110, 224, 468, 100, 218, 160]), array([453, 396,  81, 141, 395, 110, 442, 218, 405, 220, 466]), array([307,  61,  28, 442, 231, 117, 268, 466, 453, 129, 281]), array([298, 110, 453, 209, 231, 442,  61, 115, 220, 138, 224]), array([ 81, 268, 466, 110, 231, 112, 453, 358, 405,  28, 442]), array([134, 453, 307, 110, 115, 405, 218, 442,  81, 281, 209]), array([110,  61, 444, 442, 209, 218, 100, 115, 307, 468, 453]), array([209, 134, 115, 112, 110, 307, 442,  61, 453, 160,  81]), array([218, 164, 405,  81, 110, 307, 268,  28,  61, 466, 442]), array([115,  81, 307, 442, 268, 190, 128,  52, 466, 160,  28]), array([453, 164, 442, 268, 231,  85, 229,  61, 220, 466, 381]), array([ 58, 307, 453, 405, 110, 220, 218, 164, 442, 134,  81]), array([112, 405, 224, 307, 281, 442, 453, 134, 133, 110,  81]), array([405,  58, 453, 483, 417, 231, 229,  87, 220,  61, 442]), array([405, 220, 453, 218,  58, 466,  61,  81, 222, 110, 134]), array([110, 100, 444, 442, 405, 209,  61, 134, 453, 218,  81]), array([281, 442, 453, 110,  81, 466, 160,  61, 400, 134, 405]), array([466, 453, 442, 293, 110,  61,  81, 216, 231, 281, 220]), array([293,  61, 216, 466,  85, 220, 110, 102, 442, 231, 229]), array([442, 117, 453, 231, 220,  81, 110, 261,  23, 141, 405]), array([ 61, 444,  81, 100, 453, 110, 115, 466, 231, 281, 209]), array([218, 281, 453, 405, 110, 307, 442, 115, 318, 468,  61]), array([231, 405, 453, 268, 307, 442, 110, 466, 190, 281,  61]), array([190, 268, 466, 307, 453, 218, 110, 405, 358, 115,  61]), array([405, 231, 209, 453, 281, 307, 110,  81, 115, 224, 218]), array([ 28, 307, 110, 112, 466, 388, 442, 115, 216, 293, 159]), array([453,  89, 307, 468, 115, 281, 442, 110, 271, 405, 218]), array([218, 209, 110, 453, 164, 405,  81, 298, 307, 224, 442]), array([ 81, 268, 133, 164, 307, 442, 453, 405, 110, 220, 298]), array([466, 112, 307, 141,  81,  61, 442, 453, 405, 115, 164]), array([160, 434,  81, 190,  61, 110, 307, 112, 442, 400, 115]), array([307, 231, 110, 112, 453, 115, 268, 190, 160, 218, 116]), array([160,  61, 307, 110, 164, 434, 400, 281,  85, 405, 442])]


## In[ ]:


#data = [[3326571.725197324, 3154970.0720751956, 3005546.687136788, 2939471.3179067187, 2915461.2237821487],
#[4498993.820461707, 4301434.563097052, 4142249.5506123775, 4043630.336811959, 4019343.1666131536],
#[3187072.592168979, 3043408.383749654, 2949179.8489271672, 2912725.32058333, 2885924.9286203277],
#[3748274.5077334996, 3558127.020760696, 3417485.0322979433, 3319142.58714804, 3295268.8734280732],
#[15588454.012740301, 15435887.531138442, 15312090.6144848, 15235430.776575955, 15198121.868878432]]

#plt.plot(data[0])





def subsample_adata_by_batches(adata, num_batches, cells_per_batch, random_state=42):
    """
    Subsamples an AnnData object by selecting a specified number of batches
    and then sampling a specified number of cells from each selected batch.
    """
    rng = np.random.default_rng(random_state)
    all_batches = adata.obs['batch'].unique().tolist()

    if num_batches > len(all_batches):
        print(f"Warning: Requested {num_batches} batches, but only {len(all_batches)} available. Using all available batches.")
        selected_batches = all_batches
    else:
        selected_batches = rng.choice(all_batches, num_batches, replace=False)

    sampled_adatas = []
    for batch_id in selected_batches:
        batch_adata = adata[adata.obs['batch'] == batch_id].copy()
        if batch_adata.n_obs == 0:
            continue

        num_to_sample = min(cells_per_batch, batch_adata.n_obs)
        if num_to_sample > 0:
            indices = rng.choice(batch_adata.obs_names, num_to_sample, replace=False)
            sampled_batch_adata = batch_adata[indices].copy()
            sampled_adatas.append(sampled_batch_adata)
        else:
            print(f"Warning: No cells to sample for batch {batch_id} with {cells_per_batch} cells requested.")

    if not sampled_adatas:
        return None

    # Concatenate the sampled AnnData objects
    concatenated_adata = sc.concat(sampled_adatas, axis=0, join='outer', merge='unique')

    # Preserve necessary original metadata (e.g., 'timeseries' flag for scoring)
    concatenated_adata.uns = adata.uns.copy()
    concatenated_adata.var = adata.var.copy()

    return concatenated_adata


def make_timetable(datasets):
    '''
    Generates and plots a timetable of integration method runtimes
    for different dataset sizes based on varying number of batches and cells per batch.

    The x-axis configurations are: (A=2, B=200), (A=3, B=300), ..., (A=10, B=1000)
    where A is the number of batches and B is the number of cells sampled per batch.
    '''
    # The task specifies to use datasets[13] for this experiment
    original_dataset = datasets[13]

    # Define the (A, B) pairs as specified in the prompt
    x_axis_configs = []
    for A in Range(2, 11): # A ranges from 2 to 10
        B = A * 100      # B ranges from 200 to 1000 (A*100)
        x_axis_configs.append((A, B))

    all_runtimes_data = []

    for num_batches, cells_per_batch in x_axis_configs:
        print(f"Processing config: {num_batches} batches x {cells_per_batch} cells/batch...")

        # Create a subsampled AnnData object
        sub_dataset = subsample_adata_by_batches(original_dataset, num_batches, cells_per_batch)
        if sub_dataset is None:
            print(f"Skipping config {num_batches}x{cells_per_batch} due to insufficient data after subsampling.")
            continue

        # Run all integration methods on the subsampled dataset
        # run_all expects a list of datasets, so we wrap sub_dataset in a list
        _, fnames, runtimes_dict = run_all([sub_dataset], [.55])

        config_label = f"{num_batches}x{cells_per_batch}"

        # Collect runtimes for all methods for the current configuration
        for method_name in fnames:
            runtime = runtimes_dict.get(method_name, 0) # Get runtime for this method
            all_runtimes_data.append({
                'config_label': config_label,
                'num_batches': num_batches,
                'cells_per_batch': cells_per_batch,
                'method': method_name,
                'runtime': runtime
            })

    if not all_runtimes_data:
        print("No runtime data collected. Exiting timetable generation.")
        return

    # Convert collected data to a DataFrame for easy plotting
    runtimes_df = pd.DataFrame(all_runtimes_data)

    # Plot the results
    plt.figure(figsize=(12, 7))
    sns.lineplot(data=runtimes_df, x='config_label', y='runtime', hue='method', marker='o')
    plt.title('Integration Method Runtimes Across Dataset Sizes')
    plt.xlabel('Number of Batches x Cells per Batch')
    plt.ylabel('Runtime (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()

    return runtimes_df # Optionally return the dataframe




from scipy.stats import gmean

def test_make_results_table():
    print("Starting test_make_results_table...")

    # 1. Get data
    datasets, _ = get_data()
    print(f"Loaded {len(datasets)} datasets for testing.")

    # the later should be ts datasets... so we have a mix. and dont get nans :)
    datasets = datasets[:2] + datasets[-10:-8]

    # 2. Run all integration methods
    # Using a small subset of scalpvalues to speed up the test
    # And running on a single dataset for brevity
    # datasets_after_run, fnames, times = run_all([datasets[0], datasets[1]], scalpvalues=[.55, .75])
    datasets_after_run, fnames, times = run_all(datasets, scalpvalues=[.55, .75])

    print(f"Finished running {len(fnames)} methods on {len(datasets_after_run)} datasets.")
    print("Methods run:", fnames)

    # 3. Calculate scalp scores
    scores = scalpscore(datasets_after_run)
    print("Calculated scalp scores.")
    # print("Raw scores:", scores) # for debugging

    # 4. Define the chosen_scalp value for the test
    # This should match one of the 'Scalp: X.XX' strings in fnames
    chosen_scalp_option = '0.55' # Or '0.75' if you prefer

    # Filter fnames to get the exact Scalp method name
    full_scalp_method_name = [name for name in fnames if f'Scalp: {chosen_scalp_option}' in name][0]
    print(f"Chosen Scalp method for table: {full_scalp_method_name}")


    # 5. Call make_results_table
    results_df, latex_table = make_results_table(scores, datasets_after_run, chosen_scalp_option)
    print(latex_table)



def make_results_table(scores, datasets, chosen_scalp):
    """
    scores: dict of dataset_id -> {method: {'label_mean':..., 'batch_mean':...}}
    datasets: list of AnnData or similar objects with .uns['timeseries']
    chosen_scalp: string, the exact scalp variant name to include
    """
    # -------- Step 1: Build dataframe ----------
    rows = []
    for ds, methods_dict in scores.items():
        for meth, vals in methods_dict.items():
            label = vals['label_mean']
            batch = vals['batch_mean']
            geo = gmean([label, batch])
            rows.append({'dataset': ds, 'method': meth,
                         'label': label, 'batch': batch, 'geomean': geo})
    full_df = pd.DataFrame(rows)

    # check for nans
    if full_df.isnull().any().any():
        print("Warning: NaN values found in scores DataFrame. This might affect ranking.")
        print(full_df[full_df.isnull().any(axis=1)])

    ts_ids = [int(i) for i, e in enumerate(datasets) if datasets[i].uns.get('timeseries', False)]
    batch_ids = [int(i) for i, e in enumerate(datasets) if not datasets[i].uns.get('timeseries', False)]

    # -------- Step 2: Filter methods to include only chosen scalp + all non-scalp ----------
    allowed_methods = set(m for m in full_df['method'] if not m.startswith('Scalp'))
    scalp_methods = [m for m in full_df['method'].unique() if m.startswith('Scalp')]
    # If chosen_scalp is provided, filter for it
    selected_scalp_methods = [m for m in scalp_methods if str(chosen_scalp) in m]
    allowed_methods.add(selected_scalp_methods[0])


    df = full_df[full_df['method'].isin(allowed_methods)].copy()

    # -------- Step 3: Compute ranks ----------
    ranks = df.groupby('dataset')[['label', 'batch', 'geomean']].rank(ascending=False)
    df[['label_rank', 'batch_rank', 'geo_rank']] = ranks

    # check nans
    if df.isnull().any().any():
        print("Warning: NaN values found in ranks DataFrame. df is clean 1250")
        print(ranks[ranks.isnull().any(axis=1)])

    # -------- Step 4: Compute mean ranks ----------
    def mean_ranks(ids):
        sub = df[df['dataset'].isin(ids)]
        return sub.groupby('method')[['label_rank', 'batch_rank', 'geo_rank']].mean()

    ts_ranks = mean_ranks(ts_ids)
    batch_ranks = mean_ranks(batch_ids)
    total_ranks = df.groupby('method')[['label_rank', 'batch_rank', 'geo_rank']].mean()

    result = pd.concat([
        ts_ranks.add_prefix('TS_'),
        batch_ranks.add_prefix('Batch_'),
        total_ranks.add_prefix('Total_')
    ], axis=1)


    # check nan
    if result.isnull().any().any():
        print("Warning: NaN values found in ranks DataFrame. This might affect mean ranks.")
        print(ranks[ranks.isnull().any(axis=1)])
        breakpoint()




    # -------- Step 5: Transpose ----------
    result_t = result.T

    # -------- Step 6: Boldface best in each row ----------
    latex_df = result_t.copy()
    for row in latex_df.index:
        vals = latex_df.loc[row]
        minval = pd.to_numeric(vals, errors='coerce').min()
        latex_df.loc[row] = [
            f"\\textbf{{{float(v):.3f}}}" if np.isclose(float(v), minval) else f"{float(v):.3f}"
            for v in vals
        ]

    latex_code = latex_df.to_latex(escape=False)

    return result_t, latex_code







