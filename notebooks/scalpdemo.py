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

from tablemaker import make_results_table , format_res_T


'''
# use scalp env :)
import scalpdemo    as sd
datasets, dataset = sd.get_data()
datasets = Map(sd.Scalp, datasets)
scores = sd.scalpscore(datasets)
sd.scalp.plot(datasets[3], 'scalp', color = ['label', 'batch'])
'''


# In[3]:

configs  = [
 {'maxdatasets':4, 'maxcells':500,'filter_clusters': 10, 'slow':0}, # debug
    {'maxdatasets':10, 'maxcells':1000,'filter_clusters': 0, 'slow':0}, # real data
 {'maxdatasets':10, 'maxcells':1000,'filter_clusters': 10, 'slow':0} # plotting
]

def get_data(config =0):
    conf = configs[config]

    datasets = scalp.data.scmark(scalp.test_config.scmark_datapath,  **conf)
    print(f"{len(datasets)=}")
    datasets += scalp.data.timeseries(scalp.test_config.timeseries_datapath,**conf)
    print(f"{len(datasets)=}")
    datasets +=scalp.data.scib(scalp.test_config.scib_datapath,**conf)
    print(f"{len(datasets)=}")

    return datasets, datasets[12]

def get_is_ts(datasets):
    return [d.uns['timeseries'] for d in datasets]




import scanpy as sc


# # COMPARISON + SCORING

import scanpy as sc
from scalp import graph as sgraph
from scipy.sparse import csr_matrix
import scalp.data.similarity as sim
import scalp.data.transform as trans
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


    # asd = trans.split_by_obs(dataset)
    # for aa in asd:
    #     z = np.var(ut.zehidense(aa.X),axis=0)
    #     # print(z.A1.shape)
    #     aa.var['myvar'] =z
    # stair =  sim.dynamic_sim(asd, hvg='myvar')
    # print(stair)

    # hub1_algo  hub1_k  hub2_algo  hub2_k   k  outlier_threshold  config_id     score       time
    #   0       9          3       9          19           0.970536    30  2.188789  15.822385
    # 0      10          3       6  15
    # grap = scalp.graph.integrate(dataset,hub1_algo = 0, hub1_k = 10,  hub2_algo=3, hub2_k=6,  k=15,  dataset_adjacency=None, outlier_threshold=ot)

    grap = scalp.graph.integrate(dataset,hub1_algo = 2, hub1_k = 12,  hub2_algo=2, hub2_k=12,  k=12,  dataset_adjacency=False, outlier_threshold=ot)
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
    # dataset.uns.setdefault('integrated',[])
    # dataset.uns['integrated'].append('scalp')
    dataset.uns.setdefault('methods',[])
    dataset.uns['methods'].append('scalp')
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
# def run_all(datasets, scalpvalues = [.35,.55, .7,.75,.8, .9]):
def run_all(datasets, scalpvalues = [.15,.2, .25, .3, .35, .45,  .55, .7, .9]):


    # SETUP TASKS
    funcs = [scalp.mnn.harmony, scalp.mnn.scanorama, scalp.mnn.bbknnwrap, scalp.mnn.combat]
    for ot in scalpvalues:
        funcs.append( functools.partial(Scalp, ot=ot))
    fuid = Range(funcs)
    dataid = Range(datasets)
    tasks = [(f,d) for f in fuid for d in dataid]
    fnames = 'Harmony Scanorama BBKNN ComBat'.split()
    fnames+=[f'Scalp: {s}' for s in scalpvalues]

    # RUN TASKS
    def run(fd):
        starttime = time.time()
        f,d = fd
        fun = funcs[f]
        dat = datasets[d]
        stack = fun(dat)
        return stack, time.time()-starttime
    mydata = ut.xxmap(run, tasks)
    mydata, runtimes = Transpose(mydata)


    # times = defaultdict(int)
    # for (fi, di), t in zip(tasks, runtimes):
    #     times[fnames[fi]] += t

    times = defaultdict(lambda: defaultdict(float))
    for (fi, di), t in zip(tasks, runtimes):
        times[di][fnames[fi]] = t

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

    tasks = [(f,d) for f in fanmes for d in Range(datasets) if datasets[d].uns['name']!= 'done_lung' ]

    def f(item):
        fn,ds = item
        r = scalp.score.scib_scores(datasets[ds],fn)
        r.update({'method':fn})
        r.update({'dataset':ds})
        return r

    df =  pd.DataFrame(ut.xxmap(f,tasks))
    df.to_csv(saveas)
    return df



def mkscib_table_old(SCIB):

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


def ggmean(arr):
    arr = np.array(arr)
    arr = arr[arr > 0]
    return gmean(arr)

def mkscib_table(SCIB,datasets):

    # check the input:
    if not all(col in SCIB.columns for col in ['batch', 'label', 'method', 'dataset']):
        raise ValueError("Input DataFrame SCIB must contain 'batch', 'label', 'method', and 'dataset' columns.")


    #subselect from SCIP column 'method' either does not contain 'Scalp' or contains '0.2'

    selected_methods = [m for m in SCIB['method'].unique() if 'Scalp' not in m or ('0.2' in m and '.25')]
    SCIB = SCIB[SCIB['method'].isin(selected_methods)].copy()


    resdict = defaultdict(dict)

    # each method should get 9 entries:
    tsids = get_is_ts(datasets)

    for method in SCIB['method'].unique():
        for datasubset in "TS BATCH ALL".split():
            data = SCIB
            if datasubset == 'TS':
                    data = SCIB[ [ tsids[int(d)]  for d in SCIB['dataset']] ]
            if datasubset == 'BATCH':
                    data = SCIB[ [ not tsids[int(d)]  for d in SCIB['dataset']] ]
            rows = SCIB[ SCIB['method'] == method ]

            for target in "batch label all".split():
                if target in "batch label".split():
                    resdict[method][f'{datasubset}_{target}'] = ggmean(rows[target].values)
                else:
                    r = np.concatenate((rows['batch'].values, rows['label'].values))
                    resdict[method][f'{datasubset}_{target}'] = ggmean(r)


    df = pd.DataFrame(resdict)

    # df_styled = df.style.format("{:.2f}")
    # checkout make results table on how to boldface the highest entry per row


    latex_df = df.copy()
    for row in latex_df.index:
        vals = latex_df.loc[row]
        minval = pd.to_numeric(vals, errors='coerce').max()
        latex_df.loc[row] = [
            f"\\textbf{{{float(v):.2f}}}" if np.isclose(float(v), minval) else f"{float(v):.2f}"
            for v in vals
        ]


    latex_code = latex_df.to_latex(escape=False)
    latex_code = latex_code.replace('_', ' ')
    print(latex_code)
    return


    # Prepare for LaTeX output by styling the DataFrame
    def highlight_max(s):
        is_max = s == s.max()
        return ['\\textbf{%s}' % x if v else '%s' % x for v, x in zip(is_max, s.values)]

    df_styled = df.style.format("{:.2f}").apply(highlight_max, subset=pd.IndexSlice[:, :]) # Apply to all columns
    latex_output = df_styled.to_latex()

    return df, latex_output # Return both the DataFrame and its LaTeX representation

    print(df_styled)
    # Prepare for LaTeX output
    latex_output = df_styled.to_latex()

    # hrules=True,
    # caption="Performance (Geometric Mean) for Integration Methods",
    # label="tab:geomean_scores")

    print(latex_output)

    # return df

    # return result.to_latex(index=False)

    # blabla
    allmethods = SCIB['method'].unique()
    scalp_methods = [m for m in allmethods if m.startswith('Scalp')]
    other_methods = [m for m in allmethods if not m.startswith('Scalp')]

    return




fixed_color_methods = {
    "Scanorama": "blue",
    "BBKNN": "green",
    "ComBat": "purple",
    "Harmony": "black",
}


def makepalette(scalp_items:list):
    '''
    returns a palette dict.
    for the keys from fixed_color_methods, we use sns color_palete flare.
    then across the scalp items we use 'crest'
    '''

    def pal(items, cmap_name):
        sorted_items = sorted(items)
        colors = sns.color_palette(cmap_name, n_colors=len(items))
        return dict(zip(sorted_items, colors))

    old = fixed_color_methods.keys()
    ret = {}
    ret.update(pal(scalp_items,'Reds'))
    # ret.update(pal(old,'mako')) # doing this later so we can just pass all the methods i guess
    ret.update(fixed_color_methods)
    return ret

def split_sc2(scores, datasets):
    # scores is a list of dicts with a dataset id entry..
    # we need to split by get_is_ts(datasets)
    ts = get_is_ts(datasets)
    batch =  [s for s in scores if not ts[s['dataset']]]
    timeseries =  [s for s in scores if  ts[s['dataset']]]
    return batch, timeseries


import matplotlib.transforms as mtrans




def barplot(geomean, datasets):
    ax = plt.gca()

    # --- 1. Data Preparation ---
    geomean.index.name = 'method'
    df_melted = geomean.reset_index().melt(id_vars='method', var_name='dataset', value_name='score')

    # Logic for grouping
    df_melted['group'] = ['timeseries' if datasets[int(i)].uns['timeseries'] else 'batch' for i in df_melted.dataset]
    df_melted['size'] = [len(np.unique(datasets[int(i)].obs['batch'])) for i in df_melted.dataset]

    # --- 2. Ordering & Palette ---
    order = sorted([m for m in df_melted['method'].unique() if 'Scalp' in m],
                   key=lambda x: float(x.split(': ')[1])) + \
            sorted([m for m in df_melted['method'].unique() if 'Scalp' not in m])

    palette = makepalette(order)

    # --- 3. Plotting (Horizontal) ---
    # orient='h' ensures bars run horizontally
    sns.boxplot(data=df_melted, y="method", x="score", hue="group",
                palette="Blues", order=order, ax=ax, orient='h')

    # --- 4. Styling ---
    ax.set_ylabel('') # Remove generic 'method' label
    ax.set_xlabel('Score', fontsize=12)
    ax.tick_params(axis='y', labelsize=11) # Method names
    ax.tick_params(axis='x', labelsize=11) # Scores

    # Apply colors to Y-tick labels (Method names)
    [i.set_color(palette.get(i.get_text(), 'gray')) for i in ax.get_yticklabels()]

    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_axisbelow(True)

    # --- 5. Legend (Upper Right) ---
    # Moved to upper right as requested.
    # framealpha adds transparency in case it overlaps with long bars.
    ax.legend(loc='upper right', borderaxespad=0.5, fontsize=10, framealpha=0.9)
    fig = plt.gcf()
    plt.show()
    return fig



def bluestar(scores2):

    # --- 1. Data Preparation ---
    scores2_df = pd.DataFrame(scores2)
    z = scores2_df[scores2_df.method != 'scalp'].copy()

    z_stats = z.groupby("method").agg(
        label_mean=("label_mean", "mean"),
        label_std=("label_mean", "std"),
        batch_mean=("batch_mean", "mean"),
        batch_std=("batch_mean", "std")
    ).reset_index()

    # --- 2. Ordering & Color Palette ---
    all_methods = sorted(z_stats['method'].unique())
    palette = makepalette(all_methods)

    scalp_methods = sorted([m for m in all_methods if "Scalp:" in m], key=lambda x: float(x.split(': ')[1]))
    other_methods = [m for m in all_methods if "Scalp:" not in m]
    legend_order = other_methods + scalp_methods


    legend_order = legend_order[::-1]

    # --- 3. Plotting ---
    ax = plt.gca()
    sns.reset_defaults()
    for method_name in legend_order:
        row = z_stats[z_stats['method'] == method_name]
        if not row.empty:
            row = row.iloc[0]
            color = palette.get(method_name, 'gray')

            ax.errorbar(
                x=row["label_mean"],
                y=row["batch_mean"],
                xerr=row["label_std"],
                yerr=row["batch_std"],
                fmt='o',
                label=method_name,
                color=color,
                alpha=1,
                capsize=0,
                markersize=0
            )
    fontsize = 12
    sns.set(rc={ 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize})
    # --- 4. Styling ---
    # ax.tick_params(labelsize=12)
    ax.tick_params(axis='both', which='both', direction='out', length=4, width=1, colors='black', bottom=True, top=False, left=True, right=False)
    ax.grid(True, linestyle='--', alpha=0.7)

    ax.set_ylabel("Batch Mean Score", fontsize=fontsize)
    ax.set_xlabel("Label Mean Score", fontsize=fontsize)
    ax.tick_params(axis='both', labelsize=fontsize)
    # --- 5. Legend (Tighter Layout) ---
    # Adjusted bbox_to_anchor to reduce gap between axis and legend
    # Moved from -0.15/-0.25 to -0.12 to sit closer to the labels
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=fontsize)

    fig = plt.gcf()
    plt.show()
    return fig

'''


def bluestar(scores2):
    """
    Generates and displays scatter plots of 'label_mean' vs 'batch_mean' scores
    from the provided DataFrame, showing individual dataset points and
    mean points with error bars.

    Args:
        scores2_df (pd.DataFrame): DataFrame containing 'method', 'label_mean',
                                  'batch_mean' (and optionally 'label_std', 'batch_std' for error bars).
    """
    # Filter out 'scalp' method if it exists (assuming it's a generic name and specific Scalp: 0.XX are preferred)
    scores2_df = pd.DataFrame(scores2)
    z = scores2_df[scores2_df.method != 'scalp'].copy()


    palette = {
        "Scanorama": "red",
        "BBKNN": "green",
        "ComBat": "purple",
        "Harmony" : "orange",
   }

    # Define methods that should appear at the end with specific colors
    # Get all unique methods present in the dataframe
    all_methods = sorted(z['method'].unique())

    # Separate fixed-color methods from others
    other_methods = [m for m in all_methods if m not in fixed_color_methods]

    # Create a Viridis colormap for the 'other_methods'
    viridis_cmap = plt.get_cmap('viridis')
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
    # palette = {**viridis_palette, **fixed_color_methods}
    palette = makepalette(all_methods)

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
                alpha = 1,
                capsize=3,  # size of the caps on the error bars
                markersize=5
            )

    # Customize legend and labels
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.)
    ax.set_xlabel("Label Mean Score")
    ax.set_ylabel("Batch Mean Score")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    return

'''


def barplot_v2(geomean, datasets):

    # make data
    geomean.index.name = 'method'
    df_melted = geomean.reset_index().melt(id_vars='method', var_name='dataset', value_name='score')
    df_melted['group'] = [ 'timeseries' if datasets[int(i)].uns['timeseries'] else 'batch'  for i in  df_melted.dataset] # np.where(df_melted.dataset.astype(int) < 13 , 'timeseries', 'batch')
    df_melted['size'] = [ len(np.unique(datasets[int(i)].obs['batch']) ) for i in  df_melted.dataset] # np.where(df_melted.dataset.astype(int) < 13 , 'timeseries', 'batch')


    # colors, fontsize  and label order
    palette = fixed_color_methods
    size = 12
    sns.set(rc={"font.size":size,"axes.titlesize":size,"axes.labelsize":size,   'xtick.labelsize': size, 'ytick.labelsize': size,},style="white")
    order = sorted([m for m in df_melted['method'].unique() if 'Scalp' in m],
                   key=lambda x: float(x.split(': ')[1])) + sorted([m for m in
            df_melted['method'].unique() if 'Scalp' not in m])
    palette  = makepalette(order)

    # PAINTING
    g = sns.boxplot( data=df_melted, x="method", y="score", hue ="group", palette= "Blues", order = order)

    g.set_ylabel('Score')
    g.set_xlabel(None)


    # TICKS
    g.set_xticklabels(rotation=45, labels = order, ha = 'right', x= 4)
    for i, label in enumerate(g.get_xticklabels()):
        label.set_x(i+ 0.6)
    g.tick_params(axis='both', which='both', direction='out', length=4, width=1, colors='black', bottom=True, top=False, left=True, right=False)
    # Apply a translation to all x-tick labels
    offset = mtrans.ScaledTranslation(.1, 1.2, g.figure.dpi_scale_trans)
    for label in g.get_xticklabels():
        label.set_transform(g.transData + offset)
    # THIS WOULD BRING THE COLORS BACK:
    # [i.set_color( palette.get(i._text,'gray')) for  i in g.get_xticklabels()]
    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    # LEGEND + RETURN
    plt.gca().set_axisbelow(True)
    plt.legend(bbox_to_anchor=(.2, .25), loc='upper left', borderaxespad=0)
    fig = plt.gcf()
    plt.show()
    return fig

def barplot_backup_delMe(geomean, datasets):

    geomean.index.name = 'method'
    df_melted = geomean.reset_index().melt(id_vars='method', var_name='dataset', value_name='score')
    #df_melted['group'] = np.where(df_melted.dataset.astype(int) < 13 , 'timeseries', 'batch')
    df_melted['group'] = [ 'timeseries' if datasets[int(i)].uns['timeseries'] else 'batch'  for i in  df_melted.dataset] # np.where(df_melted.dataset.astype(int) < 13 , 'timeseries', 'batch')
    df_melted['size'] = [ len(np.unique(datasets[int(i)].obs['batch']) ) for i in  df_melted.dataset] # np.where(df_melted.dataset.astype(int) < 13 , 'timeseries', 'batch')
    # df_melted['group']=df_melted.dataset.astype(int) % 16

    if False: # old plot
        g = sns.catplot( data=df_melted, x="method", y="score", hue = 'group')# native_scale=True, zorder=1 )
        means = df_melted.groupby("method")["score"].mean()
        for i, day in enumerate(means.index):
            plt.scatter(i, means[day], color='black', marker='x', s=50, label='Mean' if i == 0 else "")
        g.set_xticklabels(rotation=45, labels = np.unique(df_melted.method)) # ha='right' is useful here too
        plt.show()

    palette = fixed_color_methods
    order = sorted([m for m in df_melted['method'].unique() if 'Scalp' in m], key=lambda x: float(x.split(': ')[1])) + sorted([m for m in df_melted['method'].unique() if 'Scalp' not in m])
    g = sns.boxplot( data=df_melted, x="method", y="score", hue ="group", palette= "Blues", order = order)
    g.set_xticklabels(rotation=45, labels = order, ha = 'right', x= 4)
    for i, label in enumerate(g.get_xticklabels()):
        label.set_x(i+ 0.6)
    offset = mtrans.ScaledTranslation(.1, .90, g.figure.dpi_scale_trans)
    for label in g.get_xticklabels():
        label.set_transform(g.transData + offset)

    [i.set_color( palette.get(i._text,'gray')) for  i in g.get_xticklabels()]

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_axisbelow(True)
    plt.legend(bbox_to_anchor=(.2, .25), loc='upper left', borderaxespad=0.)
    plt.show()





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


def make_timetable(datasets, datapoints = 9):
    '''
    Generates and plots a timetable of integration method runtimes
    for different dataset sizes based on varying number of batches and cells per batch.

    The x-axis configurations are: (A=2, B=200), (A=3, B=300), ..., (A=10, B=1000)
    where A is the number of batches and B is the number of cells sampled per batch.
    '''
    # The task specifies to use datasets[13] for this experiment
    original_dataset = datasets[13]
    assert datapoints < 10
    # Define the (A, B) pairs as specified in the prompt
    x_axis_configs = []
    for A in Range(2, 2+datapoints): # A ranges from 2 to 10
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
        _, fnames, runtimes_dict = run_all([sub_dataset], scalpvalues= [.55])

        runtimes_dict = runtimes_dict[0]  # Extract runtimes for the single dataset
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
    # print(runtimes_df)
    return  runtimes_df #plotruntimes(runtimes_df)
    # return runtimes_df # Optionally return the dataframe




# def score_time_front(scores, times):
#     '''
#     times is a dict of dataset -> method -> time
#     scores is a dict of dataset -> method -> score dict which has attributes label_mean and batch_mean
#     we need a scatter plot. each method gets a color, and we plot its results for each dataset, x: time, y: geomean of label and batch
#     '''

def score_time_front(scores, times):
    data = []
    for dataset_id, method_scores in scores.items():
        for method_name, score_values in method_scores.items():
            label_mean = score_values.get('label_mean', np.nan)
            batch_mean = score_values.get('batch_mean', np.nan)

            # Calculate geometric mean if both label_mean and batch_mean are available
            geomean_score = gmean([label_mean, batch_mean]) if not np.isnan(label_mean) and not np.isnan(batch_mean) else np.nan

            runtime = times.get(int(dataset_id), {}).get(method_name, np.nan)

            data.append({
                'dataset_id': dataset_id,
                'method': method_name,
                'geomean_score': geomean_score,
                'runtime': runtime
            })

    df = pd.DataFrame(data)

    # Filter out rows with NaN values in geomean_score or runtime
    df = df.dropna(subset=['geomean_score', 'runtime'])

    palette = makepalette(df.method.unique())

    fig, ax = plt.subplots(figsize=(10, 7))

    sns.scatterplot(
        data=df,
        x='runtime',
        y='geomean_score',
        hue='method',
        palette=palette,
        s=100,  # Marker size
        alpha=0.8,
        ax=ax
    )

    ax.set_xlabel('Runtime (seconds)', fontsize=12)
    ax.set_ylabel('Geometric Mean Score', fontsize=12)
    ax.set_title('Integration Method Performance: Score vs. Runtime', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Place legend outside the plot area for better readability
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    # plt.show()
    return fig



def plotruntimes_(runtimes_df):
    palette = makepalette(runtimes_df.method)

    # Plot the results
    # plt.figure(figsize=(12, 7))
    g = sns.lineplot(data=runtimes_df, x='config_label', y='runtime',palette=palette, hue='method', marker='o')

    g.grid(True, linestyle='--', alpha=0.7)
    # plt.title('Integration Method Runtimes Across Dataset Sizes')
    plt.xlabel('Number of Batches x Cells per Batch')
    plt.ylabel('Runtime (seconds)')
    plt.xticks(rotation=45, ha='right')

    plt.tick_params(axis='both', which='both', direction='out', length=4, width=1, colors='black', bottom=True, top=False, left=True, right=False)
    plt.tight_layout()

    # plt.legend(bbox_to_anchor=(0.5, -.3), loc='upper center', ncol=3)
    plt.legend(bbox_to_anchor=(0.5, -0.3), loc='upper center', ncol=3, frameon=False)
    fig = plt.gcf()
    fig.tight_layout()
    plt.show()
    return fig

    # plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    # fig = plt.gcf()
    # plt.show()
    # return fig


def plotruntimes(runtimes_df):
    palette = makepalette(runtimes_df.method.unique())

    plt.figure(figsize=(12, 7)) # Create a figure explicitly for finer control

    # Use 'g' for the Axes object returned by sns.lineplot
    g = sns.lineplot(data=runtimes_df, x='config_label', y='runtime', hue='method',
                     palette=palette, marker='o', linewidth=2.5) # Set consistent linewidth here

    g.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel('Number of Batches x Cells per Batch')
    plt.ylabel('Runtime (seconds)')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Adjust legend position to avoid overlapping with plot elements
    plt.legend(bbox_to_anchor=(0.5, -0.25), loc='upper center', ncol=3) # Moved down

    fig = plt.gcf()
    plt.show()
    return fig





def kni(datasets):
    scores_kni = { str(i): scalp.score.kni_scores(datasets[i], projection='methods')
        for i in lmz.Range(datasets) }

    dff = pd.DataFrame.from_dict(scores_kni, orient='index')
    average_scores = dff.mean()
    average_scores_formatted = average_scores.apply(lambda x: f"{x:.2f}").to_latex()
    print(average_scores_formatted)



def variance_check123(datasets):
    # find the largetst dataset
    # then sample 5x 50% and score it via all the methods (i guess run_all)
    # report a table with mean/var  per method for batch-score label-score and geomean
    pass



def variance_check(datasets):
    # Find the largest dataset by number of observations
    largest_dataset_idx = np.argmax([d.n_obs for d in datasets])
    original_dataset = datasets[largest_dataset_idx]
    print(f"Largest dataset found: {original_dataset.uns['name']} ({original_dataset.n_obs} cells)")

    all_scores = defaultdict(lambda: defaultdict(list))
    num_samples = 5
    sample_fraction = 0.5

    for i in range(num_samples):


        ##############
        #   SUBSAMPLE
        ################
        print(f"Sampling iteration {i+1}/{num_samples}...")
        num_cells_to_sample = int(original_dataset.n_obs * sample_fraction)
        if 'batch' in original_dataset.obs.columns:
            # Stratified sampling by batch to maintain batch proportions
            sampled_indices = []
            for batch in original_dataset.obs['batch'].unique():
                batch_cells = original_dataset.obs_names[original_dataset.obs['batch'] == batch]
                num_to_sample_in_batch = int(len(batch_cells) * sample_fraction)
                sampled_indices.extend(np.random.choice(batch_cells, num_to_sample_in_batch, replace=False))

            if not sampled_indices: # Fallback if stratified sampling yields no cells
                print("Warning: Stratified sampling yielded no cells, falling back to random sampling.")
                assert False
        else:
            assert False
        sub_adata = original_dataset[sampled_indices, :].copy()
        sub_adata.uns = original_dataset.uns.copy()
        # explicitly clean results... just in case
        sub_adata.uns['integrated'] = []
        sub_adata.uns['methods'] = []


        integrated_datasets, fnames, _ = run_all([sub_adata], scalpvalues=[.65])
        current_scores = scalp.score.scalp_scores(integrated_datasets[0], projection = 'methods', label_batch_split=False)

        for method_name, scores_dict in current_scores.items():
            all_scores[method_name]['label_mean'].append(scores_dict['label_mean'])
            all_scores[method_name]['batch_mean'].append(scores_dict['batch_mean'])
            all_scores[method_name]['geomean'].append(gmean([scores_dict['label_mean'], scores_dict['batch_mean']]))

    # Prepare results table
    results = []
    for method, metrics in all_scores.items():
        mean_label = np.mean(metrics['label_mean'])
        std_label = np.std(metrics['label_mean'])
        mean_batch = np.mean(metrics['batch_mean'])
        std_batch = np.std(metrics['batch_mean'])
        mean_geomean = np.mean(metrics['geomean'])
        std_geomean = np.std(metrics['geomean'])

        results.append({
            'Method': method,
            'Label Mean': f"{mean_label:.2f} ± {std_label:.2f}",
            'Batch Mean': f"{mean_batch:.2f} ± {std_batch:.2f}",
            'Geomean': f"{mean_geomean:.2f} ± {std_geomean:.2f}"
        })

    results_df = pd.DataFrame(results).set_index('Method')
    print("\nVariance Check Results (Mean ± Std Dev):")
    print(results_df.to_latex())
    return results_df
