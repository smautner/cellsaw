from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from scalp.data import transform
from scalp import data, pca, umapwrap, mnn, graph, test_config
from scalp import diffuse
from scalp.output import score
from scalp.data.align import align
from scalp.output.draw import snsplot
import scanpy as sc




def construct_sparse_adjacency_matrix_multiple(matrices, k, h):
    """
    Constructs a sparse adjacency matrix based on k-NN within each matrix,
    adds cross-edges using linear assignment pairwise between matrices,
    and filters edges based on horizon h.

    Parameters:
    - matrices: list of numpy arrays, each of shape (n_i, d)
    - k: int, number of nearest neighbors within each matrix
    - h: int, horizon parameter for filtering edges

    Returns:
    - adjacency: scipy.sparse.csr_matrix of shape (total_instances, total_instances)
    """
    from scipy.sparse import csr_matrix
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    # Construct k-NN within each matrix
    knn = []
    for matrix in matrices:
        tree = cKDTree(matrix)
        dists, neighs = tree.query(matrix, k=k+1)
        knn.append(neighs[:,1:])
    # Construct cross-edges using linear assignment pairwise between matrices
    n = sum([len(neighs) for neighs in knn])
    row, col, data = [], [], []
    for i, neighs1 in enumerate(knn):
        for j, neighs2 in enumerate(knn):
            if i == j:
                continue
            cost = cdist(matrices[i], matrices[j][neighs2.flatten()])
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind):
                row.append(neighs1[r] + i)
                col.append(neighs2[c] + j)
                data.append(1)
    row = np.concatenate(row)
    col = np.concatenate(col)
    data = np.concatenate(data)
    # Construct adjacency matrix
    adjacency = csr_matrix((data, (row, col)), shape=(n, n))
    # Filter edges based on horizon h
    for i in range(n):
        row = adjacency[i].indices
        col = adjacency[:,i].indices
        row = row[np.abs(row - i) <= h]
        col = col[np.abs(col - i) <= h]
        adjacency[i, row] = 1
        adjacency[col, i] = 1
    return adjacency





# horizonCutoff 4 10 1 # the idea is flawed
mkgraphParameters = '''
neighbors_total 15 45 1
neighbors_intra_fraction .2 .5
intra_neighbors_mutual 0 1 1
add_tree 0 1 1
horizonCutoff 50 100 1
standardize 0 1 1
'''
# copy_lsa_neighbors 0 1 1
# distance_metric ['euclidean', 'sqeuclidean' ]
# inter_outlier_threshold .60 .97
# inter_outlier_probabilistic_removal 0 1 1

from sklearn.preprocessing import StandardScaler

def mkgraph( adata ,pre_pca = 40,
            horizonCutoff = 0,
            neighbors_total = 20, neighbors_intra_fraction = .5,
              scaling_num_neighbors = 2, inter_outlier_threshold = -1,
            distance_metric = 'euclidean',
                inter_outlier_probabilistic_removal= False,
            epsilon = 1e-6,standardize=0,
                intra_neighbors_mutual = False, copy_lsa_neighbors = False,
              add_tree= False, dataset_adjacency = None, **kwargs ):
    '''
    this does our embedding,
    '''
    # adatas = pca.pca(adatas,dim = pre_pca, label = 'pca40')
    assert 'pca40' in adata.obsm

    if horizonCutoff:
        inter_outlier_threshold = 0
        inter_outlier_probabilistic_removal = False

    if standardize == 0: # no standardization
        adatas = data.transform.split_by_obs(adata)
    elif standardize == 1: # joint
        sc.pp.scale(adata)
        adatas = data.transform.split_by_obs(adata)
    elif standardize == 2: # separate
        adatas = data.transform.split_by_obs(adata)
        [sc.pp.scale(a) for a in adatas]
    else:
        assert False, f"unknown standardize value {standardize=}"


    if False:#aislop
        matrix = graph.aiSlopSolution(adatas, 20, 240)
    else:
        matrix = graph.linear_assignment_integrate(adatas,base = 'pca40',
                                                    neighbors_total=neighbors_total,
                                                distance_metric=distance_metric,
                                horizonCutoff = horizonCutoff,
                                                    neighbors_intra_fraction=neighbors_intra_fraction,
                                                      intra_neighbors_mutual=intra_neighbors_mutual,
                                                      outlier_probabilistic_removal= inter_outlier_probabilistic_removal,
                                                      scaling_num_neighbors = scaling_num_neighbors,
                                                      outlier_threshold = inter_outlier_threshold,
                                                      dataset_adjacency =  dataset_adjacency,
                                                      copy_lsa_neighbors=copy_lsa_neighbors,
                                                   epsilon=epsilon,
                                                  add_tree=add_tree)
    #data = umapwrap.graph_umap(data, matrix, label = 'graphumap')
    if False: # debug
        from scipy.sparse import csr_matrix
        import structout as so
        matrix2 = csr_matrix(matrix)
        vals = [ len(x.data) for x in matrix2]
        print(f"will plot the number of neighbors for each item... {min(vals)=},{max(vals)=}")
        so.lprint(vals)
    return  matrix



# diffuse.diffuse_label  -> diffuses the label

def graph_embed_plot(dataset,matrix, embed_label= 'embedding', snskwargs={}):
    dataset = umapwrap.graph_umap(dataset,matrix,label = embed_label)
    snsplot(dataset,coordinate_label=embed_label,**snskwargs)
    return dataset

import umap
import matplotlib.pyplot as plt
import seaborn as sns
def plot(adata,embedding,**plotargs):
    # adata.obsm['X_umap']=adata.obsm[embedding]
    # sc.pl.umap(adata,basis= embedding, **plotargs)

    if adata.obsm[embedding].shape[1] > 2:
        adata.obsm['newlayer'] =  umap.UMAP(n_components = 2).fit_transform(adata.obsm[embedding])
    else:
        adata.obsm['newlayer'] =  adata.obsm[embedding]
    title =  adata.uns.get('name','no name found in adata')
    # ax = sc.pl.embedding(adata, basis= 'newlayer' , show = False,**plotargs)
    plot_embedding_with_labels(adata, basis='newlayer', title=title)


def plot_embedding_with_labels(adata, basis='', title = ''):
    X = adata.obsm[basis]
    batch = adata.obs['batch'].astype(str)
    label = adata.obs['label'].astype(str)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6)) # Increased figure width for legend

    # First scatterplot: color by 'group'
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=batch,palette = 'viridis' , ax=ax1, s=16) # 's' controls marker size
    ax1.set_title('Batch')
    ax1.set_xlabel('') # Remove x-axis label
    ax1.set_ylabel('') # Remove y-axis label
    ax1.set_xticks([]) # Remove x-axis ticks
    ax1.set_yticks([]) # Remove y-axis ticks
    ax1.spines['top'].set_visible(False) # Remove top spine (box)
    ax1.spines['right'].set_visible(False) # Remove right spine
    ax1.spines['bottom'].set_visible(False) # Remove bottom spine
    ax1.spines['left'].set_visible(False) # Remove left spine
    leg1 = ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='') # Legend to the right
    leg1.get_frame().set_linewidth(0.0) # Remove box around legend

    # Second scatterplot: color by 'label'
    sns.scatterplot(x=X[:,0], y=X[:,1], hue=label, ax=ax2, s=16) # 's' controls marker size
    ax2.set_title('Label')
    ax2.set_xlabel('') # Remove x-axis label
    ax2.set_ylabel('') # Remove y-axis label
    ax2.set_xticks([]) # Remove x-axis ticks
    ax2.set_yticks([]) # Remove y-axis ticks
    ax2.spines['top'].set_visible(False) # Remove top spine (box)
    ax2.spines['right'].set_visible(False) # Remove right spine
    ax2.spines['bottom'].set_visible(False) # Remove bottom spine
    ax2.spines['left'].set_visible(False) # Remove left spine
    leg2 = ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='') # Legend to the right
    leg2.get_frame().set_linewidth(0.0) # Remove box around legend

    plt.suptitle(title, y=0.95, fontsize = 16) # Adjust suptitle position if needed
    plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legends
    plt.show()



def plot2(adata, embedding1, embedding2, **plotargs):
    """
    Plots two embeddings horizontally, each with a 'batch' and 'label' subplot.
    A single combined legend is placed below the entire figure.

    Parameters:
    - adata: AnnData object
    - embedding1: str, name of the first embedding in adata.obsm
    - embedding2: str, name of the second embedding in adata.obsm
    - plotargs: additional arguments passed to seaborn.scatterplot
    """
    X1 = adata.obsm[embedding1]
    X2 = adata.obsm[embedding2]

    # Reduce to 2D if embeddings are higher dimension
    if X1.shape[1] > 2:
        X1 = umap.UMAP(n_components=2, random_state=42).fit_transform(X1)
    if X2.shape[1] > 2:
        X2 = umap.UMAP(n_components=2, random_state=42).fit_transform(X2)

    batch_labels = adata.obs['batch'].astype(str)
    cell_labels = adata.obs['label'].astype(str)

    # Determine unique categories for combined legend
    all_batches = sorted(batch_labels.unique())
    all_labels = sorted(cell_labels.unique())
    all_categories = all_batches + all_labels

    # Create a custom palette for batch and label to ensure consistency
    # using 'tab20' for labels and a custom viridis-like for batches
    n_batches = len(all_batches)
    n_labels = len(all_labels)

    # Use seaborn's color palettes
    batch_palette = sns.color_palette("viridis", n_batches)
    label_palette = sns.color_palette("tab20", n_labels) if n_labels <= 20 else sns.color_palette("hls", n_labels)


    # Map categories to colors
    color_map = {}
    for i, batch in enumerate(all_batches):
        color_map[batch] = batch_palette[i]
    for i, label in enumerate(all_labels):
        color_map[label] = label_palette[i]

    # Set up the figure and a GridSpec for plots and a shared legend
    fig = plt.figure(figsize=(24, 8)) # Increased width, height adjusted for legend
    gs = fig.add_gridspec(2, 4, height_ratios=[8, 1]) # 2 rows, 4 cols for plots, 2nd row for legend

    axs = [fig.add_subplot(gs[0, i]) for i in range(4)] # First row for the four plots

    def _plot_panel(ax, x_data, y_data, hue_data, title, current_palette):
        sns.scatterplot(x=x_data, y=y_data, hue=hue_data, palette=current_palette, ax=ax, s=16, legend=False, **plotargs)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Plot 1: Embedding1 by Batch
    _plot_panel(axs[0], X1[:, 0], X1[:, 1], batch_labels, f'{embedding1} - Batch', batch_palette)

    # Plot 2: Embedding1 by Label
    _plot_panel(axs[1], X1[:, 0], X1[:, 1], cell_labels, f'{embedding1} - Label', label_palette)

    # Plot 3: Embedding2 by Batch
    _plot_panel(axs[2], X2[:, 0], X2[:, 1], batch_labels, f'{embedding2} - Batch', batch_palette)

    # Plot 4: Embedding2 by Label
    _plot_panel(axs[3], X2[:, 0], X2[:, 1], cell_labels, f'{embedding2} - Label', label_palette)

    # Create a dummy subplot for the combined legend
    legend_ax = fig.add_subplot(gs[1, :]) # Span all columns in the second row
    legend_ax.axis('off') # Hide this axes

    # Create the combined legend
    handles = []
    labels = []

    # Add batch categories to legend
    for bid , batch in enumerate(all_batches):
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Batch: {bid}',
                                  markerfacecolor=color_map[batch], markersize=10))
        labels.append(f'Batch: {bid}')

    # Add cell label categories to legend
    for label in all_labels:
        handles.append(plt.Line2D([0], [0], marker='o', color='w', label=f'Label: {label}',
                                  markerfacecolor=color_map[label], markersize=10))
        labels.append(f'Label: {label}')

    legend_ax.legend(handles=handles, labels=labels, loc='center', ncol = 7,#ncol=max(1, len(all_batches + all_labels) // 5),
                     fontsize='small', frameon=False)

    plt.suptitle(adata.uns.get('name', 'Embedding Comparison'), y=0.98, fontsize=18)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for suptitle and legend
    plt.show()


def plot2_2x2(adata, embedding1, embedding2, **plotargs):
    '''
    same as plot4 but we want it as 2x2
    '''


    X1 = adata.obsm[embedding1]
    X2 = adata.obsm[embedding2]

    if X1.shape[1] > 2:
        X1 = umap.UMAP(n_components=2, random_state=42).fit_transform(X1)
    if X2.shape[1] > 2:
        X2 = umap.UMAP(n_components=2, random_state=42).fit_transform(X2)

    batch_labels = adata.obs['batch'].astype(str)
    cell_labels = adata.obs['label'].astype(str)

    all_batches = sorted(batch_labels.unique())
    all_labels = sorted(cell_labels.unique())
    n_batches = len(all_batches)
    n_labels = len(all_labels)

    batch_palette = sns.color_palette("viridis", n_batches)
    label_palette = sns.color_palette("tab20", n_labels) if n_labels <= 20 else sns.color_palette("hls", n_labels)

    color_map = {}
    for i, batch in enumerate(all_batches):
        color_map[batch] = batch_palette[i]
    for i, label in enumerate(all_labels):
        color_map[label] = label_palette[i]

    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[8, 8, 1])

    axs = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
           fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]


    def _plot_panel(ax, x_data, y_data, hue_data, title, current_palette):
        sns.scatterplot(x=x_data, y=y_data, hue=hue_data, palette=current_palette, ax=ax, s=16, legend=False, **plotargs)
        ax.set_title(title)
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    _plot_panel(axs[0], X1[:, 0], X1[:, 1], batch_labels, f'{embedding1} - Batch', batch_palette)
    _plot_panel(axs[1], X1[:, 0], X1[:, 1], cell_labels, f'{embedding1} - Label', label_palette)
    _plot_panel(axs[2], X2[:, 0], X2[:, 1], batch_labels, f'{embedding2} - Batch', batch_palette)
    _plot_panel(axs[3], X2[:, 0], X2[:, 1], cell_labels, f'{embedding2} - Label', label_palette)

    legend_ax = fig.add_subplot(gs[2, :])
    legend_ax.axis('off')

    handles = []
    labels = []

    for bid, batch in enumerate(all_batches):
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color_map[batch], markersize=10))
        labels.append(f'Batch: {bid}')

    for label in all_labels:
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=color_map[label], markersize=10))
        labels.append(f'Label: {label}')




    labels = [lbl.split('_')[0] if '_' in lbl else lbl for lbl in labels]

    legend_ax.legend(handles=handles, labels=labels, loc='center', ncol=4, #max(1, (len(all_batches) + len(all_labels)) // 5),
                     fontsize='small', frameon=False)

    plt.suptitle(adata.uns.get('name', 'Embedding Comparison'), y=0.98, fontsize=18)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    return fig





def plot4(adatas:list, label='label', embedding_key = 'scalp', headline = 'someplot', plotnames = 'ABCD'):
    '''
    plot the 4 adatas in a row, nothing fancy, legendplacement might be like plot2
    '''

    num_adatas = len(adatas)

    # Collect all unique labels across all datasets for a combined legend
    all_unique_labels = sorted(list(set(cat for adata in adatas for cat in adata.obs[label].astype(str).unique())))
    n_labels = len(all_unique_labels)

    # Use a consistent color palette
    if n_labels <= 20:
        label_palette = sns.color_palette("tab20", n_labels)
    else:
        label_palette = sns.color_palette("hls", n_labels)

    label_to_color = {lbl: color for lbl, color in zip(all_unique_labels, label_palette)}

    # Create figure and GridSpec for plots and a shared legend
    fig_width_per_plot = 6
    fig_height = 7 # Adjust height for legend below plots
    fig = plt.figure(figsize=(num_adatas * fig_width_per_plot, fig_height))
    gs = fig.add_gridspec(2, num_adatas, height_ratios=[8, 1])

    # Plot each AnnData object
    for i, adata in enumerate(adatas):
        X = adata.obsm[embedding_key]
        if X.shape[1] > 2:
            # Reduce to 2D using UMAP if embedding is higher dimension
            X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X)
        else:
            X_2d = X

        current_labels = adata.obs[label].astype(str)

        ax = fig.add_subplot(gs[0, i]) # Plot in the first row
        sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=current_labels,
                        palette=label_to_color, ax=ax, s=16, legend=False)

        ax.set_title(plotnames[i])
        # ax.set_title(adata.uns.get('name', f'Dataset {i+1}'))
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Create a dummy subplot for the combined legend
    legend_ax = fig.add_subplot(gs[1, :]) # Span all columns in the second row
    legend_ax.axis('off') # Hide axes for the legend

    # Create the combined legend handles and labels
    handles = []
    labels = []
    # can we not display the acutal labels, but say cell type n"?  yust output the code that i need to put here!

    for idx, lbl in enumerate(all_unique_labels):
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=label_to_color[lbl],
                                  markersize=10))
        labels.append(f"Cell Type {idx+1}") # Changed to "Cell Type n"


    # for lbl in all_unique_labels:
    #     handles.append(plt.Line2D([0], [0], marker='o', color='w',
    #                               markerfacecolor=label_to_color[lbl],
    #                               markersize=10))
    #     labels.append(lbl)

    legend_ax.legend(handles=handles, labels=labels, loc='center',
                     ncol= 5,
                     fontsize='small', frameon=False)

    plt.suptitle(f"{headline}", y=0.98, fontsize=16) # Main title for the figure
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for suptitle and legend
    plt.show()



def plot4_2x2(adatas:list, label='label', embedding_key = 'scalp', headline = 'someplot', plotnames = 'ABCD', colors = 0):
    '''
    same as plot4 but we want it as 2x2
    '''




    num_adatas = len(adatas)
    if num_adatas != 4:
        raise ValueError("This function is designed for exactly 4 AnnData objects.")

    all_unique_labels = sorted(list(set(cat for adata in adatas for cat in adata.obs[label].astype(str).unique())))
    n_labels = len(all_unique_labels)




    if colors == 0: # Viridis for batches, Tab20/hls for labels
        batch_palette_colors = sns.color_palette("viridis", n_labels) # Re-using viridis for general labels here
        label_to_color = {lbl: color for lbl, color in zip(all_unique_labels, batch_palette_colors)}
    else: # Tab20/hls for labels
        label_palette_colors = sns.color_palette("tab20", n_labels) if n_labels <= 20 else sns.color_palette("hls", n_labels)
        label_to_color = {lbl: color for lbl, color in zip(all_unique_labels, label_palette_colors)}


    fig = plt.figure(figsize=(6, 6)) # Adjusted for 2x2 plots + legend
    gs = fig.add_gridspec(3, 2, height_ratios=[8, 8, 1]) # 2x2 plots, 1 row for legend

    plot_indices = [(0, 0), (0, 1), (1, 0), (1, 1)] # Grid positions for 2x2



    for i, adata in enumerate(adatas):
        X = adata.obsm[embedding_key]
        X_2d = umap.UMAP(n_components=2, random_state=42).fit_transform(X) if X.shape[1] > 2 else X
        current_labels = adata.obs[label].astype(str)

        ax = fig.add_subplot(gs[plot_indices[i]]) # Place plot in correct grid cell
        sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=current_labels,
                        palette=label_to_color, ax=ax, s=16, legend=False)

        ax.set_title(plotnames[i])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    legend_ax = fig.add_subplot(gs[2, :]) # Span both columns in the last row for the legend
    legend_ax.axis('off')

    handles = []
    labels = []
    for idx, lbl in enumerate(all_unique_labels):
        handles.append(plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor=label_to_color[lbl],
                                  markersize=10))
        labels.append(f"Cell Type {idx+1}")

    legend_ax.legend(handles=handles, labels=labels, loc='center',
                     ncol=5, fontsize='small', frameon=False)

    plt.suptitle(f"{headline}", y=0.98, fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()
    return fig




def test_scalp():
    n_cells = 100
    a = data.scib(test_config.scib_datapath, maxdatasets=3,
                           maxcells = n_cells, datasets = ["Immune_ALL_hum_mou"]).__next__()
    # print("=============== mnn ===============")
    # mnn and scanvi are no longer maintained, scanoram is second on the nature method ranking
    # a = mnn.mnn(a)
    # print(f"{ a[0].obsm[f'mnn'].shape= }")

    print("=============== PCA ===============")
    a = pca.pca(a)
    print(f"{a[0].obsm['pca40'].shape = }")
    assert a[0].obsm['pca40'].shape == (n_cells,40)
    align(a,'pca40')

    print("=============== scanorama ===============")
    a = mnn.scanorama(a)
    print(f"{ a[0].obsm[f'scanorama'].shape= }")

    print("=============== umap ===============")
    a = umapwrap.adatas_umap(a,label= 'umap10')
    print(f"{ a[0].obsm['umap10'].shape= }")
    assert a[0].obsm['umap10'].shape == (n_cells,10)

    print("=============== make lina-graph ===============")
    # matrix = graph.linear_assignment_integrate(a, base ='pca40')
    matrix = graph.integrate(a, base ='pca40')
    print(f"{matrix.shape=}")
    assert matrix.shape== (n_cells*3,n_cells*3)

    print("=============== diffuse label ===============")
    a = diffuse.diffuse_label(a, matrix, use_labels_from_dataset_ids=[2, 1], new_label ='difflabel')
    #print(f"{type(a[0].obs['difflabel'])=}")
    print(f"{a[0].obs['difflabel'].shape=}")
    assert a[0].obs['difflabel'].shape== (n_cells,)
    print(f"{Map(score.anndata_ari, a, predicted_label='difflabel')=}")

    print("=============== sklearn diffusion ===============")
    a = diffuse.diffuse_label_sklearn(a, use_labels_from_dataset_ids=[2, 1], new_label ='skdiff')
    print(f"{a[0].obs['skdiff'].shape=}")
    assert a[0].obs['skdiff'].shape== (n_cells,)

    print("=============== lina-graph umap ===============")
    a = umapwrap.graph_umap(a,matrix, label = 'graphumap')
    print(f"{ a[0].obsm['graphumap'].shape= }")
    assert a[0].obsm['graphumap'].shape== (n_cells,2)



