from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from  sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from ubergauss import tools as ut
from ubergauss.optimization import pareto_scores
import numpy as np
from sklearn.metrics import  silhouette_score
from scalp.data import transform
import anndata
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional
import anndata
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import confusion_matrix

def repeat_as_column(a,n):
    return np.tile(a,(n,1)).T

def neighbor_labelagreement(X_train, y_train,n_neighbors):
    knn = KNeighborsClassifier(n_neighbors=n_neighbors+1)


    y_train, _ = ut.labelsToIntList(y_train)
    y_train= np.array(y_train)
    # print(sum(np.isnan(X_train)))
    knn.fit(X_train, y_train)

    # Get the labels of the nearest neighbors for each training point
    _, indices = knn.kneighbors(X_train)
    neighbor_labels = y_train[indices]
    # Exclude the label of the training point itself
    neighbor_labels = neighbor_labels[:, 1:]

    # Compute the training error
    agreement = (repeat_as_column(y_train,n_neighbors) == neighbor_labels).mean()

    return agreement


def scores(data, projectionlabel = 'lsa'):
    y = data.obs['label'].tolist()
    ybatch = data.obs['batch'].tolist()
    sim = data.obsm[projectionlabel]

    score = neighbor_labelagreement(sim,y,5)
    silou = silhouette_score(sim,y)
    batchmix = -neighbor_labelagreement(sim,ybatch,5)
    return score, silou, batchmix


from sklearn.metrics import adjusted_rand_score as ari
def anndata_ari(ad, predicted_label='label', true_label='label'):
    return ari(ad.obs[true_label], ad.obs[predicted_label])


from sklearn.linear_model import SGDClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score





def score_lin(dataset, projection = 'umap'):
    y = dataset.obs['label']
    X = dataset.obsm[projection]
    prediction = SGDClassifier().fit(X,y).predict(X)
    return accuracy_score(y , prediction )

def score_lin_batch(dataset, projection = 'umap'):
    # do this per cell line
    def acc(label):
        instances = dataset.obs['label'] == label
        tmp_dataset = dataset[instances]
        y = tmp_dataset.obs['batch']
        if len(np.unique(y)) < 2:
            return np.nan
        X = tmp_dataset.obsm[projection]
        prediction = SGDClassifier().fit(X,y).predict(X)
        return balanced_accuracy_score(y , prediction, adjusted=True )

    return np.nanmean([1-acc(l) for l in np.unique(dataset.obs['label'])])


from scib.metrics import metrics
def score_scib_metrics(dataset, embed):
    # ds2 = dataset.copy()
    # ds2.X = ds2.obsm['umap']
    # https://scib.readthedocs.io/en/latest/api.html#biological-conservation-metrics
    # embed = 'umap' if 'umap' in dataset.obsm else 'X_umap'
    sc =  metrics(dataset, dataset, 'batch', 'label', embed = embed,
                       isolated_labels_asw_=True, silhouette_=True, hvg_score_=True, graph_conn_=True,
           pcr_=False, #!!!!!!!!!
             isolated_labels_f1_=True,
             trajectory_=False,
             nmi_=True,
            ari_=True )
    res =  dict(dict(sc)[0])
    for k in list(res.keys()):
        if np.isnan(res[k]):
            res.pop(k)
    res.pop('hvg_overlap',0)
    return res

from scipy.stats import gmean
def split_scib_scores(d):
    batchwords = 'PCR_batch ASW_label/batch graph_conn'.split()
    print(d.keys())
    def split(d):
        batch = gmean([v for k,v in d.items() if k in batchwords ])
        bioconservation = gmean([v for k,v in d.items() if not k in batchwords ])
        return bioconservation,batch

    return split(d)



def scib_scores(ds, projection = 'umap'):
    try:
        ds.X = ut.zehidense(ds.X)
        if 'Scalp' in projection:
            ds.obsm['garbo'] = hub.transform(ds.obsm[projection], None)
            projection = 'garbo'

        ds.obsm[projection] = ds.obsm[projection].astype(float)
        sc = score_scib_metrics(ds, projection)
        bio, batch = split_scib_scores(sc)
        return {'batch': batch,  'label':bio}
    except:
        return {'batch': -1,  'label':-1}


# def scalp_scores(ds, projection = 'umap'):
#     return {'batch': score_lin_batch(ds,projection), 'label':score_lin(ds,projection)}

def scalp_scores(data, projection ='integrated', cv=5,label_batch_split= False):
    dataset = data # if type(data)!= list else transform.stack(data)
    y = dataset.obs['label']
    ybatch = dataset.obs['batch']
    ret= {}
    for projection2 in dataset.uns[projection]:
        X = dataset.obsm[projection2]
        print(X.shape, projection2)
        ret[projection2] = getscores(X,y,ybatch,cv, label_batch_split)
    return ret

# def scalpscore(datasets):
#     scr = lambda i: scalp_scores(datasets[i], projection = 'methods', label_batch_split=False)
#     res  = ut.xxmap(scr, Range(datasets))
#     return dict(zip(Range(datasets),res))

# scalp_scores and scalpscore can be unified. write a new scalpscore function. it works the same as the old one but uses ut.xxmap on the gescores level. call xxmap only once!. returns the same as scalpscore.

import json
from datetime import datetime
import os

def scalpscores(datasets, projection='methods', cv=5, label_batch_split=False):
    tasks = [(ds.obsm[p], ds.obs['label'], ds.obs['batch'], cv, label_batch_split, i, p)
             for i, ds in enumerate(datasets)
             for p in ds.uns.get(projection, [])]

    def worker(task):
        X, y, yb, cv, lbs, idx, proj = task
        return idx, proj, getscores(X, y, yb, cv, lbs)

    results = ut.xxmap(worker, tasks)

    out = {i: {} for i in range(len(datasets))}
    for idx, proj, scores in results:
        out[idx][proj] = scores

    fname = f"./tableruns/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(datasets)}.json"
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'w') as f:
        json.dump(out, f)

    return out

def getscores(X,y,ybatch, cv, label_batch_split =False):
    r={}
    m,s = knn_cross_validation(X,y,cv, splitby = ybatch if label_batch_split else None)
    r['label_mean'] =m
    r['label_std'] =s

    m,s = knn_cross_validation(X,ybatch,cv,invert=True)
    r['batch_mean'] =m
    r['batch_std'] =s
    return r


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold


def knn_cross_validation(X, y, cv, invert=False, splitby = None):
    '''
    Perform cross-validation with a Nearest Neighbor classifier.
    '''
    # Initialize the 1-Nearest Neighbor classifier
    knn = KNeighborsClassifier(n_neighbors=5)
    # Perform cross-validation
    if splitby is None:
        kfold = StratifiedKFold(n_splits=cv, shuffle=True)
    else:
        # split by splitby, such that each label  in splitby is in turn the test set
        unique_groups = np.unique(splitby)
        kfold = []
        for group in unique_groups:
            test_indices = np.where(splitby == group)[0]
            train_indices = np.where(splitby != group)[0]
            # Ensure test set is not empty
            if len(test_indices) > 0:
                kfold.append((train_indices, test_indices))

    scores = cross_val_score(knn, X, y, cv=kfold, scoring='balanced_accuracy')
    # Calculate mean and standard deviation of the scores
    if invert:
        scores = 1-scores
    mean_accuracy = np.mean(scores)
    std_accuracy = np.std(scores)
    return mean_accuracy, std_accuracy


def pareto_avg(datadicts):
    '''
    datadicts: a dictionary of datasets, each containing a dictionary of methods with their scores
    returns: a DataFrame with the average rank of each method across all datasets
    '''
    # collect scores for all the methods
    d = []
    for i, dataset in enumerate(datadicts.keys()):
        for method, domcount in pareto_sample(datadicts[dataset]):
            d.append({'method':method,'dataset':i,'domcount':domcount})

    return calculate_average_rank(pd.DataFrame(d),'dataset','domcount','method'), pd.DataFrame(d)

def calculate_average_rank(df, group_column, rank_column, name_column):
    """
    Groups the DataFrame by `group_column`, assigns ranks within each group
    based on `rank_column`, and returns the overall average rank for each name in `name_column`.

    Parameters:
    - df: pd.DataFrame - The input DataFrame.
    - group_column: str - The column name to group by.
    - rank_column: str - The column name to rank within each group.
    - name_column: str - The column name that contains the names of the attributes being ranked.

    Returns:
    - pd.DataFrame - A DataFrame with the overall average rank for each unique name.
    """

    # Step 1: Group by `group_column` and assign rank within each group
    df['rank'] = df.groupby(group_column)[rank_column].rank()

    # Step 2: Calculate the average rank within each group for each name
    group_avg_rank = df.groupby([group_column, name_column])['rank'].mean().reset_index()

    # Step 3: Calculate the overall average rank for each name across all groups
    overall_avg_rank = group_avg_rank.groupby(name_column)['rank'].mean().reset_index()

    # Step 4: Rename columns for clarity
    overall_avg_rank.columns = [name_column, 'overall_average_rank']

    return overall_avg_rank


def pareto_sample(datadict,scorenames=['label','batch']):
    '''
    datadict: a dictionary of methods with their scores
    scorenames: the names of the scores to consider
    returns a list of tuples (method,domcount) where domcount is the number of times the method is dominated
    '''
    methods = datadict.keys()
    sample = lambda method,name,repeats: [{'method': method, 'scoretype':name, 'score': score, 'measureid':i}\
                            for i, score in enumerate(np.random.normal(loc=datadict[method][name+'_mean'], scale=datadict[method][name+'_std'], size=repeats))]

    data =  [ sample(method,name,1000)   for method in methods for name in scorenames]
    data = Flatten(data)
    df = pd.DataFrame(data)
    # {'method': '0', 'scoretype': 'label', 'score': 0.8031946991682309}
    return [(a,b) for a,b in pareto_scores(df, data= 'measureid', scoretype='scoretype')]



import ubergauss.hubness as hub
def kni_scores(data, projection ='integrated', cv=5,label_batch_split= False):
    dataset = data # if type(data)!= list else transform.stack(data)
    ret= {}
    for projection2 in dataset.uns[projection]:
        if 'Scalp' not in projection2:
            ret[projection2] = kni_score(data,projection2)['kni_score']
        else:
            projection3 = hub.transform(data.obsm[projection2], None)
            data.obsm['garbo'] = projection3
            ret[projection2] = kni_score(data,'garbo')['kni_score']
    return ret




def scale_embeddings_numpy(X: np.ndarray) -> np.ndarray:
    """Scales embeddings using quantile normalization natively in NumPy (much faster)"""
    # Center by mean
    X_mean = np.mean(X, axis=0)
    X_centered = X - X_mean

    # Calculate quantiles
    q1 = np.percentile(X_centered, 25, axis=0)
    q2 = np.percentile(X_centered, 75, axis=0)

    iqr = q2 - q1
    # Prevent division by zero
    iqr[iqr == 0] = 1.0

    return X_centered / iqr

def kni_score(
        adata: anndata.AnnData,
        embedding_key: str,
        label_key: str = 'label',
        batch_key: str = 'batch',
        n_neighbours: int = 10,
        max_prop_same_batch: float = 0.8
        ) -> dict:
    """Calculates KNI score directly off NumPy embeddings for speed."""

    # 1. Handle Obs data
    obs_df = adata.obs[[label_key, batch_key]].copy()

    if obs_df[label_key].isna().sum() > 0:
        obs_df[label_key] = obs_df[label_key].astype(str).fillna("Unknown")

    valid_mask = np.ones(len(obs_df), dtype=bool)


    # 2. Extract strictly valid data as Native NumPy Arrays
    X = ut.zehidense(adata.obsm[embedding_key][valid_mask])
    obs_df = obs_df.iloc[valid_mask].copy()

    assert X.shape[0] == obs_df.shape[0], "Mismatch in number of cells after filtering."

    # Fast NumPy scaling
    X_scaled = scale_embeddings_numpy(X)

    # Convert to categories to get integer codes
    obs_df[label_key] = obs_df[label_key].astype('category')
    obs_df[batch_key] = obs_df[batch_key].astype('category')

    # Extract integer arrays for fast operations
    y_true = obs_df[label_key].cat.codes.values
    b_true = obs_df[batch_key].cat.codes.values

    assert -1 not in y_true, "N/A cell type found in label column"

    # 3. Fit NearestNeighbors directly on the array
    nn = NearestNeighbors(n_neighbors=n_neighbours, algorithm='auto')
    nn.fit(X_scaled)
    _, indices = nn.kneighbors(X_scaled)

    # 4. Map neighbors to their cell types and batches
    knn_ct = y_true[indices]           # shape (N, k)
    knn_batch = b_true[indices]        # shape (N, k)
    batch_mat = b_true[:, None]        # shape (N, 1)

    # 5. Fast Mask Calculations
    not_same_batch_mask = knn_batch != batch_mat
    num_same_batch = np.sum(~not_same_batch_mask, axis=1)
    diverse_neighbourhood_mask = num_same_batch < (max_prop_same_batch * n_neighbours)

    # 6. Fast Loop over NumPy arrays (No Pandas .iloc!)
    N = X_scaled.shape[0]
    pred_all = np.zeros(N, dtype=int)
    pred_kni = np.zeros(N, dtype=int) - 1 # -1 acts as null prediction

    for i in range(N):
        # Predict using all neighbors
        pred_all[i] = np.bincount(knn_ct[i]).argmax()

        # Predict using only non-batch neighbors if neighborhood is diverse
        if diverse_neighbourhood_mask[i]:
            valid_ct = knn_ct[i][not_same_batch_mask[i]]
            if len(valid_ct) > 0:
                pred_kni[i] = np.bincount(valid_ct).argmax()

    # 7. Calculate overall scores directly from boolean arrays
    acc_correct = y_true == pred_all
    kni_correct = (y_true == pred_kni) & diverse_neighbourhood_mask

    acc_total = acc_correct.mean()
    kni_total = kni_correct.mean()
    diverse_pass_total = diverse_neighbourhood_mask.mean()

    # 8. Grouped Stats using fast Pandas aggregation
    res_df = pd.DataFrame({
        'batch_name': obs_df[batch_key],
        'acc_count_knn': acc_correct.astype(int),
        'kni_count': kni_correct.astype(int),
        'diverse_pass_count_knn': diverse_neighbourhood_mask.astype(int),
        'batch_count_knn': 1
    })

    # Calculate batch-level metrics
    results_by_batch = res_df.groupby('batch_name', observed=False).sum()
    results_by_batch['batch_name'] = results_by_batch.index
    results_by_batch['kni_batch'] = results_by_batch['kni_count'] / results_by_batch['batch_count_knn']
    results_by_batch['acc_knn'] = results_by_batch['acc_count_knn'] / results_by_batch['batch_count_knn']
    results_by_batch['diverse_pass_knn'] = results_by_batch['diverse_pass_count_knn'] / results_by_batch['batch_count_knn']

    # 9. Fast Confusion Matrices using scikit-learn
    labels_cats = obs_df[label_key].cat.categories

    acc_conf_mat = confusion_matrix(y_true, pred_all, labels=range(len(labels_cats)))
    acc_conf_df = pd.DataFrame(acc_conf_mat, index=labels_cats, columns=labels_cats)

    kni_mask = pred_kni != -1
    kni_conf_mat = confusion_matrix(y_true[kni_mask], pred_kni[kni_mask], labels=range(len(labels_cats)))
    kni_conf_df = pd.DataFrame(kni_conf_mat, index=labels_cats, columns=labels_cats)

    return {
        'acc_knn': acc_total,
        'kni_score': kni_total,   # <-- This is the actual KNI score
        'mean_pct_same_batch_in_knn': np.mean(num_same_batch) / n_neighbours,
        'pct_cells_with_diverse_knn': diverse_pass_total,
        'confusion_matrix': acc_conf_df,
        'kni_confusion_matrix': kni_conf_df,
        'results_by_batch': results_by_batch,
    }


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.neighbors import NearestNeighbors
import umap


def plot_kni(
    adata,
    embedding_keys: list,
    label_key: str = 'label',
    batch_key: str = 'batch',
    n_neigh: int = 50,
    max_prop_same_batch: float = 0.8
):
    """
    Plots a diagnostic scatter for KNI scores across different embeddings.

    Colors:
    - Gray: Fails diversity check (surrounded by too much of its own batch).
    - Black: Passes diversity check, but predicts the WRONG cell type using non-batch neighbors.
    - Tab20 Colors: Passes diversity check, and predicts the CORRECT cell type.
    """

    # 1. Labels and batches (No 'Unknown' filtering)
    obs_df = adata.obs[[label_key, batch_key]].copy()
    obs_df[label_key] = obs_df[label_key].astype('category')
    obs_df[batch_key] = obs_df[batch_key].astype('category')

    y_true = obs_df[label_key].cat.codes.values
    b_true = obs_df[batch_key].cat.codes.values
    N = len(y_true)

    # Setup Colors
    cmap = plt.get_cmap('tab20')
    unique_labels = np.unique(y_true)
    base_colors = {code: mcolors.to_hex(cmap(i % 20)) for i, code in enumerate(unique_labels)}

    # Setup Plot Grid
    fig, axes = plt.subplots(1, len(embedding_keys), figsize=(6 * len(embedding_keys), 6))
    if len(embedding_keys) == 1:
        axes = [axes]

    # 2. Iterate over embedding keys
    for ax, emb_key in zip(axes, embedding_keys):

        # --- PLOTTING COORDINATES (2D Check) ---
        X_orig = adata.obsm[emb_key]

        # Convert sparse or numpy.matrix to standard numpy.ndarray to prevent math errors
        if hasattr(X_orig, "toarray"):
            X_orig = X_orig.toarray()
        X_orig = np.asarray(X_orig)

        if X_orig.shape[1] > 2:
            # Create a 2D layer specifically for this embedding if it doesn't exist
            layer_2d_name = f'{emb_key}_2d'
            if layer_2d_name not in adata.obsm:
                print(f"Calculating UMAP for {emb_key}...")
                adata.obsm[layer_2d_name] = umap.UMAP(n_components=2).fit_transform(X_orig)

            X_plot = adata.obsm[layer_2d_name]
            if hasattr(X_plot, "toarray"):
                X_plot = X_plot.toarray()
            X_plot = np.asarray(X_plot)
        else:
            X_plot = X_orig

        # --- KNI EVALUATION (High-D Space) ---
        # Scale original embeddings for neighbor search (fast NumPy method)
        X_mean = np.mean(X_orig, axis=0)
        X_centered = X_orig - X_mean
        q1 = np.percentile(X_centered, 25, axis=0)
        q2 = np.percentile(X_centered, 75, axis=0)
        iqr = q2 - q1
        iqr[iqr == 0] = 1.0
        X_scaled = X_centered / iqr

        # Find neighbors
        nn = NearestNeighbors(n_neighbors=n_neigh, algorithm='auto')
        nn.fit(X_scaled)
        _, indices = nn.kneighbors(X_scaled)

        knn_ct = y_true[indices]
        knn_batch = b_true[indices]
        batch_mat = b_true[:, None]

        # Calculate diversity masks
        not_same_batch_mask = knn_batch != batch_mat
        num_same_batch = np.sum(~not_same_batch_mask, axis=1)
        diverse_mask = num_same_batch < (max_prop_same_batch * n_neigh)

        # Categorize cells into the 3 plotting groups
        correct_mask = np.zeros(N, dtype=bool)
        wrong_mask = np.zeros(N, dtype=bool)

        for i in range(N):
            if diverse_mask[i]:
                # Get cell types of neighbors from OTHER batches
                valid_ct = knn_ct[i][not_same_batch_mask[i]]
                if len(valid_ct) > 0:
                    pred = np.bincount(valid_ct).argmax()
                    if pred == y_true[i]:
                        correct_mask[i] = True
                    else:
                        wrong_mask[i] = True
                else:
                    wrong_mask[i] = True

        low_diversity_mask = ~diverse_mask

        # --- DRAW PLOT ---
        # Group 1: Low Diversity (Gray)
        ax.scatter(X_plot[low_diversity_mask, 0], X_plot[low_diversity_mask, 1],
                   c='#D3D3D3', s=4, edgecolors='none', alpha=1.0)

        # Group 2: Wrong Prediction (Black)
        ax.scatter(X_plot[wrong_mask, 0], X_plot[wrong_mask, 1],
                   c='black', s=4, edgecolors='none', alpha=1.0)

        # Group 3: Correct Prediction (Tab20 Colors)
        colors_correct = [base_colors[y] for y in y_true[correct_mask]]
        ax.scatter(X_plot[correct_mask, 0], X_plot[correct_mask, 1],
                   c=colors_correct, s=4, edgecolors='none', alpha=1.0)

        # Formatting
        ax.set_title(f'{emb_key}', fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
