from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from  sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from ubergauss import tools as ut
from ubergauss.optimization import pareto_scores
import numpy as np
from sklearn.metrics import  silhouette_score
from scalp.data import transform

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
def score_scib_metrics(dataset):
    # ds2 = dataset.copy()
    # ds2.X = ds2.obsm['umap']
    # https://scib.readthedocs.io/en/latest/api.html#biological-conservation-metrics
    embed = 'umap' if 'umap' in dataset.obsm else 'X_umap'
    sc =  metrics(dataset, dataset, 'batch', 'label', embed = embed,
                       isolated_labels_asw_=True, silhouette_=True, hvg_score_=True, graph_conn_=True,
           pcr_=True,
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


def split_scib_scores(d):
    batchwords = 'PCR_batch ASW_label/batch graph_conn'.split()

    def split(d):
        batch = np.mean([v for k,v in d.items() if k in batchwords ])
        bioconservation = np.mean([v for k,v in d.items() if not k in batchwords ])
        return bioconservation,batch

    return split(d)


def scib_scores(ds, projection = 'umap'):
    sc = score_scib_metrics(ds)
    bio, batch = split_scib_scores(sc)
    return {'batch': batch,  'label':bio}


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



import anndata
from scipy.stats import mode
from sklearn.neighbors import NearestNeighbors
from typing import Dict, List, Optional


def kni_scores(data, projection ='integrated', cv=5,label_batch_split= False):
    dataset = data # if type(data)!= list else transform.stack(data)
    ret= {}
    for projection2 in dataset.uns[projection]:
        ret[projection2] = kni_score(data,projection2)['kni_score']
    return ret

def kni_score(
    adata: anndata.AnnData,
    embedding_key: str,
    label_key: str = 'label',
    batch_key: str = 'batch',
    n_neighbors: int = 10,
    exclude_labels: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Calculates a k-nearest-neighbor based integration score (KNI).

    The score measures how well cell types are mixed across batches. It is the
    accuracy of a k-NN classifier where, for each cell, the label is predicted
    using only neighbors from different batches. A high score indicates that
    cells of the same type are close to each other, regardless of their batch.

    Args:
        adata:
            AnnData object with embeddings, labels, and batches.
        embedding_key:
            Key in `adata.obsm` where the embedding is stored.
        label_key:
            Key in `adata.obs` for cell type labels.
        batch_key:
            Key in `adata.obs` for batch labels.
        n_neighbors:
            Number of neighbors to consider for each cell.
        exclude_labels:
            A list of labels to exclude from the calculation (e.g., ['Unknown']).

    Returns:
        A dictionary with the following scores:
        - 'kni_score': The primary integration score (0 to 1, higher is better).
        - 'label_accuracy': k-NN classification accuracy using all neighbors.
        - 'mean_proportion_same_batch': Average proportion of neighbors from the
          same batch (0 to 1, lower is better).
    """
    # 1. Filter and extract data
    adata_view = adata
    if exclude_labels:
        keep_mask = ~adata_view.obs[label_key].isin(exclude_labels)
        # Use .copy() to avoid SettingWithCopyWarning on a view
        adata_view = adata_view[keep_mask, :].copy()

    if embedding_key not in adata_view.obsm:
        raise KeyError(f"Embedding key '{embedding_key}' not found in adata.obsm.")
    if not all(k in adata_view.obs for k in [label_key, batch_key]):
        raise KeyError(f"Label or batch key not found in adata.obs.")

    embeddings = adata_view.obsm[embedding_key]
    # Use pd.Categorical to handle any data type and convert to integer codes
    labels = pd.Categorical(adata_view.obs[label_key]).codes
    batches = pd.Categorical(adata_view.obs[batch_key]).codes
    n_cells = embeddings.shape[0]

    if n_cells < n_neighbors + 1:
        return {
            'kni_score': np.nan,
            'label_accuracy': np.nan,
            'mean_proportion_same_batch': np.nan
        }

    # 2. Find nearest neighbors (excluding the cell itself)
    X = embeddings.astype(np.float32)
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm='auto')
    nn.fit(X)
    _, neighbor_indices = nn.kneighbors(X)
    neighbor_indices = neighbor_indices[:, 1:]  # Exclude self (the first neighbor)

    # 3. Get labels and batches of neighbors using the indices
    neighbor_labels = labels[neighbor_indices]
    neighbor_batches = batches[neighbor_indices]

    # 4. Calculate overall label accuracy using all neighbors
    # scipy.stats.mode finds the most frequent label for each cell's neighborhood
    predicted_labels = mode(neighbor_labels, axis=1, keepdims=False)[0]
    label_accuracy = np.mean(predicted_labels == labels)

    # 5. Calculate KNI score and batch mixing
    # Reshape original cell's batch array for broadcasting against neighbor_batches
    cell_batches = batches.reshape(-1, 1)
    is_different_batch = neighbor_batches != cell_batches

    # Calculate the average proportion of neighbors from the same batch
    proportion_same_batch = (~is_different_batch).sum(axis=1) / n_neighbors
    mean_proportion_same_batch = proportion_same_batch.mean()

    # Calculate KNI score (accuracy using only different-batch neighbors)
    kni_correct_predictions = 0
    cells_with_foreign_neighbors = 0

    for i in range(n_cells):
        # Filter for neighbors from different batches
        foreign_neighbor_labels = neighbor_labels[i, is_different_batch[i]]

        if foreign_neighbor_labels.size > 0:
            cells_with_foreign_neighbors += 1
            # Predict label from these "foreign" neighbors
            foreign_predicted_label = mode(foreign_neighbor_labels, keepdims=False)[0]
            if foreign_predicted_label == labels[i]:
                kni_correct_predictions += 1

    # Avoid division by zero if no cells have neighbors from other batches
    kni_score = kni_correct_predictions / cells_with_foreign_neighbors if cells_with_foreign_neighbors > 0 else 0.0

    return {
        'kni_score': kni_score,
        'label_accuracy': label_accuracy,
        'mean_proportion_same_batch': mean_proportion_same_batch,
    }
