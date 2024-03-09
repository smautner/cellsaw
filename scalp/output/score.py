from  sklearn.neighbors import KNeighborsClassifier
from ubergauss import tools as ut
import numpy as np
from sklearn.metrics import  silhouette_score

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

    # scores = np.array([1-acc(l) for l in np.unique(dataset.obs['label'])]) ### !!!!
    # weight by label counts ..
    return np.nanmean([1-acc(l) for l in np.unique(dataset.obs['label'])])

