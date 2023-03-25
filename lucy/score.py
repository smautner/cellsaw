from  sklearn.neighbors import KNeighborsClassifier
from ubergauss import tools as ut
import numpy as np


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

