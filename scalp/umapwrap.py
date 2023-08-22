from scalp.data.transform import stack_single_attribute, attach_stack
import umap

def adatas_umap(adatas, dim = 10, label = 'umap10', from_obsm = 'pca40'):

    if label in adatas[0].obsm:
        print('redundant umap :) ')

    #attr = data.obsm.get(start,'')
    X = stack_single_attribute(adatas, attr = from_obsm)
    res = umap.UMAP(n_components = dim).fit_transform(X)
    return attach_stack(adatas, res, label)


def graph_umap(adatas, distance_adjacency_matrix,
               label=f'lsa', n_neighbors = 10,
               n_components = 2, **kwargs):


    res =  umap.UMAP(n_components=n_components,
                     metric='precomputed',n_neighbors = n_neighbors,
                     **kwargs).fit_transform(distance_adjacency_matrix)

    return attach_stack(adatas, res, label)
