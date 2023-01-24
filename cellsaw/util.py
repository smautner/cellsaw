
from cellsaw.merge.diffusion import kernel
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
def setd2_kernel_mds(m):
    data  = kernel.linear_assignment_kernel(*m.d10, neighbors= 5,
                                            neighbors_inter=5,
                                            linear_assignment_factor=5,
                                            return_dm=True)
    r  = MDS(dissimilarity='precomputed').fit_transform(data)
    m.d2 = r[:1000], r[1000:]
    return

