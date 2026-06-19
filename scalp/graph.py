
import numpy as np
from anndata import AnnData
from scipy import sparse
from scipy.sparse import csr_matrix, lil_matrix
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
from sklearn import metrics
from ubergauss import hubness as uhub
from scalp import transform
from scalp.data.transform import to_arrays


def iterated_linear_sum_assignment(
    distances: np.ndarray,
    repeats: int,
    outlier_threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    """Runs iterated linear sum assignment and returns index pairs filtered by a percentage cutoff."""
    def assignment_iteration(d_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r, c = linear_sum_assignment(d_matrix)
        d = d_matrix[r, c].copy()
        d_matrix[r, c] = np.inf
        return r, c, d

    d_copy = distances.copy()
    results = [assignment_iteration(d_copy) for _ in range(repeats)]
    cc, rr, dists = zip(*results)

    i_ids = np.hstack(cc)
    j_ids = np.hstack(rr)
    val_distances = np.hstack(dists)

    if 0 < outlier_threshold < 1:
        sorted_distances = np.sort(val_distances)
        lsa_outlier_thresh = sorted_distances[int(len(val_distances) * outlier_threshold)]
        mask = val_distances <= lsa_outlier_thresh
        return i_ids[mask], j_ids[mask]

    return i_ids, j_ids


def fast_neighborgraph(distancemat: np.ndarray, neighbors: int) -> np.ndarray:
    """Builds a nearest-neighbor matrix populated with ranked distance scores."""
    part = np.argpartition(distancemat, neighbors, axis=1)[:, :neighbors]
    neighborsgraph = np.zeros_like(distancemat)
    ranks = rankdata(distancemat, axis=1)
    np.put_along_axis(neighborsgraph, part, np.take_along_axis(ranks, part, axis=1), axis=1)
    return neighborsgraph


def pac_neighborgraph(D: np.ndarray, k: int) -> np.ndarray:
    """Samples near, mid, and far pairs to construct a balanced neighbor matrix."""
    n, s = D.shape[0], D.shape[0] // 6
    res = np.zeros_like(D, dtype=np.int8)
    idx = np.argsort(D, axis=1)
    row = np.arange(n)[:, None]

    # 1. Near: Absolute closest
    res[row, idx[:, 1:k+1]] = 1
    # 2. Mid: Sample from 2nd sextile
    m_pool = idx[:, s : 2*s]
    m_samp = m_pool[row, np.random.randint(0, s, (n, k // 2))]
    res[row, m_samp] = 2
    # 3. Far: Sample from 3rd sextile
    f_pool = idx[:, 2*s : 3*s]
    f_samp = f_pool[row, np.random.randint(0, s, (n, k * 2))]
    res[row, f_samp] = 3
    return res


def mkblock(matrix: lil_matrix, i: np.ndarray, j: np.ndarray) -> lil_matrix:
    """Reorders the rows of a sparse LIL matrix based on source-target indexing maps."""
    if len(i) == 0:
        return lil_matrix(matrix.shape)
    submat = matrix.tocsr()[j, :].tocoo()
    i_array = np.array(i)
    mapped_rows = i_array[submat.row]
    return sparse.coo_matrix((submat.data, (mapped_rows, submat.col)), shape=matrix.shape).tolil()


def integrate(
    adata: AnnData,
    batch_key: str = "batch",
    base: str = "pca40",
    k: int = 12,
    metric: str = "cosine",
    dataset_adjacency: np.ndarray | bool = False,
    hub1_k: int = 12,
    hub2_k: int = 12,
    hub1_algo: int = 2,
    hub2_algo: int = 2,
    pac: bool = False,
    outlier_threshold: float = 0.75,
) -> csr_matrix:
    """Integrates batches within a single AnnData object into a unified sparse neighbor graph."""
    adata_split = transform.split_by_obs(adata)
    Xlist = to_arrays(adata_split, base)
    n_datas = len(Xlist)

    if n_datas <= 1:
        raise ValueError("At least 2 batches are required to perform integration.")

    def adjacent(i: int, j: int) -> bool:
        if isinstance(dataset_adjacency, np.ndarray):
            return dataset_adjacency[i][j] == 1
        return True

    def make_distance_matrix(ij: tuple[int, int]) -> lil_matrix | tuple[np.ndarray, np.ndarray]:
        i, j = ij
        if not adjacent(i, j):
            if i == j:
                return sparse.lil_matrix((Xlist[i].shape[0], Xlist[i].shape[0]))
            return np.array([]), np.array([])

        distances = metrics.pairwise_distances(Xlist[i], Xlist[j], metric=metric)

        if i == j:
            assert distances.shape[0] > 50, 'Insufficient cells in dataset!'
            distances = uhub.transform(distances, k = hub1_k, algo = hub1_algo, skip_diag=True)

            f = fast_neighborgraph if not pac else pac_neighborgraph
            distances_graph = f(distances, k)

            np.fill_diagonal(distances_graph, 1)
            return sparse.lil_matrix(distances_graph)

        distances = uhub.transform(distances, k=hub2_k, algo = hub2_algo, skip_diag=False)
        return iterated_linear_sum_assignment(distances, 1, outlier_threshold)

    # Generate pairwise matrices
    tasks = [(i, j) for i in range(n_datas) for j in range(i, n_datas)]
    parts = [make_distance_matrix(t) for t in tasks]
    getpart = dict(zip(tasks, parts))

    # Reassemble blocks
    blockdict: dict[tuple[int, int], lil_matrix] = {}
    for i in range(n_datas):
        blockdict[(i, i)] = getpart[(i, i)]

    tasks_offdiag = [(i, j) for i in range(n_datas) for j in range(i + 1, n_datas)]
    for i, j in tasks_offdiag:
        blockdict[(i, j)] = mkblock(getpart[(j, j)], *getpart[(i, j)])

    for i, j in tasks_offdiag:
        imatch, jmatch = getpart[(i, j)]
        blockdict[(j, i)] = mkblock(getpart[(i, i)], jmatch, imatch)

    # Stack blocks into a single global sparse matrix
    blocks = [[blockdict[(i, j)] for j in range(n_datas)] for i in range(n_datas)]
    stack = sparse.bmat(blocks, format='csr')

    # Fast diagonal clearing to prevent SparsityEfficiencyWarning updates on CSR matrix
    stack = stack.tolil()
    stack.setdiag(0)
    stack = stack.tocsr()

    return stack


def test_integrate() -> None:
    """Verifies integration logic using a mock AnnData setup with 2 batches."""
    print("Preparing mock data...")
    X1 = np.random.random((100, 40))
    X2 = np.random.random((100, 40))
    X_combined = np.vstack([X1, X2])

    import anndata as ad
    adata = ad.AnnData(np.zeros((200, 10)))  # Dummy counts matrix
    adata.obsm["X_pca"] = X_combined
    adata.obs["batch"] = ["batch_1"] * 100 + ["batch_2"] * 100

    print("Running integration test...")
    graph = integrate(adata, batch_key="batch", base="X_pca")
    print(f"Integration completed. Graph shape: {graph.shape}")
