import torch
import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import kneighbors_graph
from sklearn.neighbors import KDTree
from scipy.sparse.csgraph import dijkstra
from typing import Union


def construct_knn_graph(
    points,
    k: int = 10,
) -> sp.csr_matrix:
    n, d = points.shape
    if k >= n:
        raise ValueError(f"k ({k}) must be < number of points ({n}).")

    tree = KDTree(points)          # O(n log n)
    dist, idx = tree.query(points, k=k + 1)  # 包含自身 → k+1

    row = np.repeat(np.arange(n), k)
    col = idx[:, 1:].reshape(-1)     # 去掉自身索引
    data = dist[:, 1:].reshape(-1)
    A = sp.coo_matrix((data, (row, col)), shape=(n, n))
    A = (A + A.T)
    A = sp.csr_matrix(A)
    return A


def con_K_geo(
    X: Union[np.ndarray, "torch.Tensor"],
    Y_idx: Union[np.ndarray, "torch.Tensor"],
    beta: float = 0.01,
    k: int = 10,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    返回 |X| × |Y| 的核矩阵，元素为 exp(-beta * geodesic_dist).
    点坐标可为 NumPy 或 PyTorch (自动搬到 CPU)。
    """
    # ---------- 数据准备 ----------
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    else:
        X = np.asarray(X, dtype=np.float32)

    if torch.is_tensor(Y_idx):
        Y_idx = Y_idx.detach().cpu().numpy()
    else:
        Y_idx = np.asarray(Y_idx, dtype=np.float32)

    A = construct_knn_graph(X, k=k)
    dist_mat = dijkstra(A, directed=False, indices=Y_idx, return_predecessors=False)

    geo_XY = dist_mat.T
    geo_XY[np.isinf(geo_XY)] = np.finfo(np.float32).max
    K = np.exp(-beta * geo_XY)
    return K


if __name__ == "__main__":
    X = np.random.rand(170000, 2)
    inducing_variables_idx = (np.random.choice(X.shape[0], 200, replace=False))
    K_geo = con_K_geo(X, inducing_variables_idx, beta=0.05, k=15)  # 200 × 150
    print(K_geo.shape)
