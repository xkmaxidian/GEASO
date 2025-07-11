import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import RBFInterpolator
from sklearn.kernel_approximation import Nystroem  # Nyström 近似
from sklearn.neighbors import NearestNeighbors


def kernel_interpolate(
        Y,  # (N, D) target points
        y,  # (M, D) source points
        x,  # (M, D) 源点 (affine/pre-aligned)
        w,  # (M,)   各控制点权重
        beta: float,  # 核宽
        lam: float,  # 正则 λ = λmd*(r/s)^2
        nystrom_K: int = 0,  # 0 表示直接求解
        kernel="gaussian",
        rigid=(1.0, np.eye(3), np.zeros(3)),  # (s, R, t)
):
    """返回:
       T  : (N, D) 变换后坐标
       V  : (N, D) 非刚性位移
       W  : (M, D) 核系数(可选返回)
    """
    s, R, t = rigid
    D = Y.shape[1]

    # --- 1. 反向仿射，把 x 映射到控制点坐标系 ---
    ix = (x - t) @ R.T / s  # ↩️ invlinear

    # --- 2. 残差 = ix - y ---
    E = ix - y  # (M, D)

    # --- 3. 核映射 Φ(y) ---
    if kernel == "gaussian":
        def k(a, b):
            # |a| = (..., D) ; |b| = (..., D)
            return np.exp(-np.sum((a - b) ** 2, axis=-1) / (2 * beta ** 2))
    else:
        raise ValueError("Only Gaussian kernel shown here")

    # ---------- 3a. Nyström 近似 ----------
    if nystrom_K and nystrom_K < y.shape[0]:
        # 用 scikit-learn 做 Nyström : Φ ≈ Z W
        ny = Nystroem(kernel=k, n_components=nystrom_K, random_state=0)
        Z = ny.fit_transform(y)  # (M, K)
        G = (Z.T * w).dot(Z)  # K × K
        G.flat[:: G.shape[0] + 1] += lam  # 加 λI
        Wcoeff = np.linalg.solve(G, (Z.T * w).dot(E))  # K × D
        V = ny.transform(Y).dot(Wcoeff)  # N × D
    # ---------- 3b. 直接求解 ----------
    else:
        # Φ_ij = k(y_i, y_j)
        Kmat = k(y[:, None, :], y[None, :, :])
        Kmat += np.diag(lam / w)  # (M, M)
        Wcoeff = np.linalg.solve(Kmat, w[:, None] * E)  # M × D
        V = k(Y[:, None, :], y[None, :, :]).dot(Wcoeff)  # N × D

    # --- 4. 输出 ---
    T = s * (Y + V).dot(R.T) + t  # rigid + non-rigid
    return T, V, Wcoeff


# --------------------------------------------
# 2. 局部高斯加权 KNN 插值。对应 interpolate_x
# --------------------------------------------
def local_gaussian_knn(
        y,  # (M, D) 控制点坐标
        X,  # (N, D) 参考样本
        r: float,  # 半径参数
        K: int = 30  # 最近邻上限
):
    neigh = NearestNeighbors(n_neighbors=min(K, X.shape[0]),
                             radius=2 * r, algorithm="kd_tree")
    neigh.fit(X)
    dist, idx = neigh.kneighbors(y, return_distance=True)
    M, D = y.shape
    x_new = np.zeros_like(y)

    for m in range(M):
        d = dist[m]
        ind = idx[m]
        # 可能小于 K，过滤无效邻居
        mask = np.isfinite(d)
        if not mask.any():  # fallback：用最近点
            d, ind = neigh.kneighbors(y[m:m + 1], n_neighbors=1, return_distance=True)
            d, ind = d[0], ind[0]
            mask = np.ones_like(d, dtype=bool)
        d, ind = d[mask], ind[mask]
        w = np.exp(-d ** 2 / (r ** 2))
        w_sum = w.sum()
        x_new[m] = (w[:, None] * X[ind]).sum(0) / w_sum
    return x_new


# ----------------------------
# ------------- DEMO ----------
if __name__ == "__main__":
    # 生成简单数据做示例
    rng = np.random.default_rng(0)
    M, N, D = 80, 400, 2
    y = rng.normal(size=(M, D))  # 控制点
    true_V = 0.1 * rng.normal(size=(M, D))  # 真位移
    x = y + true_V  # 源点 = 控制点经过非刚性形变
    Y = rng.normal(size=(N, D))  # 任意目标点
    w = np.ones(M)  # 等权
    beta, lam = 0.5, 1e-2
    s, R, t = 1.0, np.eye(D), np.zeros(D)  # 仅示例，无全局刚性

    T, V, _ = kernel_interpolate(
        Y, y, x, w, beta, lam,
        nystrom_K=40, rigid=(s, R, t)
    )
    print("Interpolated target shape:", T.shape)

    # KNN 插值示例
    X = rng.normal(size=(300, D))
    x_knn = local_gaussian_knn(y, X, r=0.6)
    print("local KNN interp shape:", x_knn.shape)
