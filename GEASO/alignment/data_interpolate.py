import numpy as np
from scipy.spatial.transform import Rotation
from scipy.interpolate import RBFInterpolator
from sklearn.kernel_approximation import Nystroem  # Nyström 近似
from sklearn.neighbors import NearestNeighbors


def kernel_interpolate(
        Y,  # (N, D) target points
        y,  # (M, D) source points
        x,  # (M, D) source points (affine/pre-aligned)
        w,  # (M,)   weight of control points
        beta: float,  # width of kernel
        lam: float,  # reg λ = λmd*(r/s)^2
        nystrom_K: int = 0,  # 0 indicates problem solve
        kernel="gaussian",
        rigid=(1.0, np.eye(3), np.zeros(3)),  # (s, R, t)
):
    """ return:
       T  : (N, D) transformed coordinate
       V  : (N, D) non-rigid vector
       W  : (M, D) kernel parameters (Optional)
    """
    s, R, t = rigid
    D = Y.shape[1]

    ix = (x - t) @ R.T / s
    E = ix - y  # (M, D)

    if kernel == "gaussian":
        def k(a, b):
            # |a| = (..., D) ; |b| = (..., D)
            return np.exp(-np.sum((a - b) ** 2, axis=-1) / (2 * beta ** 2))
    else:
        raise ValueError("Only Gaussian kernel shown here")

    # ---------- Nyström  ----------
    if nystrom_K and nystrom_K < y.shape[0]:
        # Nyström : Φ ≈ Z W
        ny = Nystroem(kernel=k, n_components=nystrom_K, random_state=0)
        Z = ny.fit_transform(y)  # (M, K)
        G = (Z.T * w).dot(Z)  # K × K
        G.flat[:: G.shape[0] + 1] += lam  # 加 λI
        Wcoeff = np.linalg.solve(G, (Z.T * w).dot(E))  # K × D
        V = ny.transform(Y).dot(Wcoeff)  # N × D
    # ---------- problem solve ----------
    else:
        # Φ_ij = k(y_i, y_j)
        Kmat = k(y[:, None, :], y[None, :, :])
        Kmat += np.diag(lam / w)  # (M, M)
        Wcoeff = np.linalg.solve(Kmat, w[:, None] * E)  # M × D
        V = k(Y[:, None, :], y[None, :, :]).dot(Wcoeff)  # N × D

    T = s * (Y + V).dot(R.T) + t  # rigid + non-rigid
    return T, V, Wcoeff


def local_gaussian_knn(
        y,
        X,
        r: float,
        K: int = 30
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

        mask = np.isfinite(d)
        if not mask.any():
            d, ind = neigh.kneighbors(y[m:m + 1], n_neighbors=1, return_distance=True)
            d, ind = d[0], ind[0]
            mask = np.ones_like(d, dtype=bool)
        d, ind = d[mask], ind[mask]
        w = np.exp(-d ** 2 / (r ** 2))
        w_sum = w.sum()
        x_new[m] = (w[:, None] * X[ind]).sum(0) / w_sum
    return x_new


def interpolate_displacement_field(
        control_points,
        control_displacements,
        query_points,
        method="knn",
        k=32,
        bandwidth=None,
        control_weights=None,
        chunk_size=50000,
        eps=1e-8,
        rbf_kernel="thin_plate_spline",
        rbf_smoothing=0.0,
        rbf_neighbors=None,
):
    """Interpolate a sparse non-rigid displacement field onto query points.

    Parameters
    ----------
    control_points
        Coordinates where displacements are known, shape (M, D).
    control_displacements
        Displacement vectors at control_points, shape (M, D).
    query_points
        Coordinates to receive interpolated displacement vectors, shape (N, D).
    method
        "knn" for local Gaussian KNN interpolation, or "rbf" for
        scipy.interpolate.RBFInterpolator.
    k
        Number of nearest control points used by the KNN interpolator.
    bandwidth
        Gaussian bandwidth. If None, it is estimated from control-point
        neighbor distances.
    control_weights
        Optional confidence weights for control points, shape (M,).
    chunk_size
        Number of query points processed per chunk.

    Returns
    -------
    np.ndarray
        Interpolated displacement vectors, shape (N, D).
    """
    control_points = np.asarray(control_points, dtype=np.float64)
    control_displacements = np.asarray(control_displacements, dtype=np.float64)
    query_points = np.asarray(query_points, dtype=np.float64)

    if control_points.ndim != 2 or query_points.ndim != 2:
        raise ValueError("control_points and query_points must be 2D arrays.")
    if control_displacements.shape != control_points.shape:
        raise ValueError("control_displacements must have the same shape as control_points.")
    if control_points.shape[1] != query_points.shape[1]:
        raise ValueError("control_points and query_points must have the same dimensionality.")
    if control_points.shape[0] == 0:
        raise ValueError("At least one control point is required.")

    method = method.lower()
    if method in ["rbf", "kernel"]:
        interpolator = RBFInterpolator(
            control_points,
            control_displacements,
            kernel=rbf_kernel,
            smoothing=rbf_smoothing,
            neighbors=rbf_neighbors,
        )
        outputs = []
        for start in range(0, query_points.shape[0], chunk_size):
            end = min(start + chunk_size, query_points.shape[0])
            outputs.append(interpolator(query_points[start:end]))
        return np.vstack(outputs)

    if method not in ["knn", "local", "local_gaussian"]:
        raise ValueError("method must be one of {'knn', 'local', 'local_gaussian', 'rbf', 'kernel'}.")

    n_controls = control_points.shape[0]
    n_neighbors = min(max(int(k), 1), n_controls)
    neigh = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
    neigh.fit(control_points)

    if bandwidth is None:
        sample_size = min(n_controls, 2000)
        sample_idx = np.linspace(0, n_controls - 1, sample_size, dtype=int)
        sample_k = min(max(n_neighbors, 2), n_controls)
        sample_dist, _ = neigh.kneighbors(control_points[sample_idx], n_neighbors=sample_k)
        positive = sample_dist[sample_dist > eps]
        bandwidth = np.median(positive) if positive.size else 1.0
    bandwidth = max(float(bandwidth), eps)

    if control_weights is None:
        control_weights = np.ones(n_controls, dtype=np.float64)
    else:
        control_weights = np.asarray(control_weights, dtype=np.float64).reshape(-1)
        if control_weights.shape[0] != n_controls:
            raise ValueError("control_weights must have shape (M,).")
        control_weights = np.maximum(control_weights, 0.0)

    interpolated = np.zeros_like(query_points, dtype=np.float64)
    for start in range(0, query_points.shape[0], chunk_size):
        end = min(start + chunk_size, query_points.shape[0])
        dist, idx = neigh.kneighbors(query_points[start:end], return_distance=True)

        local_weights = np.exp(-0.5 * (dist / bandwidth) ** 2)
        local_weights *= control_weights[idx]

        exact = dist[:, 0] <= eps
        denom = local_weights.sum(axis=1, keepdims=True)
        denom = np.maximum(denom, eps)
        values = (local_weights[..., None] * control_displacements[idx]).sum(axis=1) / denom
        if np.any(exact):
            values[exact] = control_displacements[idx[exact, 0]]
        interpolated[start:end] = values

    return interpolated


if __name__ == "__main__":
    # data generation
    rng = np.random.default_rng(0)
    M, N, D = 80, 400, 2
    y = rng.normal(size=(M, D))
    true_V = 0.1 * rng.normal(size=(M, D))  # real non-rigid vector
    x = y + true_V
    Y = rng.normal(size=(N, D))
    beta, lam = 0.5, 1e-2
    s, R, t = 1.0, np.eye(D), np.zeros(D)

    T, V, _ = kernel_interpolate(
        Y, y, x, w, beta, lam,
        nystrom_K=40, rigid=(s, R, t)
    )
    print("Interpolated target shape:", T.shape)

    # KNN interp
    X = rng.normal(size=(300, D))
    x_knn = local_gaussian_knn(y, X, r=0.6)
    print("local KNN interp shape:", x_knn.shape)
