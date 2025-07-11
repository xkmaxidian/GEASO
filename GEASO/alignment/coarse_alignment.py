import numpy as np
from GEASO.utils.utils import voxel_data
from GEASO.utils.utils import normalize_coords
from GEASO.utils.utils import cal_distance


def inlier_from_NN(x, y, distance, max_iter=100, gamma=0.5, scale_c=False):
    # x: source coordinates, shape (N, D)
    # y: target coordinates, shape (N, D)
    N, D = x.shape[0], x.shape[1]
    alpha = 1
    distance = np.maximum(0, distance)  # ensure all distance > 0
    normalize = np.max(distance) / (np.log(10) * 2)  # normalize distance to a reasonable range
    distance = distance / normalize
    R = np.eye(D)
    t = np.ones((D, 1))
    y_hat = x.copy()  # initial guess for y_hat

    sigma2 = np.sum((y_hat - y) ** 2) / (D * N)  # calculate initial sigma2 based on the residuals
    weight = np.exp(-distance * alpha)  # calculate initial weight based on distance
    init_weight = weight
    P = np.multiply(np.ones((N, 1)), weight)  # initial P
    alpha_end = 0.1
    alpha_decrease = np.power(alpha_end / alpha, 1 / (max_iter - 20))
    a = np.maximum(
        np.prod(np.max(x, axis=0) - np.min(x, axis=0)),
        np.prod(np.max(y, axis=0) - np.min(y, axis=0)),
    )
    Sp = np.sum(P)  # D_p = \sum_{i,j} p_{i,j}
    error_list = []

    for iter_n in range(max_iter):
        mu_x = np.sum(np.multiply(x, P), 0) / Sp
        mu_y = np.sum(np.multiply(y, P), 0) / Sp
        X_mu, Y_mu = x - mu_x, y - mu_y  # center the data

        A = np.dot(Y_mu.T, np.multiply(X_mu, P))  # calculate the covariance matrix A
        if not np.isfinite(A).all():
            break

        svdU, svdS, svdV = np.linalg.svd(A)  # perform SVD on A
        C = np.eye(D)
        C[-1, -1] = np.linalg.det(np.dot(svdU, svdV))  # ensure the rotation matrix is proper
        R = np.dot(np.dot(svdU, C), svdV)  # calculate the rotation matrix R

        scale = 1
        if scale_c:
            XPX = np.dot(np.transpose(P), np.sum(np.multiply(X_mu, X_mu), axis=1))
            scale = np.trace(np.dot(np.transpose(A), R)) / XPX

        t = mu_y - scale * np.dot(mu_x, R.T)
        y_hat = scale * np.dot(x, R.T) + t  # update y_hat based on the current estimate of R and t

        # get P
        term1 = np.multiply(np.exp(-(np.sum((y - y_hat) ** 2, 1, keepdims=True)) / (2 * sigma2)), weight)
        outlier_part = np.max(weight) * (1 - gamma) * np.power((2 * np.pi * sigma2), D / 2) / (gamma * a)
        P = term1 / (term1 + outlier_part)
        Sp = np.sum(P)
        gamma = np.minimum(np.maximum(Sp / N, 0.01), 0.99)
        P = np.maximum(P, 1e-6)

        # update sigma2
        if scale_c:
            xPx = np.dot(np.transpose(P), np.sum(np.multiply(Y_mu, Y_mu), axis=1))
            sigma2 = (xPx - scale * np.trace(np.dot(np.transpose(A), R))) / (Sp * D)
        else:
            sigma2 = np.sum(np.multiply((y_hat - y) ** 2, P)) / (D * Sp)  # update sigma2 based on the residuals

        error = np.sum(np.multiply((y_hat - y) ** 2, P)) / (D * Sp)
        error_list.append(error)

        if iter_n > 20:
            alpha = alpha * alpha_decrease  # alpha decreases after 20 iterations
            weight = np.exp(-distance * alpha)  # update weight based on the new alpha
            weight = weight / np.max(weight)  # normalize weight to avoid numerical issues
            if error < 1e-6:
                break

    fix_sigma2 = 1e-2  # after the iterations, we fix sigma2 to a small value to avoid numerical issues
    fix_gamma = 0.1
    term1 = np.multiply(np.exp(-(np.sum((y - y_hat) ** 2, 1, keepdims=True)) / (2 * fix_sigma2)), weight)
    outlier_part = np.max(weight) * (1 - fix_gamma) * np.power((2 * np.pi * fix_sigma2), D / 2) / (fix_gamma * a)
    P = term1 / (term1 + outlier_part)
    gamma = np.minimum(np.maximum(np.sum(P) / N, 0.01), 0.99)
    return P, R, t, init_weight, sigma2, gamma, scale, error_list


def coarse_alignment(source, target, top_K=10, dis_metric='kl', use_latent=False, scale_c=False):
    scales, means, coords = normalize_coords(source.obsm['spatial'].copy(), target.obsm['spatial'].copy(), dims=2)
    print(scales, means)
    source_coord = coords[0]
    target_coord = coords[1]
    M, N, D = source_coord.shape[0], target_coord.shape[0], target_coord.shape[1]
    if use_latent:
        coors1, exp1 = voxel_data(coords=source_coord, gene_exp=source.obsm['latent'],
                                  voxel_num=max(min(int(N / 5), 1000), 1000))
        coors2, exp2 = voxel_data(coords=target_coord, gene_exp=target.obsm['latent'],
                                  voxel_num=max(min(int(M / 5), 1000), 1000))
    else:
        coors1, exp1 = voxel_data(coords=source_coord, gene_exp=source.X.A,
                                  voxel_num=max(min(int(N / 5), 1000), 1000))
        coors2, exp2 = voxel_data(coords=target_coord, gene_exp=target.X.A,
                                  voxel_num=max(min(int(M / 5), 1000), 1000))

    exp_dist = cal_distance(exp1, exp2, metric=dis_metric)
    item2 = np.argpartition(exp_dist, top_K, axis=0)[: top_K, :].T
    item1 = np.repeat(np.arange(exp_dist.shape[1])[:, None], top_K, axis=1)
    NN1 = np.dstack((item1, item2)).reshape((-1, 2))
    distance1 = exp_dist.T[NN1[:, 0], NN1[:, 1]]

    item1 = np.argpartition(exp_dist, top_K, axis=1)[:, : top_K]
    item2 = np.repeat(np.arange(exp_dist.shape[0])[:, None], top_K, axis=1)
    NN2 = np.dstack((item1, item2)).reshape((-1, 2))
    distance2 = exp_dist.T[NN2[:, 0], NN2[:, 1]]

    NN = np.vstack((NN1, NN2))
    distance = np.r_[distance1, distance2]

    train_x, train_y = coors1[NN[:, 1], :], coors2[NN[:, 0], :]
    P, R, t, init_weight, sigma2, gamma, scale, error_list = inlier_from_NN(train_x, train_y, distance[:, None],
                                                                            max_iter=100, scale_c=scale_c)
    coarse_source = scale * (source_coord @ R.T) + t
    coarse_target = target_coord.copy()
    return R, t, scale, P, scales, means, train_x, train_y, coarse_source, coarse_target
