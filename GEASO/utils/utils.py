from typing import Optional, Union

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import torch
from GEASO.model.GNN import Args, CCA_SSG
from GEASO.model.utils import delaunay_dgl
from GEASO.model.model_train import train
import scanpy as sc


def feature_extract(adatas, graph_strategy='convex', args=None):
    inputs = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for adata in adatas:
        graph = delaunay_dgl(spatial_coords=adata.obsm['spatial'], if_plot=True, graph_strategy=graph_strategy).to(
            device)
        feature = torch.tensor(sc.pp.normalize_total(adata, target_sum=1e4, inplace=False)['X'].toarray(),
                               dtype=torch.float32).to(device)
        sample = adata.obs['batch'].unique()[0]
        inputs.append((sample, graph, feature))

    if args is None:
        args = Args(
            device=device,
            epochs=200,
            lr1=1e-3,
            use_encoder=True,
            encoder_dim=512,
        )

    in_dim = adatas[0].shape[1]
    model = CCA_SSG(in_dim=in_dim, encoder_dim=args.encoder_dim, n_layers=args.n_layers,
                    use_encoder=args.use_encoder).to(args.device)
    print(f'Training on {args.device}, please wait...')
    emb_dict, loss_log, model = train(model=model, inputs=inputs, args=args)
    return emb_dict, loss_log, model


def intersect(lst1, lst2):
    temp = set(lst2)
    lst3 = [value for value in lst1 if value in temp]
    return lst3


def normalize_coords(coordsA, coordsB, dims=2, separate_mean=True, separate_scale=False):
    normalize_scales = np.zeros((dims,))
    normalize_means = np.zeros((2, dims))

    coords = [coordsA, coordsB]

    for i in range(len(coords)):
        normalize_mean = np.einsum("ij->j", coords[i]) / coords[i].shape[0]
        normalize_means[i] = normalize_mean

    if not separate_mean:  # default is True
        global_mean = np.mean(normalize_means, axis=0)
        normalize_means = np.repeat(global_mean, 2, axis=0)

    for i in range(len(coords)):
        coords[i] -= normalize_means[i]
        normalize_scale = np.sqrt(
            np.einsum("ij->", np.einsum("ij,ij->ij", coords[i], coords[i])) / coords[i].shape[0]
        )
        normalize_scales[i] = normalize_scale

    # get the global scale for whole coords if "separate_scale" is False
    if not separate_scale:
        global_scale = np.mean(normalize_scales)
        normalize_scales = np.full((len(coords),), global_scale)

        # normalize the scale of the coords
        for i in range(len(coords)):
            coords[i] /= normalize_scales[i]

    normalize_scales = normalize_scales
    normalize_means = normalize_means
    return normalize_scales, normalize_means, coords


def voxel_data(coords, gene_exp, voxel_size=None, voxel_num=10000):
    N, D = coords.shape[0], coords.shape[1]
    coords = np.array(coords)
    gene_exp = np.array(gene_exp)

    # create the voxel grid
    min_coords = np.min(coords, axis=0)
    max_coords = np.max(coords, axis=0)
    if voxel_size is None:
        voxel_size = np.sqrt(np.prod(max_coords - min_coords)) / (np.sqrt(N) / 5)

    voxel_steps = (max_coords - min_coords) / int(np.sqrt(voxel_num))
    voxel_coords = [
        np.arange(min_coord, max_coord, voxel_step)
        for min_coord, max_coord, voxel_step in zip(min_coords, max_coords, voxel_steps)
    ]

    voxel_coords = np.stack(np.meshgrid(*voxel_coords), axis=-1).reshape(-1, D)
    voxel_gene_exps = np.zeros((voxel_coords.shape[0], gene_exp.shape[1]))
    is_voxels = np.zeros((voxel_coords.shape[0],))

    # assign the data points to the voxels
    for i, voxel_coord in enumerate(voxel_coords):
        dists = np.sqrt(np.sum((coords - voxel_coord) ** 2, axis=1))
        mask = dists < voxel_size / 2
        if np.any(mask):
            voxel_gene_exps[i] = np.mean(gene_exp[mask], axis=0)
            is_voxels[i] = 1
    # drop voxels without data points
    voxel_coords = voxel_coords[is_voxels == 1, :]
    voxel_gene_exps = voxel_gene_exps[is_voxels == 1, :]
    return voxel_coords, voxel_gene_exps


def cal_distance(X, Y, metric='euc'):
    if metric in ['euc', 'euclidean']:
        distance = euc_distance_cal(X, Y, square=False)
    elif metric in ['square_euc', 'square_euclidean']:
        distance = euc_distance_cal(X, Y, square=True)
    elif metric in ['cos', 'cosine']:
        distance = cos_distance_cal(X, Y)
    elif metric == 'kl':
        distance = kl_distance_cal(X, Y, probabilistic=True)
    else:
        raise ValueError("Invalid metric")
    return distance


def cal_probability(distance_matrix: Union[torch.Tensor, np.ndarray],
                    probability_type: str = 'gauss',
                    probability_parameter: Optional[float] = None) -> Union[torch.Tensor, np.ndarray]:
    if probability_type.lower() in ['gauss', 'gaussian']:
        if probability_type is None:
            raise ValueError("probability must be provided for Gaussian probability type.")
        if isinstance(distance_matrix, torch.Tensor):
            probability_matrix = torch.exp(-distance_matrix / (2 * probability_parameter))
        else:
            probability_matrix = np.exp(-distance_matrix / (2 * probability_parameter))
    elif probability_type.lower() in ['cos', 'cosine']:
        probability_matrix = 1 - distance_matrix
    elif probability_type.lower() in ['kl', 'kullback-leibler', 'prob']:
        probability_matrix = distance_matrix
    else:
        raise ValueError(f"Unsupported probability type: {probability_type}. Supported types are 'gauss', 'cos', 'kl'.")
    return probability_matrix


def euc_distance_cal(X, Y, square=False):
    assert X.shape[1] == Y.shape[1], "X and Y should have the same dimension"
    if isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor):
        dist2 = (X.pow(2).sum(dim=1, keepdim=True) + Y.pow(2).sum(dim=1).unsqueeze(0) - 2 * (X @ Y.t())).clamp(min=0.0)
        return dist2.sqrt() if square else dist2

    elif isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        dist2 = (np.sum(X ** 2, axis=1)[:, None] + np.sum(Y ** 2, axis=1)[None, :] - 2 * np.dot(X, Y.T))
        dist2 = np.maximum(dist2, 0)
        return np.sqrt(dist2) if square else dist2


def kl_distance_cal(X, Y, probabilistic=True, eps=1e-8):
    assert X.shape[1] == Y.shape[1], "X and Y should have the same dimension"
    if isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor):
        return kl_distance_cal_torch(X, Y, probabilistic=probabilistic, eps=eps)
    elif isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        X = X + 0.01
        Y = Y + 0.01
        if probabilistic:
            X = X / np.sum(X, axis=1, keepdims=True)
            Y = Y / np.sum(Y, axis=1, keepdims=True)
        log_X = np.log(X + eps)  # add eps to avoid log(0)
        log_Y = np.log(Y + eps)
        X_log_X = np.sum(X * log_X, axis=1, keepdims=True)
        distance = X_log_X - np.dot(X, log_Y.T)
        return distance


def kl_distance_cal_torch(
    X: torch.Tensor,
    Y: torch.Tensor,
    probabilistic: bool = True,
    eps: float = 1e-8
) -> torch.Tensor:
    X = X + 0.01
    Y = Y + 0.01
    if probabilistic:
        X = X / X.sum(dim=1, keepdim=True)
        Y = Y / Y.sum(dim=1, keepdim=True)
    log_X = (X + eps).log()
    log_Y = (Y + eps).log()
    X_log_X = (X * log_X).sum(dim=1, keepdim=True)
    cross = X @ log_Y.t()
    return X_log_X - cross


def cos_distance_cal(X, Y, eps=1e-8):
    assert X.shape[1] == Y.shape[1], "X and Y should have the same dimension"
    if isinstance(X, torch.Tensor) and isinstance(Y, torch.Tensor):
        return cos_distance_cal_torch(X, Y, eps=eps)
    elif isinstance(X, np.ndarray) and isinstance(Y, np.ndarray):
        X_norm = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        Y_norm = np.sqrt(np.sum(Y ** 2, axis=1, keepdims=True))
        X = X / np.maximum(X_norm, eps)
        Y = Y / np.maximum(Y_norm, eps)
        distance = -np.dot(X, Y.T) * 0.5 + 0.5
        return distance


def cos_distance_cal_torch(
    X: torch.Tensor,
    Y: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    X_norm = X.pow(2).sum(dim=1, keepdim=True).sqrt()
    Y_norm = Y.pow(2).sum(dim=1, keepdim=True).sqrt()
    X_unit = X / X_norm.clamp(min=eps)
    Y_unit = Y / Y_norm.clamp(min=eps)
    sim = X_unit @ Y_unit.t()
    dist = -sim * 0.5 + 0.5
    return dist


def crop_slices(
        slice1,
        slice2,
        spatial_key,
        overlay_ratio,
        axis=1,
):
    y_max, y_min = np.max(slice1.obsm[spatial_key][:, 1]), np.min(slice1.obsm[spatial_key][:, 1])
    x_max, x_min = np.max(slice1.obsm[spatial_key][:, 0]), np.min(slice1.obsm[spatial_key][:, 0])
    if axis == 0:
        slice_1_low = (1 - overlay_ratio) * (x_max - x_min) / (2 - overlay_ratio) + x_min
        slice_2_high = x_max - (1 - overlay_ratio) * (x_max - x_min) / (2 - overlay_ratio)
    elif axis == 1:
        slice_1_low = (1 - overlay_ratio) * (y_max - y_min) / (2 - overlay_ratio) + y_min
        slice_2_high = y_max - (1 - overlay_ratio) * (y_max - y_min) / (2 - overlay_ratio)
    slice1_crop = slice1[slice1.obsm[spatial_key][:, axis] > slice_1_low, :].copy()
    slice2_crop = slice2[slice2.obsm[spatial_key][:, axis] < slice_2_high, :].copy()
    return slice1_crop, slice2_crop


def rigid_transformation(
        adata,
        spatial_key,
        key_added,
        theta=None,
        inplace=True,
):
    if not inplace:
        adata = adata.copy()
    spatial = adata.obsm[spatial_key]
    mean = np.mean(spatial, axis=0)
    spatial = spatial - mean
    if theta is None:
        # random rotation
        theta = np.random.rand() * 2 * np.pi
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    spatial = np.matmul(spatial, rotation_matrix)
    spatial = spatial + mean
    adata.obsm[key_added] = spatial
    if inplace:
        pass
    else:
        return adata


def split_slice(
        adata,
        spatial_key,
        split_num=5,
        axis=2,
):
    spatial_points = adata.obsm[spatial_key]
    N = spatial_points.shape[0]
    sorted_points = np.argsort(spatial_points[:, axis])
    points_per_segment = len(sorted_points) // split_num
    split_adata = []
    for slice_id, i in enumerate(range(0, N, points_per_segment)):
        sorted_adata = adata[sorted_points[i: i + points_per_segment], :].copy()
        sorted_adata.obs["slice"] = slice_id
        split_adata.append(sorted_adata)
    return split_adata[:split_num]


def kmeans_plot_multiple(adatas, k=20, dot_size=10, minibatch=True, plot_strategy='sep', axis_off=False):
    num_plot = len(adatas)
    plot_row = int(np.floor(num_plot / 2) + 1)
    embed_stack = adatas[0].obsm['embd']
    for i in range(1, num_plot):
        embed_stack = np.row_stack((embed_stack, adatas[i].obsm['embd']))
    print(f'Perform KMeans clustering on {embed_stack.shape[0]} cells...')
    kmeans = KMeans(n_clusters=k, random_state=0).fit(embed_stack) if minibatch == False else MiniBatchKMeans(
        n_clusters=k, random_state=0).fit(embed_stack)
    cell_label = kmeans.labels_
    cluster_pl = sns.color_palette('tab20', len(np.unique(cell_label)))
    print(f'Plotting the KMeans clustering results...')
    cell_label_idx = 0
    if plot_strategy == 'sep':
        plt.figure(figsize=((20, 10 * plot_row)))
        for j in range(num_plot):
            plt.subplot(plot_row, 2, j + 1)
            coords0 = adatas[j].obsm['spatial']
            col = coords0[:, 0].tolist()
            row = coords0[:, 1].tolist()
            cell_type_t = cell_label[cell_label_idx:(cell_label_idx + coords0.shape[0])]
            cell_label_idx += coords0.shape[0]
            for i in set(cell_type_t):
                plt.scatter(np.array(col)[cell_type_t == i],
                            np.array(row)[cell_type_t == i], s=dot_size, edgecolors='none',
                            c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]], label=str(i), rasterized=True)
            plt.title('(KMeans, k = ' + str(k) + ')', fontsize=20)
            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)
            plt.axis('equal')
            if axis_off:
                plt.xticks([])
                plt.yticks([])
    else:
        plt.figure(figsize=[10, 12])
        plt.rcParams.update({'font.size': 10, 'axes.titlesize': 20, 'pdf.fonttype': 42})
        for j in range(num_plot):
            coords0 = adatas[j].obsm['spatial']
            col = coords0[:, 0].tolist()
            row = coords0[:, 1].tolist()
            cell_type_t = cell_label[cell_label_idx:(cell_label_idx + coords0.shape[0])]
            cell_label_idx += coords0.shape[0]
            for i in set(cell_type_t):
                plt.scatter(np.array(col)[cell_type_t == i],
                            np.array(row)[cell_type_t == i], s=dot_size, edgecolors='none', alpha=0.5,
                            c=np.array(cluster_pl)[cell_type_t[cell_type_t == i]], label=str(i), rasterized=True)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.axis('equal')
        if axis_off:
            plt.xticks([])
            plt.yticks([])
        plt.title('K means (k = ' + str(k) + ')', fontsize=30)
    return cell_label
