import networkx as nx
import numpy as np
import torch
from typing import List, Optional, Union
from anndata import AnnData
from GEASO.alignment.dijkstra import con_K_geo
from GEASO.utils.utils import cal_distance, cal_probability
from GEASO.alignment.utils import con_K_euc, to_torch
from tqdm import trange, tqdm

'''
    Non-rigid alignment of two slices is mainly inspired by these papers:
    1. Point Set Registration: Coherent Point Drift (CPD, TPAMI 2010)
    2. A Bayesian Formulation of Coherent Point Drift (BCPD, TPAMI 2020)
    3. Geodesic-Based Bayesian Coherent Point Drift (GBCPD, TPAMI 2022)
'''


class ElasticRegistration:
    def __init__(
            self,
            normalize_scales,
            normalize_means,
            init_P,
            guided_source,
            guided_target,
            source_slice: AnnData,
            target_slice: AnnData,
            nn_init_weight: float = 0.1,
            rep_layer: str = 'latent',
            rep_field: str = 'obsm',
            genes: Optional[Union[List[str], np.ndarray, torch.Tensor]] = None,
            spatial_key: str = 'spatial_coarse',
            spatial_add: str = 'spatial_aligned',
            dissimilarity: str = 'kl',
            probability_type: str = 'gaussian',
            probability_parameters: float = None,
            allow_flip: bool = False,
            max_iter: int = 200,

            non_rigid_start_iter: int = 80,
            SVI_mode: bool = True,
            batch_size: Optional[int] = None,
            pre_comp_dist: bool = True,
            sparse_calculation_mode: bool = False,
            sparse_top_k: int = 1024,
            lambda_VF: Union[int, float] = 1e2,
            beta: Union[int, float] = 1e0,

            K: int = 100,
            kernel_bandwidth: float = 1e-2,
            graph_knn: int = 10,
            gamma: float = 0.5,

            kappa: Union[float, np.ndarray] = 1.0,
            partial_robust_level: float = 10,
            normalize_spatial: bool = True,
            separate_mean: bool = True,
            separate_scale: bool = False,
            dtype: str = torch.float32,
            device: str = 'cpu',
            verbose: bool = True,
            return_mapping: bool = False,

            iter_key_added: Optional[str] = None,
    ):
        self.nn_init_weight = nn_init_weight
        self.init_P = init_P
        self.source_slice = source_slice
        self.target_slice = target_slice
        self.rep_layer = rep_layer
        self.rep_field = rep_field
        self.genes = genes
        self.spatial_key = spatial_key
        self.spatial_add = spatial_add
        self.dissimilarity = dissimilarity
        self.probability_type = probability_type
        self.probability_parameters = probability_parameters
        self.allow_flip = allow_flip
        self.max_iter = max_iter

        self.non_rigid_start_iter = non_rigid_start_iter
        self.SVI_mode = SVI_mode
        self.batch_size = batch_size
        self.pre_comp_dist = pre_comp_dist
        self.sparse_calculation_mode = sparse_calculation_mode
        self.sparse_top_k = sparse_top_k
        self.lambda_VF = lambda_VF
        self.beta = beta
        self.K = K
        self.kernel_bandwidth = kernel_bandwidth
        self.gamma = gamma

        self.graph_knn = graph_knn
        self.kappa = kappa
        self.partial_robust_level = partial_robust_level
        self.normalize_spatial = normalize_spatial
        self.separate_mean = separate_mean
        self.separate_scale = separate_scale
        self.dtype = dtype
        self.device = device
        self.verbose = verbose
        self.return_mapping = return_mapping
        self.iter_key_added = iter_key_added
        self.normalize_scales = normalize_scales
        self.normalize_means = normalize_means
        self.guided_source = guided_source
        self.guided_target = guided_target

        valid_metrics = ["kl", "sym_kl", "euc", "euclidean", "square_euc", "square_euclidean", "cos", "cosine"]
        self.dissimilarity = self.dissimilarity.lower()
        if self.dissimilarity not in valid_metrics:
            raise ValueError(
                f"Invalid `metric` value: {self.dissimilarity}. Available `metrics` are: " f"{', '.join(valid_metrics)}."
            )

        if self.probability_type is None:
            self.probability_type = "gauss"

        valid_metrics = ["gauss", "gaussian", "cos", "cosine", "prob"]
        self.probability_type = self.probability_type
        if self.probability_type not in valid_metrics:
            raise ValueError(f"Invalid `metric` value: {probability_type}. Available `metrics` are: " f"{', '.join(valid_metrics)}.")

        if self.sparse_calculation_mode:
            self.pre_comp_dist = False

        # construct motion kernel
        self.construct_kernel(inducing_num=K)
        # initialize variables
        if self.rep_field == 'layer':
            self.source_exp = self.source_slice.layers[self.rep_layer]
            self.target_exp = self.target_slice.layers[self.rep_layer]
        elif self.rep_field == 'obsm':
            self.source_exp = self.source_slice.obsm[self.rep_layer]
            self.target_exp = self.target_slice.obsm[self.rep_layer]

        self.coords_source = self.source_slice.obsm[spatial_key]
        self.coords_target = self.target_slice.obsm[spatial_key]

        self.initialize_variation_variables(
            source_coord=self.coords_source,
            target_coord=self.coords_target,
            source_exp=self.source_exp,
            target_exp=self.target_exp,
        )

        # convert data type to torch
        self.coords_source = torch.from_numpy(self.coords_source).to(device=device, dtype=torch.float32)
        self.coords_target = torch.from_numpy(self.coords_target).to(device=device, dtype=torch.float32)
        self.source_exp = torch.from_numpy(self.source_exp).to(device=device, dtype=torch.float32)
        self.target_exp = torch.from_numpy(self.target_exp).to(device=device, dtype=torch.float32)

        inlier_threshold = min(self.init_P[np.argsort(-self.init_P[:, 0])[20], 0], 0.5)
        inlier_set = np.where(self.init_P[:, 0] > inlier_threshold)[0]
        self.inlier_P = self.init_P[inlier_set, :]
        self.inlier_source = self.guided_source[inlier_set, :]
        self.inlier_target = self.guided_target[inlier_set, :]
        self.inlier_P = to_torch(self.inlier_P, device=self.device, dtype=self.dtype)
        self.inlier_source = to_torch(self.inlier_source, device=self.device, dtype=self.dtype)
        self.inlier_target = to_torch(self.inlier_target, device=self.device, dtype=self.dtype)

        # execute elastic registration, assume the coarse alignment has been done
        self.P = self.elastic_registratin()
        self.source_slice.obsm['aligned_spatial_rigid'] = self.optimal_transformed
        self.source_slice.obsm['aligned_spatial_nonrigid'] = self.source_transformed

    def construct_kernel(
            self,
            inducing_num,
    ):
        unique_coords, unique_idx = np.unique(self.source_slice.obsm[self.spatial_key], return_index=True, axis=0)
        if unique_coords.shape[0] > inducing_num:
            inducing_variables_idx = np.random.choice(unique_coords.shape[0], inducing_num, replace=False)
        else:
            inducing_variables_idx = np.arange(unique_coords.shape[0])

        inducing_variables_idx = unique_idx[inducing_variables_idx]
        inducing_variables = self.source_slice.obsm[self.spatial_key][inducing_variables_idx, :]
        geo_U = con_K_geo(X=self.source_slice.obsm[self.spatial_key],
                          Y_idx=inducing_variables_idx, beta=self.kernel_bandwidth, k=self.graph_knn)
        euc_U = con_K_euc(X=self.source_slice.obsm[self.spatial_key],
                          Y=inducing_variables, beta=self.kernel_bandwidth)
        self.GXU = (1 - self.gamma) * euc_U + self.gamma * geo_U
        self.GUU = self.GXU[inducing_variables_idx, :]
        self.K = inducing_variables.shape[0]
        self.GXU = to_torch(self.GXU, device=self.device, dtype=self.dtype)
        self.GUU = to_torch(self.GUU, device=self.device, dtype=self.dtype)
        print('Local structure kernel constructed with shape:', self.GXU.shape)

    def update_batch(self, epoch):
        decay_val = self.SVI_decay / (epoch + 1.0)
        one = torch.tensor(1.0, device=self.SVI_decay.device, dtype=self.SVI_decay.dtype)
        decay_val = torch.tensor(decay_val, device=self.SVI_decay.device, dtype=self.SVI_decay.dtype)
        self.step_size = torch.min(one, decay_val)
        self.batch_idx = self.batch_perm[: self.batch_size]
        self.batch_idx = self.batch_idx.cpu().numpy().astype(int)
        self.batch_perm = torch.roll(self.batch_perm, self.batch_size)

    def elastic_registratin(self):
        # record results of each iteration
        # such as: if coarse, self.init_params = {'sigma2': sigma2, 'rotation': rotation, 'translation': translation}
        # 然后传入self.rotation, self.translation, self.scale, self.sigma2等
        if (not self.SVI_mode) or (self.pre_comp_dist):
            self.exp_layer_dist = cal_distance(X=self.source_exp.cpu().numpy(), Y=self.target_exp.cpu().numpy(), metric=self.dissimilarity)
            self.exp_layer_dist = to_torch(self.exp_layer_dist, device=self.device, dtype=self.dtype)

        if self.iter_key_added is not None:
            self.iter_added = dict()
            self.iter_added[self.spatial_add] = {}
            self.iter_added['sigma2'] = {}

        epochs = trange(
            self.max_iter,
            desc="Slice Alignment",
            leave=True,
            disable=not self.verbose,
        )

        for epoch in epochs:
            if self.SVI_mode:
                self.update_batch(epoch=epoch)

            self.update_assignment_P()
            self.update_gamma()
            self.update_alpha()

            if (epoch > self.non_rigid_start_iter) or (self.deformation_flag):
                self.deformation_flag = True
                self.update_non_rigid()

            # optimize both rigid and non-rigid parameters
            self.update_rigid()
            self.source_transformed = self.displacement_source + self.source_transformed_temp
            self.update_sigma2(epoch=epoch)

        if self.return_mapping and self.SVI_mode:
            self.SVI_mode = False
            self.update_assignment_P()

        self.get_optimal_R()
        self.wrap_out()
        return self.P

    def initialize_variation_variables(
            self,
            source_coord,
            target_coord,
            source_exp,
            target_exp,
            n_subsample=20000,
    ):
        n_source, n_target, n_dims = (source_coord.shape[0], target_coord.shape[0],
                                      source_coord.shape[1])
        sub_source = np.random.choice(n_source, n_subsample,
                                      replace=False) if n_source > n_subsample else np.arange(n_source)
        sub_target = np.random.choice(n_target, n_subsample,
                                      replace=False) if n_target > n_subsample else np.arange(n_target)
        spatial_dist = cal_distance(X=source_coord[sub_source], Y=target_coord[sub_target], metric='euclidean')
        self.sigma2 = ((spatial_dist ** 2).sum() / (n_dims * sub_source.shape[0] * sub_target.shape[0])) * 0.1
        self.sigma2 = to_torch(self.sigma2, device=self.device, dtype=self.dtype)

        if self.probability_type.lower() in ['gauss',  'gaussian']:
            exp_dist = cal_distance(X=source_exp[sub_source], Y=target_exp[sub_target], metric=self.dissimilarity)
            min_exp_dist = np.min(exp_dist, axis=1)
            self.probability_parameters = np.max([min_exp_dist[np.argsort(min_exp_dist)[int(sub_source.shape[0] * 0.05)]] / 5, 0.01])
            self.probability_parameters = to_torch(self.probability_parameters, device=self.device, dtype=self.dtype)

        self.sigma2_variance = 1
        self.sigma2_variance_end = self.partial_robust_level
        self.sigma2_variance_decrease = get_annealing_factor(
            start=self.sigma2_variance,
            end=self.sigma2_variance_end,
            max_iter=100,
        )
        self.sigma2_variance = to_torch(self.sigma2_variance, device=self.device, dtype=self.dtype)
        self.sigma2_variance_end = to_torch(self.sigma2_variance_end, device=self.device, dtype=self.dtype)
        self.sigma2_variance_decrease = to_torch(self.sigma2_variance_decrease, device=self.device, dtype=self.dtype)

        self.kappa = np.ones([n_source]) * 1.0
        self.alpha = np.ones([n_source])
        self.gamma, self.gamma_a, self.gamma_b = (np.array(0.5), np.array(1.0), np.arange(1.0))
        self.kappa = to_torch(self.kappa, device=self.device, dtype=self.dtype)
        self.alpha = to_torch(self.alpha, device=self.device, dtype=self.dtype)
        self.gamma = to_torch(self.gamma, device=self.device, dtype=self.dtype)
        self.gamma_a = to_torch(self.gamma_a, device=self.device, dtype=self.dtype)
        self.gamma_b = to_torch(self.gamma_b, device=self.device, dtype=self.dtype)

        self.displacement_source = np.zeros(source_coord.shape)
        self.source_transformed = source_coord.copy()
        self.source_transformed_temp = source_coord.copy()

        self.displacement_source = to_torch(self.displacement_source, device=self.device, dtype=self.dtype)
        self.source_transformed = to_torch(self.source_transformed, device=self.device, dtype=self.dtype)
        self.source_transformed_temp = to_torch(self.source_transformed_temp, device=self.device, dtype=self.dtype)

        self.sample_s = np.max([
            np.prod(np.max(source_coord, axis=0) - np.min(source_coord, axis=0)),
            np.prod(np.max(target_coord, axis=0) - np.min(target_coord, axis=0)),
        ])
        self.outlier_s = self.sample_s * n_source
        self.coff = np.zeros(self.K)
        self.D = target_coord.shape[1]
        self.sample_s = to_torch(self.sample_s, device=self.device, dtype=self.dtype)
        self.outlier_s = to_torch(self.outlier_s, device=self.device, dtype=self.dtype)
        self.coff = to_torch(self.coff, device=self.device, dtype=self.dtype)

        self.sigma_diag = np.zeros([n_source])
        self.sigma_diag = to_torch(self.sigma_diag, device=self.device, dtype=self.dtype)

        self.rotation = np.identity(n_dims)
        self.C = np.identity(n_dims)
        self.rotation = to_torch(self.rotation, device=self.device, dtype=self.dtype)
        self.C = to_torch(self.C, device=self.device, dtype=self.dtype)

        self.gamma1 = to_torch(0.01, device=self.device, dtype=self.dtype)
        self.gamma99 = to_torch(0.99, device=self.device, dtype=self.dtype)
        self.deformation_flag = False

        if self.SVI_mode:
            self.SVI_decay = to_torch(10.0, device=self.device, dtype=self.dtype)

            if self.batch_size is None:
                self.batch_size = min(max(int(target_coord.shape[0] / 10), 1000), target_coord.shape[0])
            else:
                self.batch_size = min(self.batch_size, target_coord.shape[0])
            self.batch_perm = np.random.permutation(target_coord.shape[0])
            self.batch_perm = to_torch(self.batch_perm, device=self.device, dtype=self.dtype)
            self.Sp, self.Sp_spatial, self.Sp_sigma2 = (0., 0., 0.)
            self.PXB_term = np.zeros([n_source, n_dims])
            self.SigmaInv = np.zeros([self.K, self.K])

            self.Sp = to_torch(self.Sp, device=self.device, dtype=self.dtype)
            self.Sp_spatial = to_torch(self.Sp_spatial, device=self.device, dtype=self.dtype)
            self.Sp_sigma2 = to_torch(self.Sp_sigma2, device=self.device, dtype=self.dtype)
            self.PXB_term = to_torch(self.PXB_term, device=self.device, dtype=self.dtype)
            self.SigmaInv = to_torch(self.SigmaInv, device=self.device, dtype=self.dtype)

    def update_assignment_P(self):
        model_mul = torch.unsqueeze(
            self.alpha * torch.exp(-self.sigma_diag / self.sigma2), dim=-1)
        common_kwargs = dict(
            Dim=self.D,
            sigma2=self.sigma2,
            model_mul=model_mul,
            gamma=self.gamma,
            samples_s=self.sample_s,
            sigma2_variance=self.sigma2_variance,
            probability_type=self.probability_type,
            probability_parameters=self.probability_parameters,
        )

        spatial_dist = cal_distance(
            X=self.source_transformed,
            Y=self.coords_target[self.batch_idx, ] if self.SVI_mode else self.coords_target,
            metric="euc",
        )
        if self.pre_comp_dist:
            exp_layer_dist = (
                self.exp_layer_dist[:, self.batch_idx]
                if self.SVI_mode
                else self.exp_layer_dist
            )
        else:
            exp_layer_dist = cal_distance(
                X=self.source_exp,
                Y=self.target_exp[self.batch_idx] if self.SVI_mode else self.target_exp,
                metric=self.dissimilarity,
            )

        self.P, self.K_NA_spatial, self.K_NA_sigma2, sigma2_related = get_P_core(
            spatial_dist=spatial_dist, exp_dist=exp_layer_dist, **common_kwargs
        )

        Sp = torch.sum(self.P)
        Sp_sigma2 = torch.sum(self.K_NA_sigma2)
        Sp_spatial = torch.sum(self.K_NA_spatial)
        self.K_NA = torch.sum(self.P, dim=1)
        self.K_NB = torch.sum(self.P, dim=0)

        if self.SVI_mode:
            self.Sp_spatial = self.step_size * Sp_spatial + (1 - self.step_size) * self.Sp_spatial
            self.Sp = self.step_size * Sp + (1 - self.step_size) * self.Sp
            self.Sp_sigma2 = self.step_size * Sp_sigma2 + (1 - self.step_size) * self.Sp_sigma2
        else:
            self.Sp_spatial = Sp_spatial
            self.Sp = Sp
            self.Sp_sigma2 = Sp_sigma2

        self.sigma2_related = sigma2_related / (self.D * self.Sp_sigma2)

    def update_gamma(self):
        if self.SVI_mode:
            total = self.gamma_a + self.gamma_b + self.batch_size
        else:
            total = self.gamma_a + self.gamma_b + self.target_slice.shape[0]

        digamma_diff = torch.special.digamma(self.gamma_a + self.Sp_spatial) - torch.special.digamma(total)
        new_gamma = torch.exp(digamma_diff)
        self.gamma = torch.clamp(new_gamma, min=self.gamma1, max=self.gamma99)

    def update_alpha(self):
        term = torch.special.digamma(self.kappa + self.K_NA_spatial) \
               - torch.special.digamma(self.kappa * self.source_slice.shape[0] + self.Sp_spatial)
        if self.SVI_mode:
            updated = torch.exp(term)
            self.alpha = self.step_size * updated + (1 - self.step_size) * self.alpha
        else:
            self.alpha = torch.exp(term)

    def update_non_rigid(self):
        term = torch.einsum("ij,i->ij", self.GXU, self.K_NA)
        SigmaInv = (self.sigma2 * self.lambda_VF * self.GUU + torch.matmul(self.GXU.T, term))

        if self.SVI_mode:
            coords_target_batch = self.coords_target[self.batch_idx]
            PXB_term = torch.matmul(self.P, coords_target_batch) - torch.einsum("ij,i->ij",
                                                                                self.source_transformed_temp, self.K_NA)
            # exponential moving average
            self.SigmaInv = (self.step_size * SigmaInv + (1 - self.step_size) * self.SigmaInv)
            self.PXB_term = (self.step_size * PXB_term + (1 - self.step_size) * self.PXB_term)
        else:
            PXB_term = torch.matmul(self.P, self.coords_target) - term.matmul(self.source_transformed_temp)
            self.SigmaInv = SigmaInv
            self.PXB_term = PXB_term

        UPXB_term = torch.matmul(self.GXU.T, self.PXB_term)
        Sigma = torch.linalg.pinv(self.SigmaInv)
        self.coff = torch.matmul(Sigma, UPXB_term)
        self.displacement_source = torch.matmul(self.GXU, self.coff)

        tmp = torch.matmul(Sigma, self.GXU.T)
        full = torch.matmul(self.GXU, tmp)
        self.sigma_diag = self.sigma2 * torch.diagonal(full, 0)

    def update_rigid(self):
        PXA, PVA, PXB = (
            torch.matmul(self.K_NA.unsqueeze(0), self.coords_source),
            torch.matmul(self.K_NA.unsqueeze(0), self.displacement_source),
            torch.matmul(self.K_NB.unsqueeze(0), self.coords_target[self.batch_idx])
            if self.SVI_mode
            else torch.matmul(self.K_NB.unsqueeze(0), self.coords_target)
        )
        # solve rotation using SVD formula
        denom = self.Sp
        init_coef = self.sigma2 * self.nn_init_weight * self.Sp / torch.sum(self.inlier_P)
        # PXB, PXA, denom all get an nn_init term
        PXB = PXB + init_coef * torch.matmul(self.inlier_P.T, self.inlier_target)
        PXA = PXA + init_coef * torch.matmul(self.inlier_P.T, self.inlier_source)
        denom = denom + init_coef * torch.sum(self.inlier_P)
        mu_XB = PXB / denom
        mu_XA = PXA / denom
        mu_Vn = PVA / denom

        XA_hat = self.coords_source - mu_XA
        VnA_hat = self.displacement_source - mu_Vn
        XB_hat = (self.coords_target[self.batch_idx] - mu_XB) if self.SVI_mode else (self.coords_target - mu_XB)

        term1 = torch.matmul((XA_hat * self.K_NA.unsqueeze(1)).T, VnA_hat)
        term2 = torch.matmul(XA_hat.T, torch.matmul(self.P, XB_hat))
        A = -(term1 - term2).T
        U, S, Vh = torch.linalg.svd(A)
        self.C[-1, -1] = torch.det(torch.matmul(U, Vh))
        R_new = torch.matmul(U @ self.C, Vh)

        if self.SVI_mode and self.step_size < 1.0:
            self.rotation = self.step_size * R_new + (1 - self.step_size) * self.rotation
        else:
            self.rotation = R_new

        # 9) Solve for translation: t = (PXB - PVA - PXA Rᵀ) / denom
        t_numer = PXB - PVA - torch.matmul(PXA, self.rotation.T)
        t_new = t_numer / denom
        if self.SVI_mode and self.step_size < 1.0:
            self.t = self.step_size * t_new + (1 - self.step_size) * self.t
        else:
            self.t = t_new

        self.source_transformed_temp = torch.matmul(self.coords_source, self.rotation.T) + self.t
        self.inlier_R = torch.matmul(self.inlier_source, self.rotation.T) + self.t

    def update_sigma2(self, epoch):
        dot = torch.einsum("i,i->", self.K_NA_sigma2, self.sigma_diag)
        q = self.sigma2_related + dot / self.Sp_sigma2

        # 2) Enforce a lower bound of 1e-3
        min_val = torch.tensor(1e-3, device=q.device, dtype=q.dtype)
        self.sigma2 = torch.maximum(q, min_val)

        # 3) Decay sigma2_variance with an upper bound
        self.sigma2_variance = torch.minimum(
            self.sigma2_variance * self.sigma2_variance_decrease,
            self.sigma2_variance_end
        )
        if epoch < 100:
            min_val2 = torch.tensor(1e-2, device=self.sigma2.device, dtype=self.sigma2.dtype)
            self.sigma2 = torch.maximum(self.sigma2, min_val2)

    def get_optimal_R(self):
        mu_XnA = torch.matmul(self.K_NA, self.coords_source) / self.Sp
        if self.SVI_mode:
            mu_XnB = torch.matmul(self.K_NB, self.coords_target[self.batch_idx]) / self.Sp
        else:
            mu_XnB = torch.matmul(self.K_NB, self.coords_target) / self.Sp

        # 2) Center the coordinates
        XnABar = self.coords_source - mu_XnA
        if self.SVI_mode:
            XnBBar = self.coords_target[self.batch_idx] - mu_XnB
        else:
            XnBBar = self.coords_target - mu_XnB

        A = torch.matmul(torch.matmul(self.P, XnBBar).T, XnABar)
        U, S, Vh = torch.linalg.svd(A, full_matrices=True)
        self.C[-1, -1] = torch.linalg.det(torch.matmul(U, Vh))

        self.optimal_R = torch.matmul(torch.matmul(U, self.C), Vh)
        self.optimal_t = mu_XnB - torch.matmul(mu_XnA, self.optimal_R.T)
        self.optimal_transformed = torch.matmul(self.coords_source, self.optimal_R.T) + self.optimal_t

    def wrap_out(self):
        self.source_transformed = self.source_transformed.detach().cpu().numpy().copy()
        self.optimal_transformed = self.optimal_transformed.detach().cpu().numpy().copy()
        self.source_transformed_temp = self.source_transformed_temp.detach().cpu().numpy().copy()
        self.P = self.P.detach().cpu().numpy().copy()

        if self.normalize_spatial:
            self.source_transformed = self.source_transformed * self.normalize_scales[1] + self.normalize_means[1]
            self.source_transformed_temp = self.source_transformed_temp * self.normalize_scales[1] + self.normalize_means[1]
            self.optimal_transformed = self.optimal_transformed * self.normalize_scales[1] + self.normalize_means[1]

def get_annealing_factor(start, end, max_iter):
    factor = np.power(end / start, 1 / max_iter)
    return factor


def get_P_core(
        Dim: torch.Tensor,
        spatial_dist: torch.Tensor,
        exp_dist: [torch.Tensor],
        sigma2: float,
        model_mul: torch.Tensor,
        gamma: torch.Tensor,
        samples_s: Optional[List[float]] = None,
        sigma2_variance: float = 1,
        probability_type: Union[str, List[str]] = "Gauss",
        probability_parameters: Optional[List] = None,
        eps: float = 1e-8,
):
    """
    Compute assignment matrix P and additional results based on given distances and parameters.
    """

    spatial_prob = cal_probability(
        distance_matrix=spatial_dist, probability_type="gauss", probability_parameter=sigma2 / sigma2_variance)
    outlier_s = samples_s * spatial_dist.shape[0]
    # outlier_s = samples_s
    spatial_outlier = torch.pow((2 * torch.pi * sigma2), (Dim / 2)) * (1 - gamma) / (gamma * outlier_s)  # scalar

    denom = spatial_outlier + torch.sum(spatial_prob, dim=0, keepdim=True)
    spatial_inlier = 1 - spatial_outlier / denom

    # spatial P
    spatial_prob = spatial_prob * model_mul
    P_spatial = spatial_prob / denom
    K_NA_spatial = torch.sum(P_spatial, dim=1)

    spatial_prob = cal_probability(
        distance_matrix=spatial_dist, probability_type="gauss", probability_parameter=sigma2) * model_mul

    denom2 = torch.sum(spatial_prob, dim=0, keepdim=True) + eps
    P_sigma2 = spatial_inlier * spatial_prob / denom2
    K_NA_sigma2 = torch.sum(P_sigma2, dim=1)
    sigma2_related = torch.sum(P_sigma2 * spatial_dist)

    # Calculate probabilities for expression distances
    prob = spatial_prob.clone()
    prob = prob * cal_probability(
        distance_matrix=exp_dist,
        probability_type=probability_type,
        probability_parameter=probability_parameters
    )
    denom3 = torch.sum(prob, dim=0, keepdim=True) + eps
    P_final = spatial_inlier * prob / denom3

    return P_final, K_NA_spatial, K_NA_sigma2, sigma2_related
