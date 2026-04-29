from typing import Optional

import numpy as np
from anndata import AnnData

from GEASO.alignment.coarse_alignment import coarse_alignment
from GEASO.alignment.data_interpolate import interpolate_displacement_field
from GEASO.alignment.elastic_registration import ElasticRegistration


def _to_numpy_matrix(x):
    if hasattr(x, "A"):
        return x.A
    if hasattr(x, "toarray"):
        return x.toarray()
    return np.asarray(x)


def _aggregate_by_voxel(coords, matrices, voxel_num=5000, voxel_size=None):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2:
        raise ValueError("coords must be a 2D array.")

    n_points, n_dims = coords.shape
    if n_points == 0:
        raise ValueError("Cannot downsample an empty coordinate array.")

    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    span = max_coords - min_coords

    if voxel_size is None:
        steps_per_axis = max(int(np.ceil(float(voxel_num) ** (1.0 / n_dims))), 1)
        voxel_size = span / steps_per_axis
        voxel_size = np.where(voxel_size > 0, voxel_size, 1.0)
        keys = np.floor((coords - min_coords) / voxel_size).astype(np.int64)
        keys = np.clip(keys, 0, steps_per_axis - 1)
    else:
        voxel_size = np.asarray(voxel_size, dtype=np.float64)
        if voxel_size.ndim == 0:
            voxel_size = np.repeat(voxel_size, n_dims)
        voxel_size = np.where(voxel_size > 0, voxel_size, 1.0)
        keys = np.floor((coords - min_coords) / voxel_size).astype(np.int64)

    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_voxels = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)

    voxel_coords = np.zeros((n_voxels, n_dims), dtype=np.float64)
    np.add.at(voxel_coords, inverse, coords)
    voxel_coords /= counts[:, None]

    aggregated = {}
    for name, matrix in matrices.items():
        matrix = _to_numpy_matrix(matrix)
        if matrix.ndim == 1:
            matrix = matrix[:, None]
        if matrix.shape[0] != n_points:
            raise ValueError(f"Matrix '{name}' has {matrix.shape[0]} rows, expected {n_points}.")
        out = np.zeros((n_voxels, matrix.shape[1]), dtype=np.float64)
        np.add.at(out, inverse, matrix)
        aggregated[name] = out / counts[:, None]

    return voxel_coords, aggregated, inverse


def _make_lowres_adata(
        adata,
        spatial_key="spatial",
        rep_layer="latent",
        rep_field="obsm",
        voxel_num=5000,
        voxel_size=None,
):
    matrices = {"X": _to_numpy_matrix(adata.X)}
    rep_name = None

    if rep_field == "obsm" and rep_layer in adata.obsm:
        rep_name = f"obsm:{rep_layer}"
        matrices[rep_name] = adata.obsm[rep_layer]
    elif rep_field == "layer" and rep_layer in adata.layers:
        rep_name = f"layer:{rep_layer}"
        matrices[rep_name] = adata.layers[rep_layer]

    coords, aggregated, inverse = _aggregate_by_voxel(
        coords=adata.obsm[spatial_key],
        matrices=matrices,
        voxel_num=voxel_num,
        voxel_size=voxel_size,
    )

    low = AnnData(X=aggregated["X"], var=adata.var.copy())
    low.obsm[spatial_key] = coords
    low.obs_names = [f"voxel_{i}" for i in range(coords.shape[0])]

    if rep_field == "obsm":
        low.obsm[rep_layer] = aggregated[rep_name] if rep_name is not None else aggregated["X"]
    elif rep_field == "layer":
        low.layers[rep_layer] = aggregated[rep_name] if rep_name is not None else aggregated["X"]
    else:
        raise ValueError("rep_field must be either 'obsm' or 'layer'.")

    low.uns["coarse_to_fine"] = {
        "voxel_inverse": inverse,
        "source_size": int(adata.n_obs),
        "lowres_size": int(low.n_obs),
    }
    return low


def _normalize_with_reference(coords, means, scales, slice_index):
    return (np.asarray(coords, dtype=np.float64) - means[slice_index]) / scales[slice_index]


def _to_numpy_tensor(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def coarse_to_fine_alignment(
        source,
        target,
        spatial_key="spatial",
        rep_layer="latent",
        rep_field="obsm",
        source_voxel_num=5000,
        target_voxel_num: Optional[int] = None,
        source_voxel_size=None,
        target_voxel_size=None,
        top_K=10,
        dis_metric="kl",
        use_latent: Optional[bool] = None,
        scale_c=False,
        interpolation_method="knn",
        interpolation_k=32,
        interpolation_bandwidth=None,
        interpolation_chunk_size=50000,
        spatial_add="aligned_spatial_nonrigid",
        rigid_add="aligned_spatial_rigid",
        coarse_add="spatial_coarse",
        low_spatial_key="spatial_coarse",
        inplace=True,
        return_info=False,
        random_state: Optional[int] = 0,
        **elastic_kwargs,
):
    """Align large slices by low-resolution registration plus displacement interpolation.

    The function first voxelizes source and target slices, runs the existing
    coarse alignment and ElasticRegistration on the low-resolution control
    points, then interpolates the learned low-resolution non-rigid displacement
    field back to every source point.
    """
    if target_voxel_num is None:
        target_voxel_num = source_voxel_num

    if not inplace:
        source = source.copy()

    if use_latent is None:
        use_latent = (
            rep_field == "obsm"
            and rep_layer in source.obsm
            and rep_layer in target.obsm
        )

    rng_state = None
    if random_state is not None:
        rng_state = np.random.get_state()
        np.random.seed(random_state)

    try:
        low_source = _make_lowres_adata(
            source,
            spatial_key=spatial_key,
            rep_layer=rep_layer,
            rep_field=rep_field,
            voxel_num=source_voxel_num,
            voxel_size=source_voxel_size,
        )
        low_target = _make_lowres_adata(
            target,
            spatial_key=spatial_key,
            rep_layer=rep_layer,
            rep_field=rep_field,
            voxel_num=target_voxel_num,
            voxel_size=target_voxel_size,
        )

        R, t, scale, init_P, scales, means, guided_source, guided_target, coarse_source, coarse_target = (
            coarse_alignment(
                low_source,
                low_target,
                top_K=top_K,
                dis_metric=dis_metric,
                use_latent=use_latent,
                scale_c=scale_c,
            )
        )

        low_source.obsm[low_spatial_key] = coarse_source
        low_target.obsm[low_spatial_key] = coarse_target

        elastic_options = dict(
            rep_layer=rep_layer,
            rep_field=rep_field,
            spatial_key=low_spatial_key,
            spatial_add=spatial_add,
            normalize_spatial=True,
        )
        elastic_options.update(elastic_kwargs)

        registration = ElasticRegistration(
            normalize_scales=scales,
            normalize_means=means,
            init_P=init_P,
            guided_source=guided_source,
            guided_target=guided_target,
            source_slice=low_source,
            target_slice=low_target,
            **elastic_options,
        )

        source_norm = _normalize_with_reference(source.obsm[spatial_key], means, scales, 0)
        source_coarse_norm = scale * (source_norm @ R.T) + t
        source_coarse = source_coarse_norm * scales[1] + means[1]

        optimal_R = _to_numpy_tensor(registration.optimal_R)
        optimal_t = _to_numpy_tensor(registration.optimal_t)
        source_rigid_norm = source_coarse_norm @ optimal_R.T + optimal_t
        source_rigid = source_rigid_norm * scales[1] + means[1]

        control_points = low_source.obsm["aligned_spatial_rigid"]
        control_displacements = (
            low_source.obsm["aligned_spatial_nonrigid"]
            - low_source.obsm["aligned_spatial_rigid"]
        )
        full_displacements = interpolate_displacement_field(
            control_points=control_points,
            control_displacements=control_displacements,
            query_points=source_rigid,
            method=interpolation_method,
            k=interpolation_k,
            bandwidth=interpolation_bandwidth,
            chunk_size=interpolation_chunk_size,
        )

        source_nonrigid = source_rigid + full_displacements
        source.obsm[coarse_add] = source_coarse
        source.obsm[rigid_add] = source_rigid
        source.obsm[spatial_add] = source_nonrigid

        source.uns["coarse_to_fine_alignment"] = {
            "source_voxel_num": int(source_voxel_num),
            "target_voxel_num": int(target_voxel_num),
            "source_lowres_size": int(low_source.n_obs),
            "target_lowres_size": int(low_target.n_obs),
            "interpolation_method": interpolation_method,
            "interpolation_k": int(interpolation_k),
            "interpolation_bandwidth": (
                None if interpolation_bandwidth is None else float(interpolation_bandwidth)
            ),
        }
    finally:
        if rng_state is not None:
            np.random.set_state(rng_state)

    if return_info:
        return {
            "source": source,
            "target": target,
            "low_source": low_source,
            "low_target": low_target,
            "registration": registration,
            "scales": scales,
            "means": means,
            "R": R,
            "t": t,
            "scale": scale,
            "control_points": control_points,
            "control_displacements": control_displacements,
            "full_displacements": full_displacements,
        }
    return source


__all__ = ["coarse_to_fine_alignment"]
