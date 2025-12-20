import numpy as np
import torch
import faiss
import torch.nn as nn
from typing import List, Mapping, Optional, Union, Tuple, Any

from numpy import ndarray
from sklearn.metrics.pairwise import euclidean_distances
from anndata import AnnData


def spatial_match(embes: List[torch.Tensor],
                  reorder: Optional[bool] = True,
                  smooth: Optional[bool] = True,
                  smooth_range: Optional[int] = 20,
                  scale_coord: Optional[bool] = True,
                  adatas: Optional[List[AnnData]] = None):
    r"""
    Use embeddings th match spots from different slices based on inner product (after L2 normalization, equal to
    cosine similarity)
    :param embes: input slice embeddings, or top hvgs
    :param reorder: if reorder embeddings by cell numbers
    :param smooth: if smooth the mapping by Euclid distance
    :param smooth_range: use how many candidates to do smooth
    :param scale_coord: scale the coordinate of spots to [0, 1]
    :param adatas: List of adata object
    :return: best: similarity in embedding space, and also the most proximity in spatial $\in R^{n_{spots} \times 1}$;
    index: The order index of similarity spots $\in R^{n_{spots} \times smooth_{range}$;
    similarity: the cosine similarity of two spots $\in R^{n_{spots} \times smooth_{range}}$.
    """
    if reorder and (embes[0].shape[0] < embes[1].shape[0]):
        embd0 = embes[1]
        embd1 = embes[0]
        adatas = adatas[:: -1] if adatas is not None else None
    else:
        embd0 = embes[0]
        embd1 = embes[1]

    index = faiss.index_factory(embd1.shape[1], 'Flat', faiss.METRIC_INNER_PRODUCT)

    embed0_np = embd0.detach().cpu().numpy() if torch.is_tensor(embd0) else embd0
    embed1_np = embd1.detach().cpu().numpy() if torch.is_tensor(embd1) else embd1
    embed0_np = embed0_np.copy().astype(np.float32)
    embed1_np = embed1_np.copy().astype(np.float32)

    faiss.normalize_L2(embed0_np)
    faiss.normalize_L2(embed1_np)

    index.add(embed0_np)
    similarity, order = index.search(embed1_np, smooth_range)
    adata1_coord = adatas[0].obsm['spatial']
    adata2_coord = adatas[1].obsm['spatial']

    best = []
    if smooth and adatas is not None:
        if scale_coord:
            for i in range(2):
                adata1_coord[:, i] = (adata1_coord[:, i] - np.min(adata1_coord[:, i])) / (np.max(adata1_coord[:, i]) -
                                                                                          np.min(adata1_coord[:, i]))
                adata2_coord[:, i] = (adata2_coord[:, i] - np.min(adata2_coord[:, i])) / (np.max(adata2_coord[:, i]) -
                                                                                          np.min(adata2_coord[:, i]))
        dist_list = []
        for query in range(embed1_np.shape[0]):
            # 针对embed1中的每个表征，构建其在另一切片中的最相似参考表征list
            ref_list = order[query, :smooth_range]
            # 计算query的spot在空间上，与这些最相似的spots之间的欧氏距离
            dis = euclidean_distances(adata2_coord[query, :].reshape(1, -1), adata1_coord[ref_list, :])
            # 1 * smooth_range的向量存入dist_list中
            dist_list.append(dis)
            # 将空间上最接近的spot存入best list中
            best.append(ref_list[np.argmin(dis)])
    else:
        best = order[:, 0]
    return np.array(best), order, similarity


def region_statistics(data_input, step, start, interval):
    intervals = {'{:.3f}~{:.3f}'.format(step * x + start, step * (x + 1) + start): 0 for x in range(interval)}
    for num in data_input:
        for interval in intervals:
            lr = tuple(interval.split('~'))
            left, right = float(lr[0]), float(lr[1])
            if left <= num <= right:
                intervals[interval] = intervals[interval] + 1
    for key, value in intervals.items():
        print("%10s" % key, end='')
        print("%10s" % value, end='')
        print('%16s' % '{:.3%}'.format(value * 1.0 / len(data_input)))
