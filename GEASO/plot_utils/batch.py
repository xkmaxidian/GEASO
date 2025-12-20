from typing import List, Optional

import torch
from torch import Tensor
import numpy as np
from sklearn.utils.extmath import randomized_svd


def dual_pca(x, y, dim=50, singular=False, backend='sklearn', use_gpu=True):
    assert x.shape[1] == y.shape[1]
    device = 'cuda' if torch.cuda.is_available() and use_gpu else 'cpu'
    x = torch.Tensor(x).to(device)
    y = torch.Tensor(y).to(device)
    cor_var = x @ y.T
    if backend == 'torch':
        U, S, VT = torch.linalg.svd(cor_var)
        if not singular:
            return U[:, : dim], VT.T[:, : dim]
        z_x = U[:, : dim] @ torch.sqrt(torch.diag(S[: dim]))
        z_y = VT.T[:, : dim] @ torch.sqrt(torch.diag(S[: dim]))
        return z_x.cpu(), z_y.cpu()
    else:
        cor_var = cor_var.cpu().numpy()
        U, S, VT = randomized_svd(cor_var, n_components=dim, random_state=42)
        if not singular:
            return Tensor(U), Tensor(VT.T)
        z_x = U @ np.sqrt(np.diag(S))
        z_y = VT.T @ np.sqrt(np.diag(S))
        return Tensor(z_x), Tensor(z_y)
