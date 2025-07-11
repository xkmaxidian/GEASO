from GEASO.utils.utils import cal_distance
from typing import Union
import numpy as np
import torch


def con_K_euc(
        X: Union[np.ndarray, torch.Tensor],
        Y: Union[np.ndarray, torch.Tensor],
        beta: Union[int, float] = 0.01,
) -> Union[np.ndarray, torch.Tensor]:
    assert X.shape[1] == Y.shape[1], "X and Y do not have the same number of features."
    K = cal_distance(
        X=X,
        Y=Y,
        metric="euc",
    )
    K = np.exp(-beta * K)
    return K


def to_torch(x, device='cpu', dtype=torch.float32):
    return torch.from_numpy(x).to(device=device, dtype=dtype) if isinstance(x, np.ndarray) \
        else torch.tensor(x, device=device, dtype=dtype)
