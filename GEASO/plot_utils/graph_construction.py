from typing import Optional, Union, List

import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata import AnnData
from torch_geometric.nn import knn_graph, radius_graph
from torch_geometric.utils import to_undirected
import torch
import scanpy as sc


def cal_spatial_network(
        adata: AnnData,
        rad_cutoff: Optional[Union[None, int]]=None,
        k_cutoff: Optional[Union[None, int]] = None,
        model: Optional[str] = 'Radius',
        return_data: Optional[bool] = False,
        verbose: Optional[bool] = True
) -> None:
    assert(model in ['Radius', 'KNN'])
    if verbose:
        print('Calculating spatial neighbor graph ...')

    if model == 'KNN':
        edge_index = knn_graph(x=torch.tensor(adata.obsm['spatial']), flow='target_to_source', k=k_cutoff, loop=True,
                               num_workers=8)
    if model == 'Radius':
        edge_index = radius_graph(x=torch.tensor(adata.obsm['spatial']), flow='target_to_source', r=rad_cutoff,
                                  loop=True, num_workers=8)

    edge_index = to_undirected(edge_index, num_nodes=adata.shape[0])
    graph_df = pd.DataFrame(edge_index.numpy().T, columns=['Cell1', 'Cell2'])
    id_cell_trans = dict(zip(range(adata.n_obs), adata.obs_names))
    graph_df['Cell1'] = graph_df['Cell1'].map(id_cell_trans)
    graph_df['Cell2'] = graph_df['Cell2'].map(id_cell_trans)
    adata.uns['spatial_network'] = graph_df

    spatial_network = adata.uns['spatial_network']
    cells_id_tran = dict(zip(np.array(adata.obs_names), range(adata.shape[0])))
    adj_df = spatial_network.copy()
    adj_df['Cell1'] = adj_df['Cell1'].map(cells_id_tran)
    adj_df['Cell2'] = adj_df['Cell2'].map(cells_id_tran)
    adj_df = sp.coo_matrix((np.ones(adj_df.shape[0]), (adj_df['Cell1'], adj_df['Cell2'])),
                           shape=(adata.n_obs, adata.n_obs))
    adata.uns['adj'] = adj_df

    if verbose:
        print(f'The graph contains {graph_df.shape[0]} edges, {adata.n_obs} cells')
        print(f'{graph_df.shape[0] / adata.n_obs} neighbors per cell on average')

    if return_data:
        return adata