import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import dgl


def delaunay_dgl(spatial_coords, if_plot=True, graph_strategy='convex'):
    coords = np.column_stack((np.array(spatial_coords)[:, 0], np.array(spatial_coords)[:, 1]))
    delaunay_graph = coords2adjacentmat(coords, output_mode='raw', strategy_t=graph_strategy)
    if if_plot:
        positions = dict(zip(delaunay_graph.nodes, coords[delaunay_graph.nodes, :]))
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        nx.draw(
            delaunay_graph,
            positions,
            ax=ax,
            node_size=1,
            node_color="#000000",
            edge_color="#5A98AF",
            alpha=0.6,
        )
        plt.axis('equal')
    return dgl.from_networkx(delaunay_graph)


def coords2adjacentmat(coords, output_mode='adjacent', strategy_t='convex'):
    if strategy_t == 'convex':
        from libpysal.cg import voronoi_frames
        from libpysal import weights

        cells, _ = voronoi_frames(coords, clip="convex hull")
        delaunay_graph = weights.Rook.from_dataframe(cells).to_networkx()
    elif strategy_t == 'delaunay':
        from scipy.spatial import Delaunay
        from collections import defaultdict

        tri = Delaunay(coords)
        delaunay_graph = nx.Graph()
        coords_dict = defaultdict(list)
        for i, coord in enumerate(coords):
            coords_dict[tuple(coord)].append(i)
        for simplex in tri.simplices:
            for i in range(3):
                for node1 in coords_dict[tuple(coords[simplex[i]])]:
                    for node2 in coords_dict[tuple(coords[simplex[(i + 1) % 3]])]:
                        if not delaunay_graph.has_edge(node1, node2):
                            delaunay_graph.add_edge(node1, node2)
    if output_mode == 'adjacent':
        return nx.to_scipy_sparse_array(delaunay_graph).todense()
    elif output_mode == 'raw':
        return delaunay_graph
    elif output_mode == 'adjacent_sparse':
        return nx.to_scipy_sparse_array(delaunay_graph)
