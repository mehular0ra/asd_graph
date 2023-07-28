import torch
from .construct_hypergraph import HyperGraph

import ipdb

# def create_mini_batch(hgraph_list):
#     """ Built a sparse graph from a batch of graphs
#     Args:
#         graph_list: list of Graph objects in a batch
#     Returns:
#         a big (sparse) Graph representing the entire batch
#     """
#     # insert first graph into the structure
#     ipdb.set_trace()
#     batch_edge_index = hgraph_list[0].hyper_edge_index
#     batch_x = hgraph_list[0].x
#     batch_y = [hgraph_list[0].y.item()]
#     batch_batch = torch.zeros((hgraph_list[0].num_nodes), dtype=torch.int64)
#     num_nodes = hgraph_list[0].num_nodes
#     num_hyper_edges = hgraph_list[0].num_hyper_edges

#     # append the rest of the graphs to the structure
#     for idx, graph in enumerate(hgraph_list[1:]):
#         # concat the features
#         batch_x = torch.concat([batch_x, graph.x], dim=0)
#         # concat the labels
#         batch_y.append(graph.y.item())

#         # concat the adjacency matrix as a block diagonal matrix
#         current_hyper_edge_index = graph.hyper_edge_index[1,
#                                                           :] + num_hyper_edges
#         current_node_index = graph.hyper_edge_index[0, :] + num_nodes
#         current_index = torch.concat([current_node_index.unsqueeze(
#             0), current_hyper_edge_index.unsqueeze(0)], dim=0)
#         num_nodes += graph.num_nodes
#         num_hyper_edges += graph.num_hyper_edges

#         batch_edge_index = torch.concat(
#             [batch_edge_index, current_index], dim=1)
#         # ==========================================

#         # create the array of indexes mapping nodes in the batch-graph
#         # to the graph they belong to
#         # specify the mapping between the new nodes and the graph they belong to (idx+1)
#         batch_batch = torch.concat(
#             [batch_batch, (idx+1)*torch.ones([graph.num_nodes]).to(torch.int64)])
#         # ==========================================
#         pass

#     # create the big sparse graph
#     batch_graph = HyperGraph(batch_edge_index, batch_x, torch.tensor(batch_y))
#     # attach the index array to the Graph structure
#     batch_graph.set_batch(batch_batch)
#     # print(batch_batch.dtype)
#     return batch_graph


def create_mini_batch(hgraph_list):
    """ Built a batch of hypergraphs
    Args:
        hgraph_list: list of HyperGraph objects in a batch
    Returns:
        a batch of HyperGraphs
    """

    # store batch data in lists
    batch_fc = []
    batch_sc = []
    batch_t1 = []
    batch_x = []
    batch_y = []
    batch_k = []

    # append data from each hypergraph in the batch
    for hgraph in hgraph_list:
        batch_fc.append(hgraph.fc)
        batch_sc.append(hgraph.sc)
        batch_t1.append(hgraph.t1)
        batch_x.append(hgraph.x)
        batch_y.append(hgraph.y)
        batch_k.append(hgraph.k)

    # convert lists to tensors for PyTorch
    batch_fc = torch.cat(batch_fc, dim=0) if batch_fc[0] is not None else None
    batch_sc = torch.cat(batch_sc, dim=0) if batch_sc[0] is not None else None
    batch_t1 = torch.cat(batch_t1, dim=0) if batch_t1[0] is not None else None
    batch_x = torch.cat(batch_x, dim=0)
    batch_y = torch.tensor(batch_y)

    # all HyperGraphs in the batch share the same k
    batch_k = batch_k[0]

    # create the big batch HyperGraph
    batch_hgraph = HyperGraph(
        batch_fc, batch_sc, batch_t1, batch_x, batch_y, batch_k)

    return batch_hgraph
