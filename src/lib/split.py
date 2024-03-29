import logging
import random

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import RandomLinkSplit, RandomNodeSplit
from torch_geometric.utils import (
    negative_sampling,
    add_self_loops,
    train_test_split_edges,
    subgraph,
)
import math
from absl import flags


log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
SMALL_DATASETS = set(['cora', 'citeseer'])


def create_mask(base_mask, rows, cols):
    return base_mask[rows] & base_mask[cols]


def split_edges(edge_index, val_ratio, test_ratio):
    mask = edge_index[0] <= edge_index[1]
    perm = mask.nonzero(as_tuple=False).view(-1)
    perm = perm[torch.randperm(perm.size(0), device=perm.device)]
    num_val = int(val_ratio * perm.numel())
    num_test = int(test_ratio * perm.numel())
    num_train = perm.numel() - num_val - num_test
    train_edges = perm[:num_train]
    val_edges = perm[num_train : num_train + num_val]
    test_edges = perm[num_train + num_val :]
    train_edge_index = edge_index[:, train_edges]
    train_edge_index = torch.cat([train_edge_index, train_edge_index.flip([0])], dim=-1)
    val_edge_index = edge_index[:, val_edges]
    val_edge_index = torch.cat([val_edge_index, val_edge_index.flip([0])], dim=-1)
    test_edge_index = edge_index[:, test_edges]

    return train_edge_index, val_edge_index, test_edge_index


def do_node_inductive_edge_split(
    dataset: Dataset,
    small_dataset=False,
    split_seed=234,
    big_split_ratio_override=None,
    return_split=False,
):
    """Perform an inductive split on the dataset, as described in our paper.
    We perform the following steps:
        1) Withhold a portion of the edges (and same number of disconnected node pairs)
            to use as testing-only edges.
        2) Partition the graph into "observed" and "unobserved" nodes.
        3) Mask out some edges as testing-only edges
        4) Mask out some edges as inference-only edges
        5) Mask out some edges as validation-only edges
        6) Mask out some edge as training-only edges.

    This function returns a tuple with the following items:
        - training_data: training-only data (only observed nodes)
        - val_data: validation-only data (only observed nodes)
        - inference_data: inference-only data (all nodes)
        - data: original data (all nodes)
        - test_edge_bundle: bundle of O-O, O-U, and U-U links where
            O means observed and U means unobserved.
        - negative_samples: negative samples to be used to calculate Hits@K

    If return_split is True, it will return an additional Data object containing only
    the split nodes.

    A more detailed description of the rationale and process can be found
    in Appendix A.4 of our paper. See https://arxiv.org/pdf/2211.14394.pdf
    """
    # Use a larger ratio for smaller datasets to ensure we have enough testing data.
    if small_dataset:
        test_ratio = 0.30
        val_node_ratio = 0.30
        val_ratio = 0.30
        old_old_extra_ratio = 0.1
    elif big_split_ratio_override is None:
        test_ratio = FLAGS.big_split_ratio
        val_node_ratio = FLAGS.big_split_ratio
        val_ratio = FLAGS.big_split_ratio
        old_old_extra_ratio = 0.1
    # Useful for testing, allows us to use this function without having any global flags set
    else:
        test_ratio = big_split_ratio_override
        val_node_ratio = big_split_ratio_override
        val_ratio = big_split_ratio_override
        old_old_extra_ratio = 0.1

    # Seed our RNG
    random.seed(split_seed)
    torch.manual_seed(split_seed)

    # Assume we only have 1 graph in our dataset
    assert len(dataset) == 1
    data = dataset[0]

    # Some assertions to help with type inference
    assert isinstance(data, Data)
    assert data.num_nodes is not None

    # sample some negatives to use globally
    num_negatives = round(test_ratio * data.edge_index.size(1) / 2)
    negative_samples = negative_sampling(
        data.edge_index, data.num_nodes, num_negatives, force_undirected=True
    )

    # Split the nodes into "old" and "new" nodes
    node_splitter = RandomNodeSplit(num_val=0.0, num_test=val_node_ratio)
    new_data = node_splitter(data)

    # Separate the edges between old nodes
    rows, cols = new_data.edge_index
    old_old_edges = create_mask(new_data.train_mask, rows, cols)
    old_old_ei = new_data.edge_index[:, old_old_edges]
    old_old_train, old_old_val, old_old_test = split_edges(
        old_old_ei, old_old_extra_ratio, test_ratio
    )

    # Separate the edges between old and new nodes
    old_new_edges = (new_data.train_mask[rows] & new_data.test_mask[cols]) | (
        new_data.test_mask[rows] & new_data.train_mask[cols]
    )
    old_new_ei = new_data.edge_index[:, old_new_edges]
    old_new_train, _, old_new_test = split_edges(old_new_ei, 0.0, test_ratio)

    # Seperate the edges between new and new nodes
    new_new_edges = create_mask(new_data.test_mask, rows, cols)
    new_new_ei = new_data.edge_index[:, new_new_edges]
    new_new_train, _, new_new_test = split_edges(new_new_ei, 0.0, test_ratio)

    # Create a bundle of all of the different testing sets
    # (old-old, old-new, new-new, all testing)
    test_edge_index = torch.cat([old_old_test, old_new_test, new_new_test], dim=-1)
    test_edge_bundle = (old_old_test, old_new_test, new_new_test, test_edge_index)

    # Use the induced subgraph of only the old-old nodes
    training_only_ei = subgraph(new_data.train_mask, old_old_train, relabel_nodes=True)[
        0
    ]
    training_only_x = new_data.x[new_data.train_mask]

    given_data = Data(training_only_x, training_only_ei)
    # Split the training data into train/val sets
    val_splitter = RandomLinkSplit(0.0, val_ratio, is_undirected=True)
    training_data, _, val_data = val_splitter(given_data)

    # Create the inference-only data.
    inference_edge_index = torch.cat(
        [old_old_train, old_old_val, old_new_train, new_new_train], dim=-1
    )
    inference_data = Data(new_data.x, inference_edge_index)

    log.debug("===== Dataset Information =====")
    log.debug("#Old Nodes:\t" + str(training_only_x.size(0)))
    log.debug("#New Nodes:\t" + str(new_data.x.size(0) - training_only_x.size(0)))
    log.debug("#Old-Old testing edges:\t" + str(old_old_test.size(1)))
    log.debug("#Old-New testing edges:\t" + str(old_new_test.size(1)))
    log.debug("#New-New testing edges:\t" + str(new_new_test.size(1)))

    if return_split:
        return (
            training_data,
            val_data,
            inference_data,
            data,
            test_edge_bundle,
            negative_samples,
            new_data,
        )
    return (
        training_data,
        val_data,
        inference_data,
        data,
        test_edge_bundle,
        negative_samples,
    )


def do_transductive_edge_split(
    dataset, fast_split=False, val_ratio=0.05, test_ratio=0.1, split_seed=234
):
    """Splits a PyG dataset into train/test/valid sets.
    Returns a dictionary in a similar format to OGB's `get_edge_split` function.
    From:
    https://github.com/facebookresearch/SEAL_OGB/blob/ea426e1736009bc7f63de1c260ce3731754d21aa/utils.py#L190
    """
    data = dataset[0]
    random.seed(split_seed)
    torch.manual_seed(split_seed)

    if not fast_split:
        data = train_test_split_edges(data, val_ratio, test_ratio)
        edge_index, _ = add_self_loops(data.train_pos_edge_index)
        data.train_neg_edge_index = negative_sampling(
            edge_index,
            num_nodes=data.num_nodes,
            num_neg_samples=data.train_pos_edge_index.size(1),
        )
    else:
        num_nodes = data.num_nodes
        row, col = data.edge_index
        # Return upper triangular portion.
        mask = row < col
        row, col = row[mask], col[mask]
        n_v = int(math.floor(val_ratio * row.size(0)))
        n_t = int(math.floor(test_ratio * row.size(0)))
        # Positive edges.
        perm = torch.randperm(row.size(0))
        row, col = row[perm], col[perm]
        r, c = row[:n_v], col[:n_v]
        data.val_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v : n_v + n_t], col[n_v : n_v + n_t]
        data.test_pos_edge_index = torch.stack([r, c], dim=0)
        r, c = row[n_v + n_t :], col[n_v + n_t :]
        data.train_pos_edge_index = torch.stack([r, c], dim=0)
        # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
        neg_edge_index = negative_sampling(
            data.edge_index, num_nodes=num_nodes, num_neg_samples=row.size(0)
        )
        data.val_neg_edge_index = neg_edge_index[:, :n_v]
        data.test_neg_edge_index = neg_edge_index[:, n_v : n_v + n_t]
        data.train_neg_edge_index = neg_edge_index[:, n_v + n_t :]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge
