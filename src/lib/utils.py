from collections import Counter, defaultdict
import logging
from typing import Optional, Tuple, Union
import torch
from torch_geometric.utils import to_networkx
from torch.nn.functional import one_hot
import matplotlib
from absl import flags
import numpy as np
import pandas as pd
import random

from .models import GraceEncoder

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
SMALL_DATASETS = set(['cora', 'citeseer'])
# Used for formatting output
SHORT_DIVIDER = '=' * 10
LONG_DIVIDER_STR = '=' * 30

def load_trained_encoder(encoder, ckpt_path, device):
    r"""Utility for loading the trained encoder."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(checkpoint['model'], strict=True)
    return encoder.to(device)

def set_random_seeds(random_seed=0):
    r"""Sets the seed for generating random numbers."""
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


def to_torch_coo_tensor(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    size: Optional[Union[int, Tuple[int, int]]] = None,
) -> torch.Tensor:
    """Converts a sparse adjacency matrix defined by edge indices and edge
    attributes to a :class:`torch.sparse.Tensor`.
    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): The edge attributes.
            (default: :obj:`None`)
        size (int or (int, int), optional): The size of the sparse matrix.
            If given as an integer, will create a quadratic sparse matrix.
            If set to :obj:`None`, will infer a quadratic sparse matrix based
            on :obj:`edge_index.max() + 1`. (default: :obj:`None`)
    :rtype: :class:`torch.sparse.FloatTensor`
    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],
        ...                            [1, 0, 2, 1, 3, 2]])
        >>> to_torch_coo_tensor(edge_index)
        tensor(indices=tensor([[0, 1, 1, 2, 2, 3],
                            [1, 0, 2, 1, 3, 2]]),
            values=tensor([1., 1., 1., 1., 1., 1.]),
            size=(4, 4), nnz=6, layout=torch.sparse_coo)
    """
    if size is None:
        size = int(edge_index.max()) + 1
    if not isinstance(size, (tuple, list)):
        size = (size, size)

    if edge_attr is None:
        edge_attr = torch.ones(edge_index.size(1), device=edge_index.device)

    size = tuple(size) + edge_attr.size()[1:]
    out = torch.sparse_coo_tensor(edge_index, edge_attr, size,
                                  device=edge_index.device)
    out = out.coalesce()
    return out


def print_run_num(run_num):
    log.info(LONG_DIVIDER_STR)
    log.info(LONG_DIVIDER_STR)
    log.info(SHORT_DIVIDER + f'  Run #{run_num}  ' + SHORT_DIVIDER)
    log.info(LONG_DIVIDER_STR)
    log.info(LONG_DIVIDER_STR)


def add_node_feats(data, device, type='degree'):
    assert type == 'degree'

    G = to_networkx(data)
    degrees = torch.tensor([v for (_, v) in G.degree()])  # type: ignore
    data.x = one_hot(degrees).to(device).float()
    return data


def keywise_agg(dicts):
    df = pd.DataFrame(dicts)
    mean_dict = df.mean().to_dict()
    std_dict = df.std().to_dict()
    return mean_dict, std_dict


def keywise_prepend(d, prefix):
    out = {}
    for k, v in d.items():
        out[prefix + k] = v
    return out


def is_small_dset(dset):
    return dset in SMALL_DATASETS


def merge_multirun_results(all_results):
    """Merges results from multiple runs into a single dictionary."""
    runs = zip(*all_results)
    agg_results = []
    val_mean = test_mean = None

    for run_group in runs:
        group_type = run_group[0]['type']
        val_res = [run['val'] for run in run_group]
        test_res = [run['test'] for run in run_group]

        val_mean, val_std = keywise_agg(val_res)
        test_mean, test_std = keywise_agg(test_res)
        agg_results.append(
            {
                'type': group_type,
                'val_mean': val_mean,
                'val_std': val_std,
                'test_mean': test_mean,
                'test_std': test_std,
            }
        )

    assert val_mean is not None
    assert test_mean is not None
    return agg_results, {
        **keywise_prepend(val_mean, 'val_mean_'),
        **keywise_prepend(test_mean, 'test_mean_'),
    }

def compute_representations(net, dataset, device):
    r"""Pre-computes the representations for the entire dataset.

    Returns:
        [torch.Tensor, torch.Tensor]: Representations and labels.
    """
    net.eval()
    reps = []
    labels = []

    for data in dataset:
        # forward
        data = data.to(device)
        with torch.no_grad():
            reps.append(net(data))
            labels.append(data.y)

    reps = torch.cat(reps, dim=0)
    labels = torch.cat(labels, dim=0)
    return [reps, labels]


def compute_representations_only(
    net, dataset, device, has_features=True, feature_type='degree'
):
    """Pre-computes the representations for the entire dataset.
    Does not include node labels.

    Returns:
        torch.Tensor: Representations
    """
    net.eval()
    reps = []

    for data in dataset:
        # forward
        data = data.to(device)
        if not has_features:
            if data.x is not None:
                log.warn('[WARNING] node features overidden in Data object')
            data.x = net.get_node_feats().weight.data
        elif data.x is None:
            data = add_node_feats(data, device=device, type=feature_type)

        with torch.no_grad():
            if isinstance(net, GraceEncoder):
                reps.append(net(data.x, data.edge_index))
            else:
                reps.append(net(data))

    reps = torch.cat(reps, dim=0)
    return reps


def compute_data_representations_only(net, data, device, has_features=True):
    r"""Pre-computes the representations for the entire dataset.
    Does not include node labels.

    Returns:
        torch.Tensor: Representations
    """
    net.eval()
    reps = []

    if not has_features:
        if data.x is not None:
            log.warn('[WARNING] features overidden in adj matrix')
        data.x = net.get_node_feats().weight.data

    with torch.no_grad():
        reps.append(net(data))

    reps = torch.cat(reps, dim=0).to(device)
    return reps

def plot_tsne(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    mistakes = None,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs
):

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(16, 16))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.6), "s": kwargs.get("s", 16)}
    error_params = {"alpha": kwargs.get("alpha", 0.9), "s": kwargs.get("s", 72)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))
    npc = np.array(point_colors)

    if mistakes is None:
        ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=False, **plot_params)
    else:
        mistakes = np.concatenate(mistakes)
        mistake_count = Counter(mistakes)
        inverse_count = defaultdict(list)
        markers = ['v', '+', 'x']

        for k, v in mistake_count.items():
            inverse_count[v].append(k)
        unique_mistakes = np.unique(mistakes)
        idxs = np.arange(y.shape[0])
        correct_mask = ~np.isin(idxs, unique_mistakes, assume_unique=True)
        # plot 0 mistakes first
        ax.scatter(x[correct_mask, 0], x[correct_mask, 1], c=npc[correct_mask], rasterized=False, **plot_params)
        for m, k in zip(markers, inverse_count.keys()):
            mistake_idxs = np.array(inverse_count[k])
            mistake_mask = np.isin(idxs, mistake_idxs, assume_unique=True)
            ax.scatter(x[mistake_mask, 0], x[mistake_mask, 1], c=npc[mistake_mask], rasterized=False, marker=m, **error_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=10,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)