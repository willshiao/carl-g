import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

EPS = 1e-15

def barlow_twins_loss(
    z_a: torch.Tensor,
    z_b: torch.Tensor,
) -> torch.Tensor:
    """Computes the Barlow Twins loss on the two input matrices.
    From the offical GBT implementation at:
    https://github.com/pbielak/graph-barlow-twins/blob/ec62580aa89bf3f0d20c92e7549031deedc105ab/gssl/loss.py
    """

    batch_size = z_a.size(0)
    feature_dim = z_a.size(1)
    _lambda = 1 / feature_dim

    # Apply batch normalization
    z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + EPS)
    z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + EPS)

    # Cross-correlation matrix
    c = (z_a_norm.T @ z_b_norm) / batch_size

    # Loss function
    off_diagonal_mask = ~torch.eye(feature_dim).bool()
    loss = (1 - c.diagonal()).pow(2).sum() + _lambda * c[off_diagonal_mask].pow(2).sum()

    return loss


def cca_ssg_loss(z1, z2, cca_lambda, N):
    """Computes the CCA-SSG loss.
    From the official CCA-SSG implemntation at:
    https://github.com/hengruizhang98/CCA-SSG/blob/cea6e73962c9f2c863d1abfcdf71a2a31de8f983/main.py#L75
    """

    c = torch.mm(z1.T, z2)
    c1 = torch.mm(z1.T, z1)
    c2 = torch.mm(z2.T, z2)

    c = c / N
    c1 = c1 / N
    c2 = c2 / N
    loss_inv = -torch.diagonal(c).sum()
    iden = torch.eye(c.shape[0]).to(z1.device)
    loss_dec1 = (iden - c1).pow(2).sum()
    loss_dec2 = (iden - c2).pow(2).sum()

    return (1 - cca_lambda) * loss_inv + cca_lambda * (loss_dec1 + loss_dec2)

def hopkins_loss(z1, k=500):
    print(z1.shape)
    rand = torch.rand((k, z1.size(1)), device=z1.device)
    rand_norm = F.normalize(rand, dim=1)
    z1_norm = F.normalize(z1, dim=1)
    u_dists = torch.cdist(rand_norm, z1_norm)

    # inefficient (use pdist instead), but we can fix later. we just want a poc
    w_dists = torch.cdist(z1_norm, z1_norm)
    n = z1_norm.size(0)
    w_dists = w_dists.masked_select(~torch.eye(n, dtype=bool, device=z1.device)).view(n, n - 1)
    # w_dists.fill_diagonal_(1e10)
    u = torch.min(u_dists, dim=1)[0]
    w = torch.min(w_dists, dim=1)[0]

    num = u.sum()
    denom = num + w.sum()
    return num / denom

def grouped_hopkins_loss(z, k=5, n=500, labels=None, goal = 1.0, wandb=None):
    if labels is None:
        raise NotImplementedError()

    total_loss = 0.0
    n_samples = 0
    loss_dict = {}
    for i in range(0, k):
        target_idxs = (labels == i)
        selected_z = z[target_idxs]
        if selected_z.size(0) > 1:
            cluster_k_loss = hopkins_loss(selected_z, k=n)
            total_loss += cluster_k_loss
            loss_dict[f'hopkins/clust{i}'] = float(cluster_k_loss)
            loss_dict[f'clust/size_{i}'] = selected_z.size(0)
            n_samples += 1
    avg_dist = total_loss / n_samples
    goal_diff = goal - avg_dist

    loss_dict['hopkins_score'] = avg_dist.item()
    loss_dict['goal_diff'] = goal_diff.item()

    if wandb is not None:
        wandb.log(loss_dict)

    return abs(goal_diff)

def get_silhouette_score(feats, labels, wandb=None, goal=1.0):
    device, dtype = feats.device, feats.dtype
    unique_labels = torch.unique(labels)
    num_samples = len(feats)
    if not (1 < len(unique_labels) < num_samples):
        raise ValueError("num unique labels must be > 1 and < num samples")

    scores = []
    score_dict = {}

    for L in unique_labels:
        if L < 0:
            continue
        curr_cluster = feats[labels == L]
        num_elements = len(curr_cluster)
        if num_elements > 1:
            intra_cluster_dists = torch.cdist(curr_cluster, curr_cluster)
            mean_intra_dists = torch.sum(intra_cluster_dists, dim=1) / (
                num_elements - 1
            )  # minus 1 to exclude self distance
            dists_to_other_clusters = []
            for otherL in unique_labels:
                if otherL != L:
                    other_cluster = feats[labels == otherL]
                    inter_cluster_dists = torch.cdist(curr_cluster, other_cluster)
                    mean_inter_dists = torch.sum(inter_cluster_dists, dim=1) / (
                        len(other_cluster)
                    )
                    dists_to_other_clusters.append(mean_inter_dists)
            dists_to_other_clusters = torch.stack(dists_to_other_clusters, dim=1)
            min_dists, _ = torch.min(dists_to_other_clusters, dim=1)
            curr_scores = (min_dists - mean_intra_dists) / (
                torch.maximum(min_dists, mean_intra_dists)
            )
        else:
            curr_scores = torch.tensor([0], device=device, dtype=dtype)

        with torch.no_grad():
            score_dict[f'silhouette/clust{L}'] = torch.mean(curr_scores).item()
            score_dict[f'clust/size_{L}'] = num_elements
        scores.append(curr_scores)

    scores = torch.cat(scores, dim=0)
    # if len(scores) != num_samples:
    #     raise ValueError(
    #         f"scores (shape {scores.shape}) should have same length as feats (shape {feats.shape})"
    #     )
    mean_score = torch.mean(scores)
    goal_diff = goal - mean_score

    score_dict['silhouette_base_score'] = mean_score.item()
    score_dict['goal_diff'] = goal_diff.item()

    if wandb is not None:
        wandb.log(score_dict)
    return abs(goal_diff)

# returns a negative version of the VRC index
# (suitable for minimization)
def vrc_index(feats: torch.Tensor, labels: torch.Tensor, wandb=None):
    device, dtype = feats.device, feats.dtype
    unique_labels = torch.unique(labels)
    k = (unique_labels >= 0).sum()

    num_samples = len(feats)
    if not (1 < len(unique_labels) < num_samples):
        raise ValueError("num unique labels must be > 1 and < num samples")

    extra_disp, intra_disp = 0.0, 0.0
    mean = torch.mean(feats, dim=0)

    for k in range(k):
        cluster_k = feats[labels == k]
        mean_k = torch.mean(cluster_k, dim=0)
        extra_disp += len(cluster_k) * torch.sum((mean_k - mean) ** 2)
        intra_disp += torch.sum((cluster_k - mean_k) ** 2)

    vrc_index = extra_disp * (num_samples - k) / (intra_disp * (k - 1.0))

    if wandb is not None:
        wandb.log({ 'vrc_index': vrc_index })

    return -vrc_index


def get_fast_silhouette(feats: torch.Tensor, labels: torch.Tensor, wandb=None, goal=1.0):
    device, dtype = feats.device, feats.dtype
    unique_labels = torch.unique(labels)
    k = (unique_labels >= 0).sum()
    unique_label_idxs = torch.arange(k, device=device)
    num_samples = len(feats)
    if not (1 < len(unique_labels) < num_samples):
        raise ValueError("num unique labels must be > 1 and < num samples")

    scores = []
    score_dict = {}
    # note: we have to compute this since we need gradients
    cluster_centroids = []

    for L in unique_labels:
        if L < 0:
            continue
        curr_cluster = feats[labels == L]
        num_elements = len(curr_cluster)
        cluster_centroids.append(curr_cluster.mean(dim=0))
    cluster_centroids = torch.vstack(cluster_centroids)

    for Li in unique_label_idxs:
        L = unique_labels[Li]
        curr_cluster = feats[labels == L]
        dists = torch.cdist(curr_cluster, cluster_centroids)
        a = dists[:, Li]
        selector = torch.ones(k, dtype=torch.bool, device=device)
        selector[Li] = False
        b = torch.min(dists[:, selector], dim=1).values
        sils = (b - a) / torch.max(a, b)
        scores.append(sils)

        # with torch.no_grad():
        #     score_dict[f'silhouette/clust{L}'] = torch.mean(curr_scores).item()
        #     score_dict[f'clust/size_{L}'] = num_elements

    scores = torch.cat(scores, dim=0)
    # if len(scores) != num_samples:
    #     raise ValueError(
    #         f"scores (shape {scores.shape}) should have same length as feats (shape {feats.shape})"
    #     )
    mean_score = torch.mean(scores)
    goal_diff = goal - mean_score

    score_dict['silhouette_base_score'] = mean_score.item()
    score_dict['goal_diff'] = goal_diff.item()

    if wandb is not None:
        wandb.log(score_dict)
    return abs(goal_diff)
