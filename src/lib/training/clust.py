import copy
import os
import time
from kmedoids import fasterpam
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import euclidean_distances
import torch
import logging
from torch.nn import functional as F
from absl import flags
from torch import cosine_similarity
from torch.optim import AdamW
from tqdm import tqdm
import wandb
from torch_geometric.loader import NeighborLoader


from ..classification import fit_logistic_regression, fit_logistic_regression_preset_splits
from ..loss import get_silhouette_score, grouped_hopkins_loss
from .utils import get_time_bundle
from ..scheduler import CosineDecayScheduler
from ..utils import compute_data_representations_only, compute_representations
from ..transforms import compose_transforms
from ..models import BGRL, MlpPredictor

FLAGS = flags.FLAGS
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def perform_clust_gnn_training(
    data,
    dataset,
    output_dir,
    device,
    input_size: int,
    has_features: bool,
    g_zoo,
    train_cb=None,
    extra_return=False,
    wiki_cs_masks=None
):
    representation_size = FLAGS.graph_encoder_layer_dims[-1]

    data = data.to(device)
    if wiki_cs_masks is not None:
        train_masks, val_masks, test_masks = wiki_cs_masks
        num_eval_splits = train_masks.shape[1]

    encoder = g_zoo.get_model(
        FLAGS.graph_encoder_model,
        input_size,
        has_features,
        data.num_nodes,
        n_feats=data.x.size(1),
    ).to(device)
    predictor = MlpPredictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size).to(device)

    # optimizer
    optimizer = AdamW(
        list(encoder.parameters()) + list(predictor.parameters()), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
    )

    # scheduler
    lr_scheduler = CosineDecayScheduler(FLAGS.lr, 25, FLAGS.epochs)

    #####
    # Train & eval functions
    #####
    def full_train(step, state_dict):
        encoder.train()

        # update learning rate
        lr = lr_scheduler.get(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        q = encoder(data)

        if FLAGS.use_predictor:
            q = predictor(q)

        norm_q = F.normalize(q, dim=1)
        if FLAGS.normalize_q:
            in_data = norm_q.cpu()
        else:
            in_data = q.cpu()
        n_nodes = in_data.size(0)

        with torch.no_grad():
            keep_mask = None

            if keep_mask is not None:
                all_data = in_data
                in_data = in_data[keep_mask, :]
                rev_mask_lookup = np.arange(n_nodes)[keep_mask]

            # standard K-Means clustering
            if FLAGS.clustering_method == 'kmeans':
                km = MiniBatchKMeans(n_clusters=FLAGS.num_clusters, n_init=5, init='random')
                cluster_ids_x = torch.tensor(km.fit_predict(in_data.numpy()))
            elif FLAGS.clustering_method == 'online_kmeans':
                km = MiniBatchKMeans(n_clusters=FLAGS.num_clusters, init=state_dict['last_centroids'])
                cluster_ids_x = torch.tensor(km.fit_predict(in_data.numpy()))
                state_dict['last_centroids'] = km.cluster_centers_
            elif FLAGS.clustering_method == 'kmedoids':
                dists = euclidean_distances(in_data.numpy())
                modified_dist = dists
                res = fasterpam(modified_dist, medoids=FLAGS.num_clusters, max_iter=500)
                cluster_ids_x = torch.tensor(res.labels.astype('int32'))

            # train using last medoids
            elif FLAGS.clustering_method == 'online_kmedoids':
                dists = euclidean_distances(in_data.numpy())
                modified_dist = dists
                res = fasterpam(modified_dist, medoids=state_dict['last_medoids'], max_iter=500)

                state_dict['last_medoids'] = res.medoids
                cluster_ids_x = torch.tensor(res.labels.astype('int32'))
            else:
                raise ValueError('Invalid clustering method')

            if keep_mask is not None:
                new_cluster_ids = torch.full((n_nodes,), -1, dtype=cluster_ids_x.dtype)
                new_cluster_ids[keep_mask] = cluster_ids_x
                cluster_ids_x = new_cluster_ids

        n_clust = torch.unique(cluster_ids_x).size(0)
        if FLAGS.clustering_metric == 'hopkins' or n_clust == 1:
            if n_clust == 1:
                print('Falling back to hopkins since there is only one cluster: ', n_clust)
            loss = grouped_hopkins_loss(q, k=FLAGS.num_clusters, n=500, labels=cluster_ids_x, goal=FLAGS.score_target, wandb=wandb)
        elif FLAGS.clustering_metric == 'silhouette':
            loss = get_silhouette_score(norm_q, cluster_ids_x, wandb=wandb, goal=FLAGS.score_target)
        elif FLAGS.clustering_metric == 'hopsil':
            loss = (grouped_hopkins_loss(q, k=FLAGS.num_clusters, n=500, labels=cluster_ids_x, goal=FLAGS.score_target, wandb=wandb) \
                + get_silhouette_score(norm_q, cluster_ids_x, wandb=wandb, goal=FLAGS.score_target)) / 2
        else:
            raise ValueError('Invalid clustering metric')

        loss.backward()

        # update online network
        optimizer.step()

        # log scalars
        log_dict = {
            'train_loss': loss,
            'epoch': step
        }
        if FLAGS.scheduler != 'none':
            log_dict['curr/lr'] = lr

        wandb.log(log_dict, step=wandb.run.step)
        return state_dict

    @torch.no_grad()
    def eval(epoch, repeat = FLAGS.num_eval_splits):
        # make temporary copy of encoder
        tmp_encoder = copy.deepcopy(encoder).eval()
        representations, labels = compute_representations(tmp_encoder, dataset, device)
        repr_normal = F.normalize(representations, dim=1)

        if FLAGS.dataset != 'wiki-cs':
            res = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),
                                             data_random_seed=FLAGS.data_seed, repeat=repeat, return_mistakes=False)
            if len(res) == 3:
                test_scores, val_scores, mistakes = res
            else:
                test_scores, val_scores = res
        else:
            test_scores, val_scores = fit_logistic_regression_preset_splits(representations.cpu().numpy(), labels.cpu().numpy(),
                                                           train_masks, val_masks, test_masks)

        mean_test_scores, std_test_scores = np.mean(test_scores), np.std(test_scores)
        mean_val_scores, std_val_scores = np.mean(val_scores), np.std(val_scores)

        wandb.log({
            'test_acc_mean': mean_test_scores,
            'test_acc_std': std_test_scores,
            'val_acc_mean': mean_val_scores,
            'val_acc_std': std_val_scores
        }, step=wandb.run.step)

        return mean_test_scores, std_test_scores, mean_val_scores, std_val_scores

    #####
    ## End train & eval functions
    #####
    if FLAGS.batch_graphs:
        raise NotImplementedError()
    data = data.to(device)
    best_model_path = os.path.join(output_dir, 'best-encoder.pt')
    best_epoch = None
    times = []

    max_val_score = None
    state_dict = {
        'last_medoids': FLAGS.num_clusters,
        'last_centroids': 'random'
    }

    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        if train_cb is not None:
            train_cb(epoch - 1, encoder)

        # Calculate the amount of time for each epoch
        st_time = time.time_ns()
        state_dict = full_train(epoch-1, state_dict)
        end_time = time.time_ns()
        times.append(end_time - st_time)

        if FLAGS.eval_epochs == 1 or epoch % FLAGS.eval_epochs == 0 or (epoch == 1 and FLAGS.initial_eval):
            mean_test_scores, std_mean_scores, mean_val_scores, std_val_scores = eval(epoch)

            if max_val_score is None or mean_val_scores > max_val_score:
                max_val_score = mean_val_scores
                best_epoch = epoch
                wandb.log({
                    'max_test_acc_mean': mean_test_scores,
                    'max_test_acc_std': std_mean_scores,
                    'max_val_acc_mean': mean_val_scores,
                    'max_val_acc_std': std_val_scores,
                    'max_epoch': epoch
                }, step=wandb.run.step)

                torch.save({ 'model': encoder.state_dict() }, best_model_path)
            # no improvement
            elif FLAGS.early_stop and best_epoch is not None and (epoch - best_epoch >= FLAGS.early_stop_epochs):
                print(f'Performing early stopping since we have not improved for {best_epoch} epochs')
                break

    time_bundle = get_time_bundle(times)

    # save encoder weights
    torch.save(
        {'model': encoder.state_dict()},
        os.path.join(output_dir, f'hopkins-{FLAGS.dataset}.pt'),
    )
    encoder.load_state_dict(torch.load(best_model_path)['model'])

    representations = compute_data_representations_only(
        encoder, data, device, has_features=has_features
    )
    torch.save(
        representations, os.path.join(output_dir, f'bgrl-{FLAGS.dataset}-repr.pt')
    )

    return encoder, representations, time_bundle
