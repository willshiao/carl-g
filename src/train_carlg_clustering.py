import random
import time
import copy
import logging
import os

from absl import app
from absl import flags
import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.cluster import DBSCAN, AgglomerativeClustering, SpectralClustering, MiniBatchKMeans
import numpy as np
import wandb
from fastcluster import linkage
from scipy.cluster.hierarchy import fcluster
from torch_geometric import utils as putils

from lib.utils import compute_representations, set_random_seeds
from lib.data import get_dataset, get_wiki_cs
from lib.classification import fit_logistic_regression, fit_logistic_regression_preset_splits
from lib.models.encoders import GCN
from lib.models.non_contrastive import MlpPredictor
from lib.scheduler import CosineDecayScheduler
from lib.loss import get_fast_silhouette, get_silhouette_score, vrc_index
from lib.clustering import PoincareKMeans
from kmedoids import fasterpam
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.ensemble import IsolationForest


from lib.subspace import ElasticNetSubspaceClustering
from lib.clustering_init import kpp_init
from lib.graph import floyd_warshall, graph_dist_transform
from lib.eval import eval_clustering, run_similarity_search
# from lib.kmm import KMeansMM

log = logging.getLogger(__name__)
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_seed', None, 'Random seed used for model initialization and training.')
flags.DEFINE_integer('data_seed', 1, 'Random seed used to generate train/val/test split.')
flags.DEFINE_integer('num_eval_splits', 3, 'Number of different train/test splits the model will be evaluated over.')

# Dataset.
flags.DEFINE_enum('dataset', 'coauthor-cs',
                  ['amazon-computers', 'amazon-photos', 'coauthor-cs', 'coauthor-physics'],
                  'Which graph dataset to use.')
flags.DEFINE_bool('plot_tsne', False, 'Whether or not to plot TSNE plots at regular intervals')
flags.DEFINE_bool('initial_eval', False, 'Whether or not to evaluate the model at epoch 0')
flags.DEFINE_bool('use_predictor', True, 'Whether or not to use a predictor layer')

flags.DEFINE_bool('online_kmedoids_kpp', False, 'Whether or not to use KMeans++ init.')
flags.DEFINE_enum('scheduler', 'cosine', ['none', 'cosine'], 'Which scheduler to use')
flags.DEFINE_enum('clustering_method', 'online_kmeans', ['kmeans', 'online_kmeans', 'dbscan', 'ward', 'ward_graph', 'single', 'fast_ward', 'centroid', 'spectral', 'spectral_knn', 'hkmeans', 'kmeansmm', 'kmedoids', 'online_kmedoids', 'ensc'], 'Which clustering method to use')
flags.DEFINE_enum('clustering_metric', 'fast_silhouette',
                  ['silhouette', 'fast_silhouette', 'vrc'],
                  'Which clustering metric to optimize for')
flags.DEFINE_enum('clustering_metric_agg_method', 'mean', ['mean', 'weighted_mean', 'median', 'min', 'max'], 'Which clustering aggregation method to use')
flags.DEFINE_bool('early_stop', True, 'Should we perform early stopping?')
flags.DEFINE_integer('early_stop_epochs', 10, 'How many epochs for early stopping.')
flags.DEFINE_integer('kmeans_batch_size', 2048, 'Batch size for k-means.')
flags.DEFINE_integer('k_init', 5, 'Number of times to perform reinitialization for k-means')

flags.DEFINE_integer('dbscan_min_samples', 5, 'Minimum DBSCAN samples,')
flags.DEFINE_float('dbscan_eps', 0.5, 'Max distance between 2 samples to be considered in the same neighborhood.')
flags.DEFINE_enum('distance_metric', 'l2', ['cosine', 'l2', 'graph_l2'], 'Distance metric to use')
flags.DEFINE_float('graph_l2_lambda', 0.8, 'Lambda value used for graph_l2')
flags.DEFINE_bool('normalize_q', True, 'Whether or not to normalize q')
flags.DEFINE_enum('ad_method', 'none', ['none', 'isolation'], 'What method to use for anomaly detection')

flags.DEFINE_string('dataset_dir', './data', 'Where the dataset resides.')

# Architecture.
flags.DEFINE_multi_integer('graph_encoder_layer', [512, 256], 'Conv layer sizes.')
flags.DEFINE_string('graph_encoder_layer_str', '',
                    'Override for graph_encoder_layer in string form')
flags.DEFINE_integer('predictor_hidden_size', 512, 'Hidden size of projector.')

# Training hyperparameters.
flags.DEFINE_integer('epochs', 100, 'The number of training epochs.')
flags.DEFINE_float('lr', 1e-5, 'The learning rate for model training.')
flags.DEFINE_float('weight_decay', 1e-5, 'The value of the weight decay for training.')
flags.DEFINE_float('mm', 0.99, 'The momentum for moving average.')
flags.DEFINE_integer('lr_warmup_epochs', 25, 'Warmup period for learning rate.')

# Augmentations.
flags.DEFINE_integer('num_clusters', 20, 'Number of clusters to use')
flags.DEFINE_float('score_target', 0.8, 'Target clustering score')
# flags.DEFINE_float('drop_edge_p', 0.0, 'Probability of edge dropout 1.')
# flags.DEFINE_float('drop_feat_p', 0.0, 'Probability of node feature dropout 1.')
# flags.register_validator(
#     'drop_edge_p', lambda x: 0 <= x < 1, 'must be between 0 and 1'
# )
# flags.register_validator(
#     'drop_feat_p', lambda x: 0 <= x < 1, 'must be between 0 and 1'
# )
# flags.DEFINE_enum(
#     'graph_transforms',
#     'none',
#     list(VALID_TRANSFORMS.keys()),
#     'Which graph augmentations to use.',
# )

# Logging and checkpoint.
flags.DEFINE_string('logdir', None, 'Where the checkpoint and logs are stored.')
flags.DEFINE_integer('log_steps', 10, 'Log information at every log_steps.')

# Evaluation
flags.DEFINE_integer('eval_epochs', 1, 'Evaluate every eval_epochs.')

def main(argv):
    # use CUDA_VISIBLE_DEVICES to select gpu
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    log.info('Using {} for training.'.format(device))

    # set random seed and store for reproducability
    if FLAGS.model_seed is None:
        FLAGS.model_seed = random.randint(0, 2000000000)

    log.info('Random seed set to {}.'.format(FLAGS.model_seed))
    set_random_seeds(random_seed=FLAGS.model_seed)

    if FLAGS.logdir is None:
        new_logdir = f'./runs/{FLAGS.dataset}'
        log.info(f'No logdir set, using default of {new_logdir}')
        FLAGS.logdir = new_logdir

    if FLAGS.graph_encoder_layer_str:
        log.info(f'Overriding graph layers with {FLAGS.graph_encoder_layer_str}')
        layer_list = [int(x) for x in FLAGS.graph_encoder_layer_str.split(',')]
        FLAGS.graph_encoder_layer = layer_list

    wandb.init(
        project=f'cgnn-clust',
        config={'model_name': 'carlg-gnn', **FLAGS.flag_values_dict()},
    )

    timestr = time.strftime('%Y-%m-%d-%H.%M.%S')
    OUTPUT_DIR = os.path.join(FLAGS.logdir, f'{timestr}_{wandb.run.id}')

    # create log directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, 'config.cfg'), "w") as file:
        file.write(FLAGS.flags_into_string())  # save config file

    # load data
    if FLAGS.dataset != 'wiki-cs':
        dataset = get_dataset(FLAGS.dataset_dir, FLAGS.dataset)
        num_eval_splits = FLAGS.num_eval_splits
    else:
        dataset, train_masks, val_masks, test_masks = get_wiki_cs(FLAGS.dataset_dir)
        num_eval_splits = train_masks.shape[1]

    data = dataset[0]  # all dataset include one graph
    log.info('Dataset {}, {}.'.format(dataset.__class__.__name__, data))
    data = data.to(device)  # permanently move in gpy memory

    # prepare transforms
    # transform_1 = compose_transforms(
    #     FLAGS.graph_transforms,
    #     drop_edge_p=FLAGS.drop_edge_p,
    #     drop_feat_p=FLAGS.drop_feat_p,
    # )

    # build networks
    input_size, representation_size = data.x.size(1), FLAGS.graph_encoder_layer[-1]
    encoder = GCN([input_size] + FLAGS.graph_encoder_layer, batchnorm=True).to(device)  # 512, 256, 128
    predictor = MlpPredictor(representation_size, representation_size, hidden_size=FLAGS.predictor_hidden_size).to(device)

    # optimizer
    optimizer = AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    # scheduler
    if FLAGS.scheduler == 'cosine':
        lr_scheduler = CosineDecayScheduler(FLAGS.lr, FLAGS.lr_warmup_epochs, FLAGS.epochs)

    # setup tensorboard and make custom layout
    writer = SummaryWriter(OUTPUT_DIR)
    layout = {'accuracy': {'accuracy/test': ['Multiline', [f'accuracy/test_{i}' for i in range(num_eval_splits)]]}}
    writer.add_custom_scalars(layout)

    if FLAGS.clustering_method == 'ward_graph':
        G = putils.to_dense_adj(data.edge_index).cpu().numpy().squeeze()

    CACHE_DIR = '../cache'
    if FLAGS.distance_metric == 'graph_l2':
        # TODO(author): consider optimizing this, e.g. with Seidel's algorithm
        apsp_cache_dir = os.path.join(CACHE_DIR, 'apsp')
        os.makedirs(apsp_cache_dir, exist_ok=True)
        target_fn = os.path.join(apsp_cache_dir, f'{FLAGS.dataset}.pt')

        if os.path.exists(target_fn):
            print('Using cached distance matrix')
            D = torch.load(target_fn)
        else:
            ei = data.edge_index.cpu().numpy()
            print('Cached distance matrix not found, computing...')
            D = floyd_warshall(data.num_nodes, ei)
            print('Distance matrix computed, saving...')
            torch.save(D, target_fn)
            print('Distance matrix saved successfully')

        d_trans = graph_dist_transform(D)

    def train(step, state_dict):
        encoder.train()

        # update learning rate
        if FLAGS.scheduler == 'cosine':
            lr = lr_scheduler.get(step)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        # forward
        optimizer.zero_grad()

        # x = transform_1(data)

        # q1, y2 = model(data, data)
        q = encoder(data)
        if FLAGS.use_predictor:
            q = predictor(q)
        # q2, y1 = model(x2, x1)

        # d_model = DBSCAN(min_samples=10, eps=0.3, metric='cosine')
        # cluster_ids_x = d_model.fit_predict(q1.clone().detach().cpu())
        # print(np.unique(cluster_ids_x))
        # k = np.max(cluster_ids_x) + 1

        norm_q = F.normalize(q, dim=1)
        if FLAGS.normalize_q:
            in_data = norm_q.cpu()
        else:
            in_data = q.cpu()
        n_nodes = in_data.size(0)

        # we don't need (or want) gradients from clustering
        # even if the clustering method supports it
        with torch.no_grad():
            keep_mask = None

            if FLAGS.ad_method == 'isolation':
                forest = IsolationForest()
                scores = forest.fit_predict(X=norm_q.cpu())
                keep_mask = torch.tensor(scores == 1)
                n_anom = (~keep_mask).sum()
                wandb.log({ 'n_anomalies': n_anom }, step=wandb.run.step)

            if keep_mask is not None:
                all_data = in_data
                in_data = in_data[keep_mask, :]
                rev_mask_lookup = np.arange(n_nodes)[keep_mask]

            # standard K-Means clustering
            if FLAGS.clustering_method == 'kmeans':
                km = MiniBatchKMeans(n_clusters=FLAGS.num_clusters, n_init=FLAGS.k_init, init='random')
                cluster_ids_x = torch.tensor(km.fit_predict(in_data.numpy()))

            elif FLAGS.clustering_method == 'online_kmeans':
                if isinstance(state_dict['last_centroids'], int):
                    n_init = FLAGS.k_init
                else:
                    n_init = 1
                km = MiniBatchKMeans(n_clusters=FLAGS.num_clusters,
                                        init=state_dict['last_centroids'],
                                        n_init=n_init,
                                        batch_size=FLAGS.kmeans_batch_size)
                cluster_ids_x = torch.tensor(km.fit_predict(in_data.numpy()))
                state_dict['last_centroids'] = km.cluster_centers_

            elif FLAGS.clustering_method == 'kmeansmm':
                raise NotImplementedError()
                # kmm = KMeansMM(num_clusters=FLAGS.num_clusters)
                # cluster_ids_x = kmm.fit_predict(norm_q.cpu(), device=device, tol=1e-5)
            elif FLAGS.clustering_method == 'ensc':
                ensc = ElasticNetSubspaceClustering(n_clusters=FLAGS.num_clusters, algorithm='lasso_lars', gamma=50)
                ensc.fit(norm_q.cpu())
                cluster_ids_x = torch.tensor(ensc.labels_)
            elif FLAGS.clustering_method == 'dbscan':
                d_model = DBSCAN(min_samples=FLAGS.dbscan_min_samples, eps=FLAGS.dbscan_eps, metric='cosine')
                cluster_ids_x = torch.tensor(d_model.fit_predict(in_data))

            elif FLAGS.clustering_method == 'kmedoids':
                dists = euclidean_distances(in_data.numpy())
                if FLAGS.distance_metric == 'graph_l2':
                    lam = FLAGS.graph_l2_lambda
                    if keep_mask is not None:
                        modified_dist = lam * dists + (1-lam) * d_trans[keep_mask, keep_mask]
                    else:
                        modified_dist = lam * dists + (1-lam) * d_trans
                else:
                    modified_dist = dists

                res = fasterpam(modified_dist, medoids=FLAGS.num_clusters, max_iter=500)
                cluster_ids_x = torch.tensor(res.labels.astype('int32'))

            # train using last medoids
            elif FLAGS.clustering_method == 'online_kmedoids':
                dists = euclidean_distances(in_data.numpy())

                if keep_mask is not None:
                    if not keep_mask[state_dict['last_medoids']].all():
                        print('WARNING: medoid was classified as anomaly')
                        wandb.alert(
                            title='Medoid Classified as Anomaly',
                            text=f'A medoid was classified as an anomaly:\n{state_dict["last_medoids"]}'
                        )

                if FLAGS.distance_metric == 'graph_l2':
                    lam = FLAGS.graph_l2_lambda
                    if keep_mask is not None:
                        modified_dist = lam * dists + (1-lam) * d_trans[keep_mask, keep_mask]
                    else:
                        modified_dist = lam * dists + (1-lam) * d_trans
                else:
                    modified_dist = dists
                res = fasterpam(modified_dist, medoids=state_dict['last_medoids'], max_iter=500)

                if keep_mask is not None:
                    state_dict['last_medoids'] = rev_mask_lookup[res.medoids]
                else:
                    state_dict['last_medoids'] = res.medoids
                cluster_ids_x = torch.tensor(res.labels.astype('int32'))

            elif FLAGS.clustering_method == 'ward':
                c_model = AgglomerativeClustering(n_clusters=FLAGS.num_clusters)
                cluster_ids_x = torch.tensor(c_model.fit_predict(norm_q.cpu()))
            elif FLAGS.clustering_method == 'ward_graph':
                c_model = AgglomerativeClustering(n_clusters=FLAGS.num_clusters, connectivity=G)
                cluster_ids_x = torch.tensor(c_model.fit_predict(norm_q.cpu()))
            elif FLAGS.clustering_method == 'spectral':
                c_model = SpectralClustering(n_clusters=FLAGS.num_clusters)
                cluster_ids_x = torch.tensor(c_model.fit_predict(norm_q.cpu()))
            elif FLAGS.clustering_method == 'spectral_knn':
                c_model = SpectralClustering(n_clusters=FLAGS.num_clusters, affinity='nearest_neighbors', n_neighbors=20)
                cluster_ids_x = torch.tensor(c_model.fit_predict(norm_q.cpu()))
            elif FLAGS.clustering_method == 'hkmeans':
                hkmeans = PoincareKMeans(n_clusters=FLAGS.num_clusters)
                cluster_ids_x = torch.tensor(hkmeans.fit_predict(q.cpu().numpy()))
            elif FLAGS.clustering_method == 'fast_ward':
                h_clust = linkage(norm_q.cpu(), method='ward', metric='euclidean')
                cluster_ids_x = torch.tensor(fcluster(h_clust, t=FLAGS.num_clusters, criterion='maxclust'))
            elif FLAGS.clustering_method == 'centroid':
                h_clust = linkage(norm_q.cpu(), method='centroid', metric='euclidean')
                cluster_ids_x = torch.tensor(fcluster(h_clust, t=FLAGS.num_clusters, criterion='maxclust'))
            elif FLAGS.clustering_method == 'single':
                h_clust = linkage(norm_q.cpu(), method='single', metric='euclidean')
                cluster_ids_x = torch.tensor(fcluster(h_clust, t=FLAGS.num_clusters, criterion='maxclust'))
            else:
                raise ValueError('Invalid clustering method')

            if keep_mask is not None:
                new_cluster_ids = torch.full((n_nodes,), -1, dtype=cluster_ids_x.dtype)
                new_cluster_ids[keep_mask] = cluster_ids_x
                cluster_ids_x = new_cluster_ids

        # loss = 2 - cosine_similarity(q1, y2.detach(), dim=-1).mean() - cosine_similarity(q2, y1.detach(), dim=-1).mean()
        if FLAGS.clustering_metric == 'silhouette':
            loss = get_silhouette_score(norm_q, cluster_ids_x, wandb=wandb, goal=FLAGS.score_target)
        elif FLAGS.clustering_metric == 'fast_silhouette':
            loss = get_fast_silhouette(norm_q, cluster_ids_x, wandb=wandb, goal=FLAGS.score_target)
        elif FLAGS.clustering_metric == 'vrc':
            loss = vrc_index(norm_q, cluster_ids_x, wandb=wandb)
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
            writer.add_scalar('params/lr', lr, step)

        wandb.log(log_dict, step=wandb.run.step)
        writer.add_scalar('train/loss', loss, step)
        return state_dict, cluster_ids_x

    @torch.no_grad()
    def eval(epoch, cluster_ids_x, repeat = FLAGS.num_eval_splits):
        # make temporary copy of encoder
        tmp_encoder = copy.deepcopy(encoder).eval()
        representations, labels = compute_representations(tmp_encoder, dataset, device)

        metric_dict = eval_clustering(representations, labels)
        nmi_info, homo_info = metric_dict['nmi'], metric_dict['homo']
        nmi_mean, nmi_std = nmi_info
        homo_mean, homo_std = homo_info

        comp_metric_dict = eval_clustering(representations, cluster_ids_x)
        comp_nmi_info, comp_homo_info = comp_metric_dict['nmi'], comp_metric_dict['homo']
        comp_nmi_mean, comp_nmi_std = comp_nmi_info
        comp_homo_mean, comp_homo_std = comp_homo_info

        sim_search_res = run_similarity_search(representations, labels)

        # if FLAGS.plot_tsne:
        #     tsne = TSNE(metric="euclidean", n_jobs=8, verbose=True)
        #     embedding_train = tsne.fit(repr_normal.cpu().numpy())
        #     plot_tsne(embedding_train, data.y.cpu().numpy(), mistakes=mistakes)
        #     plt.savefig(os.path.join(OUTPUT_DIR, f'tSNE_epoch_{epoch}.png'))

        wandb.log({
            'nmi_mean': nmi_mean,
            'nmi_std': nmi_std,
            'homo_mean': homo_mean,
            'homo_std': homo_std,
            'comp_nmi_mean': comp_nmi_mean,
            'comp_nmi_std': comp_nmi_std,
            'comp_homo_mean': comp_homo_mean,
            'comp_homo_std': comp_homo_std,
            **sim_search_res
        }, step=wandb.run.step)

        if FLAGS.dataset != 'wiki-cs':
            res = fit_logistic_regression(representations.cpu().numpy(), labels.cpu().numpy(),
                                             data_random_seed=FLAGS.data_seed, repeat=repeat, return_mistakes=FLAGS.plot_tsne)
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

        return (
            mean_test_scores, std_test_scores, mean_val_scores, std_val_scores,
            nmi_mean, nmi_std, homo_mean, homo_std, sim_search_res['sim@5'], sim_search_res['sim@10']
        )

    max_val_score = None
    state_dict = {
        'last_medoids': FLAGS.num_clusters,
        'last_centroids': 'random'
    }

    if FLAGS.online_kmedoids_kpp:
        with torch.no_grad():
            encoder.eval()
            # x = transform_1(data)
            q = encoder(x)
            q = predictor(q)

            norm_q = F.normalize(q, dim=1)
            if FLAGS.normalize_q:
                in_data = norm_q.cpu()
            else:
                in_data = q.cpu()
        state_dict['last_medoids'] = kpp_init(in_data.numpy(), FLAGS.num_clusters)

    best_model_path = os.path.join(OUTPUT_DIR, 'best-encoder.pt')
    best_epoch = None
    times = []

    gmax_nmi = None
    gmax_homo = None
    gmax_sim5 = None
    gmax_sim10 = None

    for epoch in tqdm(range(1, FLAGS.epochs + 1)):
        # Calculate the amount of time for each epoch
        st_time = time.time_ns()
        state_dict, cluster_ids_x = train(epoch-1, state_dict)
        end_time = time.time_ns()
        times.append(end_time - st_time)

        if FLAGS.eval_epochs == 1 or epoch % FLAGS.eval_epochs == 0 or (epoch == 1 and FLAGS.initial_eval):
            (mean_test_scores, std_test_scores, mean_val_scores, std_val_scores,
            nmi_mean, nmi_std, homo_mean, homo_std, sim_5, sim_10) = eval(epoch, cluster_ids_x)

            if gmax_nmi is None or gmax_nmi < nmi_mean:
                gmax_nmi = nmi_mean
                wandb.log({ 'gmax_nmi': gmax_nmi }, step=wandb.run.step)
            if gmax_homo is None or gmax_homo < homo_mean:
                gmax_homo = homo_mean
                wandb.log({ 'gmax_homo': gmax_homo }, step=wandb.run.step)
            if gmax_sim5 is None or gmax_sim5 < sim_5:
                gmax_sim5 = sim_5
                wandb.log({ 'gmax_sim5': gmax_sim5 }, step=wandb.run.step)
            if gmax_sim10 is None or gmax_sim10 < sim_10:
                gmax_sim10 = sim_10
                wandb.log({ 'gmax_sim10': gmax_sim10 }, step=wandb.run.step)

            if max_val_score is None or mean_val_scores > max_val_score:
                max_val_score = mean_val_scores
                best_epoch = epoch
                wandb.log({
                    'max_nmi_mean': nmi_mean,
                    'max_nmi_std': nmi_std,
                    'max_homo_mean': homo_mean,
                    'max_homo_std': homo_std,
                    'max_sim@5_mean': sim_5,
                    'max_sim@10_mean': sim_10,
                    'max_test_acc_mean': mean_test_scores,
                    'max_test_acc_std': std_test_scores,
                    'max_val_acc_mean': mean_val_scores,
                    'max_val_acc_std': std_val_scores,
                    'max_epoch': epoch
                }, step=wandb.run.step)

                torch.save({ 'model': encoder.state_dict() }, best_model_path)
            # no improvement
            elif FLAGS.early_stop and best_epoch is not None and (epoch - best_epoch >= FLAGS.early_stop_epochs):
                print(f'Performing early stopping since we have not improved for {best_epoch} epochs')
                break

    # save encoder weights
    torch.save({'model': encoder.state_dict()}, os.path.join(OUTPUT_DIR, 'carlg-gnn.pt'))

    # we should have run at least 1 epoch
    assert(best_epoch is not None)
    wandb.log({
        'epoch_time_mean': np.mean(times) / 1e9,
        'epoch_time_std': np.std(times) / 1e9,
        'total_time': np.sum(times) / 1e9,
        'total_end_time': np.sum(times[:best_epoch-1]) / 1e9,
    })

    # load & evaluate the best model
    (mean_test_scores, std_test_scores, mean_val_scores, std_val_scores,
            nmi_mean, nmi_std, homo_mean, homo_std, sim_5, sim_10) = eval(FLAGS.epochs)

    wandb.log({
        'best_test_acc_mean': mean_test_scores,
        'best_test_acc_std': std_test_scores,
        'best_val_acc_mean': mean_val_scores,
        'best_val_acc_std': std_val_scores,
        'best_nmi_mean': nmi_mean,
        'best_nmi_std': nmi_std,
        'best_homo_mean': homo_mean,
        'best_homo_std': homo_std,
        'best_sim@5': sim_5,
        'best_sim@10': sim_10
    })


if __name__ == "__main__":
    log.info('PyTorch version: %s' % torch.__version__)
    app.run(main)
