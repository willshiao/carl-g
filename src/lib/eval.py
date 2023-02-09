from typing import Dict
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import torch
import json
from os import path
from absl import flags
from sklearn.metrics import homogeneity_score, normalized_mutual_info_score, roc_auc_score
from torch import nn
from torch_geometric.utils import negative_sampling
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import logging
from sklearn.metrics import pairwise
from tqdm import tqdm
import wandb

from .models.decoders import DecoderZoo, GBTLogisticRegression

from .classification import do_classification_eval
from .utils import compute_representations_only

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
FLAGS = flags.FLAGS

# Based off of: https://github.com/Namkyeong/AFGRL/blob/master/evaluate.py
def eval_clustering(embeddings, labels):
    embeddings = F.normalize(embeddings, dim = -1, p = 2).detach().cpu().numpy()
    nb_class = len(labels.unique())
    true_y = labels.detach().cpu().numpy()

    estimator = KMeans(n_clusters = nb_class)

    NMI_list = []
    h_list = []

    for _ in range(10):
        estimator.fit(embeddings)
        y_pred = estimator.predict(embeddings)
        
        h_score = homogeneity_score(true_y, y_pred)
        s1 = normalized_mutual_info_score(true_y, y_pred, average_method='arithmetic')
        NMI_list.append(s1)
        h_list.append(h_score)

    return {
        'nmi': (np.mean(NMI_list), np.std(NMI_list)),
        'homo': (np.mean(h_list), np.std(h_list))
    }
    # print('Evaluate clustering results')
    # print('** Clustering NMI: {:.4f} | homogeneity score: {:.4f} **'.format(s1, h_score))

# Based off of: https://github.com/Namkyeong/AFGRL/blob/master/evaluate.py
def run_similarity_search(embeddings, labels):
    test_embs = embeddings.detach().cpu().numpy()
    test_lbls = labels.detach().cpu().numpy()
    numRows = test_embs.shape[0]

    cos_sim_array = pairwise.cosine_similarity(test_embs) - np.eye(numRows)
    target_k = [5, 10]
    st = []
    for N in target_k:
        indices = np.argsort(cos_sim_array, axis=1)[:, -N:]
        tmp = np.tile(test_lbls, (numRows, 1))
        selected_label = tmp[np.repeat(np.arange(numRows), N), indices.ravel()].reshape(numRows, N)
        original_label = np.repeat(test_lbls, N).reshape(numRows,N)
        st.append(np.round(np.mean(np.sum((selected_label == original_label), 1) / N),4))

    return { f'sim@{target_k[i]}': st[i] for i in range(len(target_k)) }

    print('Evaluate similarity search results')
    print("** sim@5 : {} | sim@10 : {} **".format(st[0], st[1]))

def eval_hits(y_pred_pos, y_pred_neg, K):
    '''
    compute Hits@K
    For each positive target node, the negative target nodes are the same.
    y_pred_neg is an array.
    rank y_pred_pos[i] against y_pred_neg for each i
    From:
    https://github.com/snap-stanford/ogb/blob/1c875697fdb20ab452b2c11cf8bfa2c0e88b5ad3/ogb/linkproppred/evaluate.py#L214
    '''

    if len(y_pred_neg) < K:
        log.warn(f'[WARNING]: hits@{K} defaulted to 1')
        return {'hits@{}'.format(K): 1.0}

    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(
        y_pred_pos
    )
    return {'hits@{}'.format(K): hitsK}


def eval_roc(y_pred_pos, y_pred_neg):
    """Computes the ROC-AUC.
    From: https://github.com/snap-stanford/ogb/blob/1c875697fdb20ab452b2c11cf8bfa2c0e88b5ad3/ogb/linkproppred/evaluate.py#L280
    """
    ground_truth = torch.concat(
        (torch.ones(y_pred_pos.size(0)), torch.zeros(y_pred_neg.size(0)))
    )
    preds = torch.concat((y_pred_pos, y_pred_neg))
    return {'roc': roc_auc_score(ground_truth.cpu(), preds.cpu())}


def eval_all(y_pred_pos, y_pred_neg):
    """Computes the following evaluation metrics:
        - Hits@3
        - Hits@10
        - Hits@20
        - Hits@30
        - Hits@50
        - Hits@100
        - ROC-AUC
    and returns a dictionary with all of the metrics.
    """

    return {
        **eval_hits(y_pred_pos, y_pred_neg, 3),
        **eval_hits(y_pred_pos, y_pred_neg, 10),
        **eval_hits(y_pred_pos, y_pred_neg, 20),
        **eval_hits(y_pred_pos, y_pred_neg, 30),
        **eval_hits(y_pred_pos, y_pred_neg, 50),
        **eval_hits(y_pred_pos, y_pred_neg, 100),
        **eval_roc(y_pred_pos, y_pred_neg),
    }

def big_linear_eval(num_classes, train_data, val_data, test_data, device):
    r"""
    Trains a linear layer on top of the representations. This function is specific to the PPI dataset,
    which has multiple labels.
    
    From: https://github.com/nerdslab/bgrl/blob/dec99f8c605e3c4ae2ece57f3fa1d41f350d11a9/bgrl/linear_eval_ppi.py#L6
    """
    # one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    # train_y = torch.tensor(one_hot_encoder.fit_transform(train_data[1].view(-1, 1)), dtype=torch.bool)
    # train_y = F.one_hot(train_data[1].ravel()).type(train_data[0].type())
    # valid_y = F.one_hot(val_data[1].ravel()).type(val_data[0].type())
    # test_y = F.one_hot(test_data[1].ravel()).type(test_data[0].type())
    # detach any existing grad on representations, but allow gradients to update
    # train_x = train_data[0].detach()
    # train_x.requires_grad = True

    def train(classifier, train_x, train_y, optimizer):
        classifier.train()

        x, label = train_x, train_y
        x, label = x.to(device), label.to(device).squeeze()
        for step in range(1000):
            # forward
            optimizer.zero_grad()
            pred_logits = classifier(x)

            # loss and backprop
            loss = criterion(pred_logits, label)
            wandb.log({ 'linear_loss': loss })
            loss.backward()
            optimizer.step()

    def test(classifier, eval_x, eval_y):
        classifier.eval()
        x, label = eval_x, eval_y
        label = label.cpu().numpy().squeeze()

        # feed to network and classifier
        with torch.no_grad():
            pred_logits = classifier(x.to(device))
            y_pred_val = torch.argmax(pred_logits, dim=1).squeeze()
            # pred_class = (pred_logits > 0).float().cpu().numpy()

        return metrics.accuracy_score(label, y_pred_val.cpu().numpy())
        # return metrics.f1_score(label, pred_class, average='micro') if pred_class.sum() > 0 else 0

    num_feats = train_data[0].size(1)
    criterion = torch.nn.CrossEntropyLoss()

    # normalization
    mean, std = train_data[0].mean(0, keepdim=True), train_data[0].std(0, unbiased=False, keepdim=True)
    train_data[0] = (train_data[0] - mean) / std
    val_data[0] = (val_data[0] - mean) / std
    test_data[0] = (test_data[0] - mean) / std

    best_val_f1 = 0
    test_f1 = 0
    for weight_decay in 2.0 ** np.arange(-10, 11, 2):
        classifier = torch.nn.Linear(num_feats, num_classes).to(device)
        for m in classifier.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

        optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=0.0001, weight_decay=weight_decay)

        train(classifier, train_data[0], train_data[1], optimizer)
        val_f1 = test(classifier, val_data[0], val_data[1])
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            test_f1 = test(classifier, test_data[0], test_data[1])

    return best_val_f1, test_f1


def do_inductive_eval(
    model_name,
    output_dir,
    encoder,
    valid_models,
    train_data,
    val_data,
    inference_data,
    lp_zoo,
    device,
    test_edge_bundle,
    negative_samples,
    wb=None,
    output_fn='results.json',
    return_extra=False,
):
    """Trains a link prediction model on the provided embeddings in the inductive setting.
    Returns the trained link predictor and various evaluation metrics.
    """
    all_results = []

    train_embeddings = torch.nn.Embedding.from_pretrained(
        compute_representations_only(encoder, [train_data], device), freeze=True
    )
    inference_embeddings = torch.nn.Embedding.from_pretrained(
        compute_representations_only(encoder, [inference_data], device), freeze=True
    )

    log.info('Beginning evaluation...')

    if FLAGS.do_classification_eval:
        raise NotImplementedError()

    nn_model = None
    for i in range(len(valid_models)):
        log.info('Performing NN-based eval')
        nn_model, results = perform_inductive_nn_link_eval(
            lp_zoo=lp_zoo,
            train_data=train_data,
            val_data=val_data,
            train_embeddings=train_embeddings,
            inference_embeddings=inference_embeddings,
            device=device,
            test_edge_bundle=test_edge_bundle,
            negative_samples=negative_samples,
            wb=wb,
            model_idx=i,
        )
        all_results.append(results)

    results_path = path.join(output_dir, output_fn)
    if path.exists(results_path):
        log.info('Existing file found, appending results')
        with open(results_path, 'rb') as f:
            contents = json.load(f)
        log.debug(f'Existing contents: {contents}')

        contents['results'].extend(all_results)

        mn = model_name
        if contents['model_name'] != mn:
            log.warn(
                f'[WARNING]: Model names do not match - {contents["model_name"]} vs {mn}'
            )

        with open(results_path, 'w') as f:
            json.dump(
                {
                    'model_name': mn,
                    'results': contents['results'],
                },
                f,
                indent=4,
            )
        log.info(f'Appended results to {results_path}')
    else:
        log.info('No results file found, writing to new one')
        with open(results_path, 'w') as f:
            json.dump(
                {
                    'model_name': model_name,
                    'results': all_results,
                },
                f,
                indent=4,
            )
        log.info(f'Wrote results to file at {results_path}')

    if return_extra:
        return nn_model, all_results
    return all_results


def do_all_eval(
    model_name,
    output_dir,
    valid_models,
    dataset,
    edge_split,
    embeddings,
    lp_zoo,
    wb,
    patience=50,
    output_fn='results.json',
):
    """Train a link prediction model in the transductive setting.
    If `FLAGS.do_classification_eval` is true, it will also perform node classification.
    """
    all_results = []
    classification_results = []

    log.info('Beginning evaluation...')

    if FLAGS.do_classification_eval:
        log.info('Performing classification performance evaluation')
        class_acc = do_classification_eval(dataset, embeddings)
        classification_results.append(class_acc)
        wb.log({'class_acc': class_acc})

    for i in range(len(valid_models)):
        log.info('Performing NN-based eval')
        _, results = perform_nn_link_eval(
            lp_zoo, dataset, edge_split, wb, embeddings, model_idx=i, patience=patience
        )
        all_results.append(results)

    results_path = path.join(output_dir, output_fn)
    if path.exists(results_path):
        log.info('Existing file found, appending results')
        with open(results_path, 'rb') as f:
            contents = json.load(f)
        log.debug(f'Existing contents: {contents}')

        # Patch to solve error introduced in earlier versions
        while isinstance(contents['results'], dict):
            contents['results'] = contents['results']['results']
        contents['results'].extend(all_results)

        if 'class_results' not in contents:
            contents['class_result'] = classification_results
        else:
            contents['class_results'].extend(classification_results)

        mn = model_name
        if contents['model_name'] != mn:
            log.warn(
                f'[WARNING]: Model names do not match - {contents["model_name"]} vs {mn}'
            )

        with open(results_path, 'w') as f:
            json.dump(
                {
                    'model_name': mn,
                    'results': contents['results'],
                    'class_results': classification_results,
                },
                f,
                indent=4,
            )
        log.info(f'Appended results to {results_path}')
    else:
        log.info('No results file found, writing to new one')
        with open(results_path, 'w') as f:
            json.dump(
                {
                    'model_name': model_name,
                    'results': all_results,
                    'class_results': classification_results,
                },
                f,
                indent=4,
            )
        log.info(f'Wrote results to file at {results_path}')
    return all_results, classification_results

def gbt_train_pytorch_model(
    emb_dim: int,
    num_cls: int,
    X: np.ndarray,
    y: np.ndarray,
    train_idx,
    val_idx,
    test_idx
) -> "GBTLogisticRegression":
    # Define parameter space
    wd = 2.0 ** np.arange(-10, 10, 2)

    best_clf = None
    best_acc = -1

    pbar = tqdm(wd, desc="Train best classifier")
    for weight_decay in pbar:
        lr_model = GBTLogisticRegression(
            in_dim=emb_dim,
            out_dim=num_cls,
            weight_decay=weight_decay,
            is_multilabel=False,
        )

        lr_model.fit(X[train_idx], y[train_idx])

        acc = metrics.classification_report(
            y_true=y[val_idx],
            y_pred=lr_model.predict(X[val_idx]),
            output_dict=True,
            zero_division=0,
        )["accuracy"]

        if acc > best_acc:
            best_acc = acc
            best_clf = lr_model

            pbar.set_description(f"Best acc: {best_acc * 100.0:.2f}")

    pbar.close()

    return best_clf


def perform_nn_link_eval(
    lp_zoo, dataset, edge_split, wb, embeddings: nn.Embedding, model_idx=0, patience=200
):
    """Trains a NN-based link prediction model on the provided embeddings in the transductive setting.
    Returns the trained link predictor and various evaluation metrics.
    """
    data = dataset[0]
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    valid_model = DecoderZoo.filter_models(FLAGS.link_pred_model)[
        model_idx
    ]  # should exist, checked in main()
    model = lp_zoo.get_model(valid_model, embedding_size=embeddings.embedding_dim).to(
        device
    )

    train_edge, valid_edge, test_edge = (
        edge_split['train']['edge'].T.to(device),
        edge_split['valid']['edge'].T.to(device),
        edge_split['test']['edge'].T.to(device),
    )
    valid_edge_neg, test_edge_neg = edge_split['valid']['edge_neg'].T.to(
        device
    ), edge_split['test']['edge_neg'].T.to(device)

    optimizer = AdamW(model.parameters(), lr=FLAGS.link_mlp_lr, weight_decay=FLAGS.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    if FLAGS.batch_links:
        train_dl = DataLoader(range(train_edge.size(0)), FLAGS.link_batch_size, shuffle=True)  # type: ignore

    def train():
        model.train()
        optimizer.zero_grad()

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=train_edge,
            num_nodes=data.num_nodes,
            num_neg_samples=train_edge.size(1),
            method='sparse',
        )

        edge_label_index = torch.cat(
            [train_edge, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat(
            [
                train_edge.new_ones(train_edge.size(1)),
                train_edge.new_zeros(neg_edge_index.size(1)),
            ],
            dim=0,
        )

        edge_embeddings = embeddings(edge_label_index)
        combined = torch.hstack((edge_embeddings[0, :, :], edge_embeddings[1, :, :]))

        out = model(combined)
        loss = criterion(out.view(-1), edge_label.float())
        loss.backward()
        optimizer.step()
        return loss

    def batched_train(epoch):
        model.train()

        total_loss = total_examples = 0
        for perm in train_dl:
            optimizer.zero_grad()

            pos_edge = train_edge[perm]
            pos_combined = torch.hstack(
                (embeddings(pos_edge[0]), embeddings(pos_edge[1]))
            )
            pos_out = model(pos_combined).squeeze()

            if FLAGS.trivial_neg_sampling == 'true':
                neg_edge = torch.randint(
                    0, data.num_nodes, pos_edge.size(), dtype=torch.long, device=device
                )
            elif FLAGS.trivial_neg_sampling == 'false':
                neg_edge = negative_sampling(
                    data.edge_index, data.num_nodes, pos_edge.size(1)
                ).to(device)

            neg_combined = torch.hstack(
                (embeddings(neg_edge[0]), embeddings(neg_edge[1]))
            )
            neg_out = model(neg_combined).squeeze()

            all_out = torch.cat([pos_out, neg_out])
            edge_label = torch.cat(
                [
                    train_edge.new_ones(train_edge.size(1)),
                    train_edge.new_zeros(neg_edge.size(1)),
                ],
                dim=0,
            )

            loss = criterion(all_out.view(-1), edge_label.float())
            loss.backward()
            optimizer.step()

            num_examples = pos_out.size(0)
            total_loss += loss.item() * num_examples
            total_examples += num_examples

        return total_loss / total_examples

    @torch.no_grad()
    def test(valid_edge, neg_edge):
        model.eval()
        n_edges = valid_edge.size(1)
        all_edge_idx = torch.hstack((valid_edge, neg_edge))

        edge_embeddings = embeddings(all_edge_idx)
        combined = torch.hstack((edge_embeddings[0, :, :], edge_embeddings[1, :, :]))
        edge_label = torch.cat(
            [
                train_edge.new_ones(valid_edge.size(1)),
                train_edge.new_zeros(neg_edge.size(1)),
            ],
            dim=0,
        )
        out = model.predict(combined).view(-1)

        y_pred_pos, y_pred_neg = out[:n_edges], out[n_edges:]
        return {
            'loss': float(criterion(out.view(-1), edge_label.float())),
            **eval_all(y_pred_pos, y_pred_neg),
        }

    best_val_hits = final_test_hits = -1
    best_val_res = best_test_res = None
    best_hits_epoch = 0

    TARGET_METRIC = 'hits@50'
    log.info(f'Target metric is: {TARGET_METRIC}')

    for epoch in range(1, FLAGS.link_nn_epochs + 1):
        if FLAGS.batch_links:
            loss = batched_train(epoch)
        else:
            loss = train()

        if wb is not None:
            wb.log({'link_train_loss': loss, 'epoch': epoch})

        val_res = test(valid_edge, valid_edge_neg)
        test_res = test(test_edge, test_edge_neg)
        metric_names = list(test_res.keys())

        for metric_name in metric_names:
            val_hits = val_res[metric_name]
            test_hits = test_res[metric_name]
            if wb is not None:
                wb.log(
                    {
                        f'val_{metric_name}': val_hits,
                        f'test_{metric_name}': test_hits,
                        'epoch': epoch,
                    },
                    step=wb.run.step,
                )

        # Score result with the best hits@50
        if val_res[TARGET_METRIC] > best_val_hits:
            best_hits_epoch = epoch
            best_val_hits = val_res[TARGET_METRIC]
            final_test_hits = test_res[TARGET_METRIC]

            best_val_res = val_res
            best_test_res = test_res
            if wb is not None:
                wb.log(
                    {
                        **{
                            f'best_{metric_name}': best_test_res[metric_name]
                            for metric_name in metric_names
                        },
                        'epoch': epoch,
                    },
                    step=wb.run.step,
                )
        elif epoch - best_hits_epoch > patience:  # early stopping
            break
        if epoch % 5 == 0:
            log.info(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_res[TARGET_METRIC]:.4f}, '
                f'Test: {test_res[TARGET_METRIC]:.4f}'
            )

    log.info(f'Final Test: {final_test_hits:.4f}')
    results = {
        'target_metric': TARGET_METRIC,
        'type': valid_model,
        'val': best_val_res,
        'test': best_test_res,
        'fixed': True,
    }
    return model, results


def perform_inductive_nn_link_eval(
    lp_zoo,
    train_data,
    val_data,
    train_embeddings,
    inference_embeddings,
    device,
    test_edge_bundle,
    negative_samples,
    wb,
    model_idx=0,
    patience=50,
    flags=FLAGS,
):
    """Trains a NN-based link prediction model on the provided embeddings in the inductive setting.
    Returns the trained link predictor and various evaluation metrics.
    """
    assert wb is not None

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    valid_model = DecoderZoo.filter_models(flags.link_pred_model)[
        model_idx
    ]  # should exist, checked in main()
    model = lp_zoo.get_model(
        valid_model, embedding_size=train_embeddings.embedding_dim
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=flags.lr, weight_decay=flags.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    if flags.batch_links:
        train_dl = DataLoader(range(train_edge.size(0)), flags.link_batch_size, shuffle=True)  # type: ignore

    # TODO(author): check for OGBL datasets
    train_edge = train_data.edge_index

    def train():
        model.train()
        optimizer.zero_grad()

        # We perform a new round of negative sampling for every training epoch:
        neg_edge_index = negative_sampling(
            edge_index=train_edge,
            num_nodes=train_data.num_nodes,
            num_neg_samples=train_edge.size(1),
            method='sparse',
        )

        edge_label_index = torch.cat(
            [train_edge, neg_edge_index],
            dim=-1,
        )
        edge_label = torch.cat(
            [
                train_edge.new_ones(train_edge.size(1)),
                train_edge.new_zeros(neg_edge_index.size(1)),
            ],
            dim=0,
        )

        edge_embeddings = train_embeddings(edge_label_index)
        combined = torch.hstack((edge_embeddings[0, :, :], edge_embeddings[1, :, :]))

        out = model(combined)
        loss = criterion(out.view(-1), edge_label.float())
        loss.backward()
        optimizer.step()
        return loss

    def batched_train(epoch):
        raise NotImplementedError()

    @torch.no_grad()
    def test(valid_edge, neg_edge, embeddings):
        model.eval()
        n_edges = valid_edge.size(1)
        all_edge_idx = torch.hstack((valid_edge, neg_edge)).to(device)

        edge_embeddings = embeddings(all_edge_idx)
        combined = torch.hstack((edge_embeddings[0, :, :], edge_embeddings[1, :, :]))
        edge_label = torch.cat(
            [
                train_edge.new_ones(valid_edge.size(1)),
                train_edge.new_zeros(neg_edge.size(1)),
            ],
            dim=0,
        )
        out = model.predict(combined).view(-1)

        y_pred_pos, y_pred_neg = out[:n_edges], out[n_edges:]
        return {
            'loss': float(criterion(out.view(-1), edge_label.float())),
            **eval_all(y_pred_pos, y_pred_neg),
        }

    best_val_hits = final_test_hits = -1
    best_hits_epoch = 0
    TARGET_METRIC = 'hits@50'
    log.info(f'Target metric is: {TARGET_METRIC}')

    (
        test_old_old_ei,
        test_old_new_ei,
        test_new_new_ei,
        test_edge_index,
    ) = test_edge_bundle
    best_old_old_res = (
        best_old_new_res
    ) = best_new_new_res = best_test_res = best_val_res = None

    for epoch in range(1, flags.link_nn_epochs + 1):
        if flags.batch_links:
            loss = batched_train(epoch)
        else:
            loss = train()

        wb.log({'link_train_loss': loss, 'epoch': epoch})

        # test on validation set using train_embeddings
        val_res = test(
            val_data.edge_label_index[:, val_data.edge_label == 1],
            val_data.edge_label_index[:, val_data.edge_label == 0],
            train_embeddings,
        )
        # test on testing set using inference_embeddings
        test_res = test(test_edge_index, negative_samples, inference_embeddings)

        # use same negative samples for all of them
        old_old_res = test(test_old_old_ei, negative_samples, inference_embeddings)
        old_new_res = test(test_old_new_ei, negative_samples, inference_embeddings)
        new_new_res = test(test_new_new_ei, negative_samples, inference_embeddings)

        metric_names = list(test_res.keys())

        for metric_name in metric_names:
            val_hits = val_res[metric_name]
            test_hits = test_res[metric_name]
            old_old_hits = old_old_res[metric_name]
            old_new_hits = old_new_res[metric_name]
            new_new_hits = new_new_res[metric_name]

            wb.log(
                {
                    f'val_{metric_name}': val_hits,
                    f'test_{metric_name}': test_hits,
                    f'oldold_{metric_name}': old_old_hits,
                    f'oldnew_{metric_name}': old_new_hits,
                    f'newnew_{metric_name}': new_new_hits,
                    'epoch': epoch,
                },
                step=wb.run.step,
            )

        if val_res[TARGET_METRIC] > best_val_hits:
            best_hits_epoch = epoch
            best_val_hits = val_res[TARGET_METRIC]
            final_test_hits = test_res[TARGET_METRIC]

            best_val_res = val_res
            best_test_res = test_res
            best_old_old_res = old_old_res
            best_old_new_res = old_new_res
            best_new_new_res = new_new_res

            wb.log(
                {
                    **{
                        f'best_{metric_name}': best_test_res[metric_name]
                        for metric_name in metric_names
                    },
                    **{
                        f'best_old-old_{metric_name}': best_old_old_res[metric_name]
                        for metric_name in metric_names
                    },
                    **{
                        f'best_old-new_{metric_name}': best_old_new_res[metric_name]
                        for metric_name in metric_names
                    },
                    **{
                        f'best_new-new_{metric_name}': best_new_new_res[metric_name]
                        for metric_name in metric_names
                    },
                    'epoch': epoch,  # yapf: disable
                },
                step=wb.run.step,
            )

        elif epoch - best_hits_epoch > patience:  # early stopping
            break
        if epoch % 5 == 0:
            log.info(
                f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Val: {val_res[TARGET_METRIC]:.4f}, '
                f'Test: {test_res[TARGET_METRIC]:.4f}; O-O: {old_old_res[TARGET_METRIC]:.4f}; O-N: {old_new_res[TARGET_METRIC]:.4f}; N-N: {new_new_res[TARGET_METRIC]:.4f}'
            )

    log.info(f'Final Test: {final_test_hits:.4f}')
    results = {
        'target_metric': TARGET_METRIC,
        'type': valid_model,
        'val': best_val_res,
        'test': best_test_res,
        'old_old': best_old_old_res,
        'old_new': best_old_new_res,
        'new_new': best_new_new_res,
        'fixed': True,
    }
    return model, results

def ppi_train_linear_layer(num_classes, train_data, val_data, test_data, device):
    r"""
    Trains a linear layer on top of the representations. This function is specific to the PPI dataset,
    which has multiple labels.
    """
    def train(classifier, train_data, optimizer):
        classifier.train()

        x, label = train_data
        x, label = x.to(device), label.to(device)
        for step in range(100):
            # forward
            optimizer.zero_grad()
            pred_logits = classifier(x)

            # loss and backprop
            loss = criterion(pred_logits, label)
            loss.backward()
            optimizer.step()

    def test(classifier, data):
        classifier.eval()
        x, label = data
        label = label.cpu().numpy().squeeze()

        # feed to network and classifier
        with torch.no_grad():
            pred_logits = classifier(x.to(device))
            pred_class = (pred_logits > 0).float().cpu().numpy()

        return metrics.f1_score(label, pred_class, average='micro') if pred_class.sum() > 0 else 0

    num_feats = train_data[0].size(1)
    criterion = torch.nn.BCEWithLogitsLoss()

    # normalization
    mean, std = train_data[0].mean(0, keepdim=True), train_data[0].std(0, unbiased=False, keepdim=True)
    train_data[0] = (train_data[0] - mean) / std
    val_data[0] = (val_data[0] - mean) / std
    test_data[0] = (test_data[0] - mean) / std

    best_val_f1 = 0
    test_f1 = 0
    for weight_decay in 2.0 ** np.arange(-10, 11, 2):
        classifier = torch.nn.Linear(num_feats, num_classes).to(device)
        optimizer = torch.optim.AdamW(params=classifier.parameters(), lr=0.01, weight_decay=weight_decay)

        train(classifier, train_data, optimizer)
        val_f1 = test(classifier, val_data)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            test_f1 = test(classifier, test_data)

    return best_val_f1, test_f1