import torch
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering, KMeans, MiniBatchKMeans

from scipy.optimize import linear_sum_assignment as linear_assignment
import random
import os
import argparse

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score

from sklearn import metrics
import numpy as np


def extract_features(net, data_loader):
    all_feats = torch.tensor([], device="cuda")

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Extracting features: ", position=0, leave=True):
            data = batch["data"].cuda()
            _, features = net(data, return_feature=True)
            all_feats = torch.cat([all_feats, features], axis=0)

    return all_feats


def clustering(config, features):
    if config.clustering.method == "agglomerative":
        clustering = AgglomerativeClustering().fit(features)
    elif config.clustering.method == "kmeans":
        # clustering = KMeans(n_clusters=config.clustering.num_clusters, random_state=0, n_init="auto").fit(features)
        clustering = MiniBatchKMeans(n_clusters=config.clustering.n_clusters, batch_size=1024, random_state=42).fit(
            features
        )

    pred = clustering.predict(features)
    return pred



# adapted from generalized-category-discovery https://github.com/sgvaze/generalized-category-discovery/tree/main

def cluster_acc(y_true, y_pred, return_ind=False):
    """
    Calculate clustering accuracy. Require scikit-learn installed

    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`

    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(int)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    ind = linear_assignment(w.max() - w)
    ind = np.vstack(ind).T

    if return_ind:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size, ind, w
    else:
        return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
