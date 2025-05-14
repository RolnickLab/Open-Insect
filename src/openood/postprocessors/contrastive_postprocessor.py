from typing import Any
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .base_postprocessor import BasePostprocessor
import torch.nn.functional as F

import openood.utils.comm as comm


class ContrastivePostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_classes = self.config.dataset.num_classes
        self.setup_flag = False
        self.all_feats = None
        self.all_labels

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # extract reference embedding
            all_feats = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict["train"], desc="Setup: ", position=0, leave=True):
                    data, labels = batch["data"].cuda(), batch["label"]
                    logits, features = net(data, return_feature=True)
                    all_feats.append(features.cpu())
                    all_labels.append(deepcopy(labels))
                    all_preds.append(logits.argmax(1).cpu())

            all_feats = torch.cat(all_feats)
            all_labels = torch.cat(all_labels)
            all_preds = torch.cat(all_preds)
            # sanity check on train acc
            train_acc = all_preds.eq(all_labels).float().mean()

            self.all_feats = all_feats
            self.all_labels = all_labels
            print(f" Train acc: {train_acc:.2%}")

        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feat_post=True)
        pred = torch.argmax(logits.detach(), dim=1)
        conf = get_distance(features, pred, self.all_feats, self.all_labels, "cosine_similarity")
        return pred, conf


def get_reference_eval_pred(
    features,
    pred_labels,
    all_feats,
    all_labels,
    similarity_measure,
):
    embeddings1 = torch.tensor([], dtype=torch.float32, device=features.device)
    embeddings2 = torch.tensor([], dtype=torch.float32, device=features.device)

    # values = torch.tensor([], dtype=torch.float32, device=features.device)
    if pred_labels.dim() == 0:
        pred_labels = pred_labels.unsqueeze(0)
    for i, species in enumerate(pred_labels):
        e1 = features[i]
        embedding_ref = all_feats[all_labels == species]

        assert len(embedding_ref) > 0
        if similarity_measure == "cosine_similarity":
            cos_sim = F.cosine_similarity(e1.expand_as(embedding_ref), embedding_ref, dim=1)
            index = torch.argmax(cos_sim).item()
            # value = cos_sim.max()
        elif similarity_measure == "euclidean_distance":
            distance = F.pairwise_distance(e1.expand_as(embedding_ref), embedding_ref)
            index = torch.argmin(distance).item()
            # distance = distance.min()

        # values = torch.cat((values, value), axis=0)

        e2 = embedding_ref[index].unsqueeze(0)
        embeddings1 = torch.cat((embeddings1, e1.unsqueeze(0)), axis=0)
        embeddings2 = torch.cat((embeddings2, e2), axis=0)

    return embeddings1, embeddings2


def get_distance(
    features,
    pred_labels,
    all_feats,
    all_labels,
    similarity_measure,
):
    # embeddings1 = torch.tensor([], dtype=torch.float32, device=features.device)
    # embeddings2 = torch.tensor([], dtype=torch.float32, device=features.device)

    values = torch.tensor([], dtype=torch.float32, device=features.device)
    if pred_labels.dim() == 0:
        pred_labels = pred_labels.unsqueeze(0)
    for i, species in enumerate(pred_labels):
        e1 = features[i]
        embedding_ref = all_feats[all_labels == species]

        assert len(embedding_ref) > 0
        if similarity_measure == "cosine_similarity":
            cos_sim = F.cosine_similarity(e1.expand_as(embedding_ref), embedding_ref, dim=1)
            value = cos_sim.max()
        elif similarity_measure == "euclidean_distance":
            distance = F.pairwise_distance(e1.expand_as(embedding_ref), embedding_ref)
            distance = distance.min()

        values = torch.cat((values, value), axis=0)
    return values
