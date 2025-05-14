from typing import Any
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .base_postprocessor import BasePostprocessor
import torch.nn.functional as F

import openood.utils.comm as comm


class OSRBranchPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.num_classes = self.config.dataset.num_classes
        self.setup_flag = False
        self.all_feats = None
        self.all_labels = None

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # extract reference embedding
            print("\n Extract training features...")
            all_feats = []
            all_labels = []
            all_preds = []
            with torch.no_grad():
                for batch in tqdm(id_loader_dict["train"], desc="Setup: ", position=0, leave=True):
                    data, labels = batch["data"].cuda(), batch["label"].cuda()
                    logits, features = net(data, return_feature=True)
                    all_feats.append(features)
                    all_labels.append(deepcopy(labels))
                    all_preds.append(logits.argmax(1))

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
        logits, features = net(data, return_feature=True)
        pred = torch.argmax(logits.detach(), dim=1)
        features1, features2 = get_feature_pair(features, pred, self.all_feats, self.all_labels, "cosine_similarity")
        conf = net([features1, features2], return_score=True)
        return pred, conf


def get_feature_pair(
    features,
    pred_labels,
    all_feats,
    all_labels,
    similarity_measure,
):
    features1 = torch.tensor([], dtype=torch.float32, device=features.device)
    features2 = torch.tensor([], dtype=torch.float32, device=features.device)
    if pred_labels.dim() == 0:
        pred_labels = pred_labels.unsqueeze(0)
    for i, species in enumerate(pred_labels):
        f1 = features[i]
        feat_ref = all_feats[all_labels == species]

        assert len(feat_ref) > 0
        if similarity_measure == "cosine_similarity":
            cos_sim = F.cosine_similarity(f1.expand_as(feat_ref), feat_ref, dim=1)
            index = torch.argmax(cos_sim).item()
        elif similarity_measure == "euclidean_distance":
            distance = F.pairwise_distance(f1.expand_as(feat_ref), feat_ref)
            index = torch.argmin(distance).item()

        f2 = feat_ref[index].unsqueeze(0)
        features1 = torch.cat((features1, f1.unsqueeze(0)), axis=0)
        features2 = torch.cat((features2, f2), axis=0)

    return features1, features2
