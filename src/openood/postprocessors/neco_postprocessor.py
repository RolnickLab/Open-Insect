from typing import Any
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from .base_postprocessor import BasePostprocessor
from sklearn.decomposition import PCA
from numpy import linalg


class NecoPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.dim = self.args.dim
        self.setup_flag = False

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        if not self.setup_flag:
            # pca for training ID data
            ss = StandardScaler()

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
            print(f" Train acc: {train_acc:.2%}")
            complete_vectors_train = ss.fit_transform(all_feats)
            self.ss = ss
            pca_estimator = PCA(all_feats.shape[1])
            _ = pca_estimator.fit_transform(complete_vectors_train)
            self.pca_estimator = pca_estimator

        else:
            pass

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        logits, features = net(data, return_feature=True)
        features = features.cpu()
        pred = logits.argmax(1)
        complete_vectors = self.ss.transform(features)
        cls_reduced_all = self.pca_estimator.transform(complete_vectors)
        cls_reduced = cls_reduced_all[:, : self.dim]
        conf = []
        for i in range(cls_reduced.shape[0]):
            sc_complet = linalg.norm((complete_vectors[i, :]))
            sc = linalg.norm(cls_reduced[i, :])
            sc_finale = sc / sc_complet
            conf.append(sc_finale)
        conf = torch.tensor(conf)
        return pred, conf

    def set_hyperparam(self, hyperparam: list):
        self.dim = hyperparam[0]

    def get_hyperparam(self):
        return self.dim
