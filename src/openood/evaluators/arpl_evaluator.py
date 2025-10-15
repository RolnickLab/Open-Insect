from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openood.postprocessors import BasePostprocessor
from openood.utils import Config

from .ood_evaluator import OODEvaluator
import pandas as pd
import os


def get_osa_threshold(score_known, score_unknown, pred_known, label_known, alpha):

    def osa_at_theta(score_known, score_unknown, pred_known, label_known, theta, alpha):

        def ccr_theta(score_known, pred_known, label_known, theta):
            correct = pred_known == label_known
            confident = score_known >= theta
            return (correct & confident).sum() / len(label_known)

        def urr_theta(score_unknown, theta):
            return (score_unknown < theta).mean()

        osa = alpha * ccr_theta(score_known, pred_known, label_known, theta) + (
            1 - alpha
        ) * urr_theta(score_unknown, theta)

        return osa

    all_scores = np.concatenate([score_known, score_unknown])
    thresholds = np.unique(all_scores)

    osa_list = []
    for theta in thresholds:
        osa = osa_at_theta(
            score_known, score_unknown, pred_known, label_known, theta, alpha
        )
        osa_list.append(osa)

    return max(osa_list), thresholds[np.argmax(osa_list)]


class ARPLEvaluator(OODEvaluator):
    def __init__(self, config: Config):
        self.config = config

    def eval_acc(
        self,
        net: dict,
        data_loader: DataLoader,
        postprocessor: BasePostprocessor = None,
        epoch_idx: int = -1,
        fsood: bool = False,
        csid_data_loaders: Dict[str, DataLoader] = None,
    ):
        criterion = net["criterion"]
        net = net["netF"]
        net.eval()

        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Eval: ", position=0, leave=True):
                # prepare data
                data = batch["data"].cuda()
                target = batch["label"].cuda()

                # forward
                _, feat = net(data, return_feature=True)
                output, loss = criterion(feat, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        if not fsood:
            metrics = {}
            metrics["epoch_idx"] = epoch_idx
            metrics["loss"] = loss_avg / len(data_loader)
            metrics["acc"] = correct / len(data_loader.dataset)
            return metrics
        else:
            all_correct = 0
            all_total = 0
            all_correct += correct
            all_total += len(data_loader.dataset)

            assert csid_data_loaders is not None
            for dataset_name, csid_dl in csid_data_loaders.items():
                correct = 0
                with torch.no_grad():
                    for batch in tqdm(csid_dl, desc="Eval: ", position=0, leave=True):
                        # prepare data
                        data = batch["data"].cuda()
                        target = batch["label"].cuda()

                        # forward
                        _, feat = net(data, return_feature=True)
                        output, loss = criterion(feat, target)

                        # accuracy
                        pred = output.data.max(1)[1]
                        correct += pred.eq(target.data).sum().item()

                all_correct += correct
                all_total += len(csid_dl.dataset)

            metrics = {}
            metrics["epoch_idx"] = epoch_idx
            metrics["acc"] = all_correct / all_total
            return metrics

    def eval_ood(
        self,
        net: dict,
        id_data_loader: DataLoader,
        ood_data_loaders: Dict[str, Dict[str, DataLoader]],
        postprocessor: BasePostprocessor,
        fsood: bool = False,
    ):
        criterion = net["criterion"]
        net = net["netF"]
        net = nn.Sequential(
            net,
            criterion,
        )
        net.eval()
        # load training in-distribution data
        assert "test" in id_data_loader, "id_data_loaders should have the key: test!"
        dataset_name = self.config.dataset.name
        print(f"Performing inference on {dataset_name} dataset...", flush=True)
        id_pred, id_conf, id_gt = postprocessor.inference(net, id_data_loader["test"])
        if self.config.recorder.save_scores:
            self._save_scores(id_pred, id_conf, id_gt, dataset_name)

        ood_pred, ood_conf, ood_gt = postprocessor.inference(
            net, ood_data_loaders["val"]
        )

        osa, threshold = get_osa_threshold(
            id_conf, ood_conf, id_pred, id_gt, 0.5
        )  # negate the scores so we have positive confidence

        postprocessor.osa_threshold = threshold

        print(f"OSA threshold from val set: {postprocessor.osa_threshold}")
        print(f"Val OSA : {osa}")

        if fsood:
            # load csid data and compute confidence
            for dataset_name, csid_dl in ood_data_loaders["csid"].items():
                print(f"Performing inference on {dataset_name} dataset...", flush=True)
                csid_pred, csid_conf, csid_gt = postprocessor.inference(net, csid_dl)
                if self.config.recorder.save_scores:
                    self._save_scores(csid_pred, csid_conf, csid_gt, dataset_name)
                id_pred = np.concatenate([id_pred, csid_pred])
                id_conf = np.concatenate([id_conf, csid_conf])
                id_gt = np.concatenate([id_gt, csid_gt])

        # load nearood data and compute ood metrics
        near_metrics = self._eval_ood(
            net,
            [id_pred, id_conf, id_gt],
            ood_data_loaders,
            postprocessor,
            ood_split="nearood",
        )

        # load farood data and compute ood metrics
        far_metrics = self._eval_ood(
            net,
            [id_pred, id_conf, id_gt],
            ood_data_loaders,
            postprocessor,
            ood_split="farood",
        )

        combined_metrics = near_metrics + far_metrics
        nearood_keys = list(ood_data_loaders["nearood"].keys())
        farood_keys = list(ood_data_loaders["farood"].keys())
        index_labels = nearood_keys + farood_keys
        df = pd.DataFrame(combined_metrics, index=index_labels).round(2)

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.float_format",
            "{:,.2f}".format,
        ):  # more options can be specified also
            print(df)

        csv_path = os.path.join(self.config.output_dir, "ood.csv")
        df.to_csv(csv_path)
