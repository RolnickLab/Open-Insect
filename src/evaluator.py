from typing import Callable, List, Type

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import pathlib
import json
from collections import Counter
from openood.evaluation_api import Evaluator

# need to change the import
from openood.evaluation_api.postprocessor import get_postprocessor
from openood.evaluators.metrics import acc
from openood.networks.ash_net import ASHNet
from openood.networks.react_net import ReactNet
from openood.networks.scale_net import ScaleNet
from sklearn import metrics


class BioEvaluator(Evaluator):
    def __init__(
        self,
        net,
        config=None,
        data_config_path=None,
        config_root="configs",
        dataloader_dict=None,
        postprocessor_name=None,
        postprocessor=None,
        save_arrays=False,
    ) -> None:

        # check the arguments
        if postprocessor_name is None and postprocessor is None:
            raise ValueError("Please pass postprocessor_name or postprocessor")
        if postprocessor_name is not None and postprocessor is not None:
            print("Postprocessor_name is ignored because postprocessor is passed")

        # get postprocessor
        if postprocessor is None:
            postprocessor = get_postprocessor(
                config_root, postprocessor_name, "cifar10"
            )  # use cifar10 to set the number of classes of the postprocessor

        self._update_attributes(postprocessor, config)
        self.config = config

        # wrap base model to work with certain postprocessors
        if postprocessor_name == "react":
            net = ReactNet(net)
        elif postprocessor_name == "ash":
            net = ASHNet(net)
        elif postprocessor_name == "scale":
            net = ScaleNet(net)

        # postprocessor setup
        postprocessor.setup(net, dataloader_dict["id"], dataloader_dict["ood"])
        self.save_arrays = save_arrays
        if self.save_arrays:
            save_dir = ("/").join(self.config.network.checkpoint.split("/")[:-1])
            save_dir += f"/{self.config.dataset.name}/{self.config.trainer.name}/{self.config.postprocessor.name}"
            pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
            self.save_dir = save_dir
            print(self.save_dir)
        self.id_name = config.dataset.name
        self.net = net
        self.postprocessor = postprocessor
        self.postprocessor_name = postprocessor_name
        self.dataloader_dict = dataloader_dict
        self.metrics = {"id_acc": None, "csid_acc": None, "ood": None, "fsood": None}
        self.scores = {
            "id": {"train": None, "val": None, "test": None},
            "ood": {
                "val": None,
                "nearood": {k: None for k in dataloader_dict["ood"]["nearood"].keys()},
                "farood": {k: None for k in dataloader_dict["ood"]["farood"].keys()},
            },
            "id_preds": None,
            "id_labels": None,
        }
        # perform hyperparameter search if have not done so
        if (
            self.postprocessor.APS_mode
            and not self.postprocessor.hyperparam_search_done
        ):
            self.hyperparam_search()

        self.net.eval()

    def _update_attributes(self, postprocessor, config):
        num_classes = config.dataset.num_classes
        if hasattr(postprocessor, "num_classes"):
            postprocessor.num_classes = num_classes

        if hasattr(postprocessor, "nc"):
            postprocessor.nc = num_classes

        # for RP
        if hasattr(postprocessor, "targets"):
            label_filename = config.dataset.train.csv_path
            with open(config.dataset.train.json_path, "r") as file:
                category_map = json.load(file)
            species = pd.read_csv(label_filename).speciesKey
            cls_idx = [
                category_map[str(s)] if str(s) in category_map.keys() else -1
                for s in species
            ]
            # TODO: update cls_idx
            cls_idx = np.array(cls_idx, dtype="int")
            label_stat = Counter(cls_idx)
            cls_num = [-1 for _ in range(num_classes)]
            for i in range(num_classes):
                cat_num = int(label_stat[i])
                cls_num[i] = cat_num
            targets = cls_num / np.sum(cls_num)
            targets = torch.tensor(targets).cuda()
            targets = targets.unsqueeze(0)
            postprocessor.targets = targets

    def eval_ood(self, fsood: bool = False, progress: bool = True):
        id_name = "id" if not fsood else "csid"
        task = "ood" if not fsood else "fsood"
        if self.metrics[task] is None:
            self.net.eval()

            # id score
            if self.scores["id"]["test"] is None:
                print(f"Performing inference on {self.id_name} test set...", flush=True)
                id_pred, id_conf, id_gt = self.postprocessor.inference(
                    self.net, self.dataloader_dict["id"]["test"], progress
                )
                if self.save_arrays:
                    np.save(f"{self.save_dir}/id_test_pred.npy", id_pred)
                    np.save(f"{self.save_dir}/id_test_conf.npy", id_conf)

                self.scores["id"]["test"] = [id_pred, id_conf, id_gt]
            else:
                id_pred, id_conf, id_gt = self.scores["id"]["test"]

            if fsood:
                csid_pred, csid_conf, csid_gt = [], [], []
                for i, dataset_name in enumerate(self.scores["csid"].keys()):
                    if self.scores["csid"][dataset_name] is None:
                        print(
                            f"Performing inference on {self.id_name} "
                            f"(cs) test set [{i+1}]: {dataset_name}...",
                            flush=True,
                        )
                        temp_pred, temp_conf, temp_gt = self.postprocessor.inference(
                            self.net,
                            self.dataloader_dict["csid"][dataset_name],
                            progress,
                        )
                        self.scores["csid"][dataset_name] = [
                            temp_pred,
                            temp_conf,
                            temp_gt,
                        ]

                    csid_pred.append(self.scores["csid"][dataset_name][0])
                    csid_conf.append(self.scores["csid"][dataset_name][1])
                    csid_gt.append(self.scores["csid"][dataset_name][2])

                csid_pred = np.concatenate(csid_pred)
                csid_conf = np.concatenate(csid_conf)
                csid_gt = np.concatenate(csid_gt)

                id_pred = np.concatenate((id_pred, csid_pred))
                id_conf = np.concatenate((id_conf, csid_conf))
                id_gt = np.concatenate((id_gt, csid_gt))

            # load nearood data and compute ood metrics
            near_metrics = self._eval_ood(
                [id_pred, id_conf, id_gt], ood_split="nearood", progress=progress
            )
            # load farood data and compute ood metrics
            far_metrics = self._eval_ood(
                [id_pred, id_conf, id_gt], ood_split="farood", progress=progress
            )

            if self.metrics[f"{id_name}_acc"] is None:
                self.eval_acc(id_name)
            near_metrics[:, -1] = np.array(
                [self.metrics[f"{id_name}_acc"]] * len(near_metrics)
            )
            far_metrics[:, -1] = np.array(
                [self.metrics[f"{id_name}_acc"]] * len(far_metrics)
            )

            df = pd.DataFrame(
                np.concatenate([near_metrics, far_metrics], axis=0),
                index=list(self.dataloader_dict["ood"]["nearood"].keys())
                + ["nearood"]
                + list(self.dataloader_dict["ood"]["farood"].keys())
                + ["farood"],
                columns=["TPR@5", "FPR@95", "AUROC", "AUPR", "ACC"],
            )

            self.metrics[task] = df
        else:
            print("Evaluation has already been done!")

        with pd.option_context(
            "display.max_rows",
            None,
            "display.max_columns",
            None,
            "display.float_format",
            "{:,.2f}".format,
        ):  # more options can be specified also
            print(self.metrics[task])

        return self.metrics[task]

    def _classifier_inference(
        self, data_loader: DataLoader, msg: str = "Acc Eval", progress: bool = True
    ):
        self.net.eval()

        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in data_loader:
                data = batch["data"].cuda()
                logits = self.net(data)
                preds = logits.argmax(1)
                all_preds.append(preds.cpu())
                all_labels.append(batch["label"])

        all_preds = torch.cat(all_preds)
        all_labels = torch.cat(all_labels)
        return all_preds, all_labels

    def _eval_ood(
        self,
        id_list: List[np.ndarray],
        ood_split: str = "nearood",
        progress: bool = True,
    ):
        print(f"Processing {ood_split} ood...", flush=True)
        [id_pred, id_conf, id_gt] = id_list
        metrics_list = []
        for dataset_name, ood_dl in self.dataloader_dict["ood"][ood_split].items():
            if self.scores["ood"][ood_split][dataset_name] is None:
                print(f"Performing inference on {dataset_name} dataset...", flush=True)
                ood_pred, ood_conf, ood_gt = self.postprocessor.inference(
                    self.net, ood_dl, progress
                )
                if self.save_arrays:

                    np.save(
                        f"{self.save_dir}/{ood_split}_{dataset_name}_pred.npy", ood_pred
                    )
                    np.save(
                        f"{self.save_dir}/{ood_split}_{dataset_name}_conf.npy", ood_conf
                    )
                self.scores["ood"][ood_split][dataset_name] = [
                    ood_pred,
                    ood_conf,
                    ood_gt,
                ]
            else:
                print(
                    "Inference has been performed on " f"{dataset_name} dataset...",
                    flush=True,
                )
                [ood_pred, ood_conf, ood_gt] = self.scores["ood"][ood_split][
                    dataset_name
                ]

            ood_gt = -1 * np.ones_like(ood_gt)  # hard set to -1 as ood
            pred = np.concatenate([id_pred, ood_pred])
            conf = np.concatenate([id_conf, ood_conf])
            label = np.concatenate([id_gt, ood_gt])

            conf = conf.squeeze()
            # remove nan or inf
            mask = np.isfinite(conf)

            if len(conf[~mask]) > 0:
                print(f"remove {len(conf[~mask])} nan/ inf values")

            conf, label, pred = conf[mask], label[mask], pred[mask]

            print(f"Computing metrics on {dataset_name} dataset...")
            ood_metrics = compute_all_metrics(conf, label, pred)
            metrics_list.append(ood_metrics)

        print("Computing mean metrics...", flush=True)
        metrics_list = np.array(metrics_list)
        metrics_mean = np.mean(metrics_list, axis=0, keepdims=True)
        return np.concatenate([metrics_list, metrics_mean], axis=0) * 100

    # TODO: unpack
    def _print_metrics(self, metrics):
        [fpr, auroc, aupr_in, aupr_out, _] = metrics

        # print ood metric results
        print(
            "FPR@95: {:.2f}, AUROC: {:.2f}".format(100 * fpr, 100 * auroc),
            end=" ",
            flush=True,
        )
        print(
            "AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}".format(100 * aupr_in, 100 * aupr_out),
            flush=True,
        )
        print("\u2500" * 70, flush=True)
        print("", flush=True)


def get_tpr_at_fpr(label, conf, fpr_th):
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1  # OOD = 1, ID = 0

    fpr, tpr, thresholds = metrics.roc_curve(ood_indicator, -conf)
    idx = np.where(fpr <= fpr_th)[-1][-1]

    threshold = thresholds[idx]
    tpr_at_fpr = tpr[idx]

    return tpr_at_fpr


def get_tnr(label, conf, fnr_th):
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    fpr, tpr, thresholds = metrics.roc_curve(ood_indicator, -conf)
    target_tpr = 1 - fnr_th
    idx = np.where(tpr >= target_tpr)[0][0]

    # Get the corresponding threshold and tnr
    threshold = thresholds[idx]
    tnr = 1 - fpr[idx]  # tnr = 1 - fpr

    return tnr


def auc_and_fpr_recall(conf, label, tpr_th):
    """
    code adapted from openood
    """
    # following convention in ML we treat OOD as positive
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr_th)]

    auroc = metrics.roc_auc_score(ood_indicator, -conf)
    aupr = metrics.average_precision_score(ood_indicator, -conf)
    return auroc, aupr, fpr


def compute_all_metrics(conf, label, pred):
    np.set_printoptions(precision=3)
    recall = 0.95
    auroc, aupr, fpr = auc_and_fpr_recall(conf, label, recall)
    tpr = get_tpr_at_fpr(label, conf, fpr_th=0.05)

    accuracy = acc(pred, label)

    results = [tpr, fpr, auroc, aupr, accuracy]

    return results
