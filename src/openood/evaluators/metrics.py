import numpy as np
from sklearn import metrics
from XCurve.Metrics import OpenAUC
import matplotlib.pyplot as plt


def get_tnr(label, conf, fnr_th):
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1

    fpr, tpr, thresholds = metrics.roc_curve(ood_indicator, -conf)
    target_tpr = 1 - fnr_th
    idx = np.where(tpr >= target_tpr)[0][0]

    # Get the corresponding threshold and tnr
    threshold = thresholds[idx]
    tnr = 1 - fpr[idx]  # tnr = 1 - fpr

    return tnr * 100


def ccr_theta(score_known, pred_known, label_known, theta):
    correct = pred_known == label_known
    confident = score_known >= theta
    return (correct & confident).sum() / len(label_known)


def urr_theta(score_unknown, theta):
    return (score_unknown < theta).mean()


def compute_auoscr(x1, x2, pred, labels):
    """
    :param x1: open set score for each known class sample (B_k,)
    :param x2: open set score for each unknown class sample (B_u,)
    :param pred: predicted class for each known class sample (B_k,)
    :param labels: correct class for each known class sample (B_k,)
    :return: Open Set Classification Rate

    Adapted from https://github.com/sgvaze/osr_closed_set_all_you_need/blob/main/test/utils.py#L125
    """

    x1, x2 = -x1, -x2

    # x1, x2 = np.max(pred_k, axis=1), np.max(pred_u, axis=1)
    # pred = np.argmax(pred_k, axis=1)

    correct = pred == labels
    m_x1 = np.zeros(len(x1))
    m_x1[pred == labels] = 1
    k_target = np.concatenate((m_x1, np.zeros(len(x2))), axis=0)
    u_target = np.concatenate((np.zeros(len(x1)), np.ones(len(x2))), axis=0)
    predict = np.concatenate((x1, x2), axis=0)
    n = len(predict)

    # Cutoffs are of prediction values

    CCR = [0 for x in range(n + 2)]
    FPR = [0 for x in range(n + 2)]

    idx = predict.argsort()

    s_k_target = k_target[idx]
    s_u_target = u_target[idx]

    for k in range(n - 1):
        CC = s_k_target[k + 1 :].sum()
        FP = s_u_target[k:].sum()

        # True	Positive Rate
        CCR[k] = float(CC) / float(len(x1))
        # False Positive Rate
        FPR[k] = float(FP) / float(len(x2))

    CCR[n] = 0.0
    FPR[n] = 0.0
    CCR[n + 1] = 1.0
    FPR[n + 1] = 1.0

    # Positions of ROC curve (FPR, TPR)
    ROC = sorted(zip(FPR, CCR), reverse=True)
    FPR_sorted, CCR_sorted = zip(*ROC)
    AUOSCR = 0
    # Compute AUROC Using Trapezoidal Rule
    for j in range(n + 1):
        h = ROC[j][0] - ROC[j + 1][0]
        w = (ROC[j][1] + ROC[j + 1][1]) / 2.0
        AUOSCR = AUOSCR + h * w

    return AUOSCR, FPR_sorted, CCR_sorted


def get_osa_threshold(score_known, score_unknown, pred_known, label_known, alpha):
    all_scores = np.concatenate([score_known, score_unknown])
    thresholds = np.unique(all_scores)

    osa_list = []
    for theta in thresholds:
        osa = osa_at_theta(
            score_known, score_unknown, pred_known, label_known, theta, alpha
        )
        osa_list.append(osa)

    return max(osa_list), thresholds[np.argmax(osa_list)]


def osa_at_theta(score_known, score_unknown, pred_known, label_known, theta, alpha):
    osa = alpha * ccr_theta(score_known, pred_known, label_known, theta) + (
        1 - alpha
    ) * urr_theta(score_unknown, theta)

    return osa


def compute_all_metrics(conf, label, pred, theta):
    np.set_printoptions(precision=3)

    results = {}
    recall = 0.95
    results["TNR@10"] = get_tnr(label, conf, 0.1)
    results["TNR@20"] = get_tnr(label, conf, 0.2)
    results["AUROC"], results["AUPR"], results["FPR"] = auc_and_fpr_recall(
        conf, label, recall
    )

    results["ACC"] = acc(pred, label) * 100

    score_known, score_unknown = -conf[label != -1], -conf[label == -1]

    pred_known, label_known = pred[label != -1], label[label != -1]
    AUOSCR, FPR_sorted, CCR_sorted = compute_auoscr(
        score_known, score_unknown, pred_known, label_known
    )

    results["AUOSCR"] = AUOSCR * 100
    results["OpenAUC"] = (
        OpenAUC(score_known, score_unknown, pred_known, label_known) * 100
    )
    if theta:
        oosa = osa_at_theta(
            -score_known, -score_unknown, pred_known, label_known, theta, alpha=0.5
        )  # negate the scores so we have positive confidence
        osa, _ = get_osa_threshold(
            -score_known, -score_unknown, pred_known, label_known, alpha=0.5
        )

        results["OSA"] = osa * 100
        results["OOSA"] = oosa * 100
    else:
        results["OSA"] = -1
        results["OOSA"] = -1

    return results, FPR_sorted, CCR_sorted


# accuracy
def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)

    return acc


# fpr_recall
def fpr_recall(conf, label, tpr):
    gt = np.ones_like(label)
    gt[label == -1] = 0

    fpr_list, tpr_list, threshold_list = metrics.roc_curve(gt, conf)
    fpr = fpr_list[np.argmax(tpr_list >= tpr)]
    thresh = threshold_list[np.argmax(tpr_list >= tpr)]
    return fpr, thresh


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
    return auroc * 100, aupr * 100, fpr * 100


# ccr_fpr
def ccr_fpr(conf, fpr, pred, label):
    ind_conf = conf[label != -1]
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]

    ood_conf = conf[label == -1]

    num_ind = len(ind_conf)
    num_ood = len(ood_conf)

    fp_num = int(np.ceil(fpr * num_ood))
    thresh = np.sort(ood_conf)[-fp_num]
    num_tp = np.sum((ind_conf > thresh) * (ind_pred == ind_label))
    ccr = num_tp / num_ind

    return ccr


def detection(ind_confidences, ood_confidences, n_iter=100000, return_data=False):
    # calculate the minimum detection error
    Y1 = ood_confidences
    X1 = ind_confidences

    start = np.min([np.min(X1), np.min(Y1)])
    end = np.max([np.max(X1), np.max(Y1)])
    gap = (end - start) / n_iter

    best_error = 1.0
    best_delta = None
    all_thresholds = []
    all_errors = []
    for delta in np.arange(start, end, gap):
        tpr = np.sum(np.sum(X1 < delta)) / np.float(len(X1))
        error2 = np.sum(np.sum(Y1 > delta)) / np.float(len(Y1))
        detection_error = (tpr + error2) / 2.0

        if return_data:
            all_thresholds.append(delta)
            all_errors.append(detection_error)

        if detection_error < best_error:
            best_error = np.minimum(best_error, detection_error)
            best_delta = delta

    if return_data:
        return best_error, best_delta, all_errors, all_thresholds
    else:
        return best_error, best_delta
