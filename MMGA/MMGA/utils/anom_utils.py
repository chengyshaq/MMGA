import numpy as np
import pandas as pd
import torch.nn as nn
import sklearn.metrics as sk
import sklearn.neighbors
import sklearn.ensemble
import time
import torch
from torch.autograd import Variable
import os.path

recall_level_default = 0.95

class ToLabel(object):
     def __call__(self, inputs):
        return (torch.from_numpy(np.array(inputs)).long())


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)  # unique 函数取出真实标签中的不重复元素。
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values  对模型预测得分进行降序排列，并按照此顺序对真实标签进行对应修改。
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve. 找出模型预测得分中不同值所对应的索引，并添加一个索引表示预测得分最低值，存储在threshold_idxs中。
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]  # r_沿着第一个轴将切片对象转换为串联。这是一种快速构建数组的简单方法。

    # accumulate the true positives with decreasing threshold  计算每个阈值下正例和负例被正确预测的数量，其中tps表示真正例，fps表示假正例
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]  #将阈值和每个阈值下的召回率（recall）分别存储起来。召回率表示所有正例中被正确预测的比例。


    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]  # 将阈值和每个阈值下的召回率（recall）分别存储起来。召回率表示所有正例中被正确预测的比例。
    # 自加画表格
    writer = pd.ExcelWriter('/home/yscheng/px/multi-label-ood-master-jointenergy/visualization/exp3yuzhi/pascal_recall.xlsx')
    recall_ = pd.DataFrame(recall)
    thresholds_ = pd.DataFrame(thresholds)
    recall_.to_excel(writer, 'recall', float_format='%.5f')
    thresholds_.to_excel(writer, 'thresholds', float_format='%.5f')
    writer.save()
    writer.close()
    print("yuzhi_huatu")
    
    cutoff = np.argmin(np.abs(recall - recall_level))  #将阈值和召回率列表逆序，确保按照降序排列。如果最高召回率（即 1.0）不在列表中，则加入；fps、tps列表也被逆序以便求解FPR等指标。
    if np.array_equal(classes, [1]):
        return thresholds[cutoff]  # return threshold  如果只有正类，则直接返回阈值。

    return fps[cutoff] / (np.sum(np.logical_not(y_true))), thresholds[cutoff]  #如果有负类，则计算并返回 FPR 和阈值，其中 FPR 表示在指定召回率下模型预测为假正例的比率。


def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr, threshould = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr, threshould

def print_measures(auroc, aupr, fpr, ood, method, recall_level=recall_level_default):
    print('\t\t\t' + ood+'_'+method)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))

def get_and_print_results(out_score, in_score, ood, method):
    aurocs, auprs, fprs = [], [], []
    measures = get_measures(out_score, in_score)
    aurocs.append(measures[0]); auprs.append(measures[1]); fprs.append(measures[2])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); fpr = np.mean(fprs)

    print_measures(auroc, aupr, fpr, ood, method)
    return auroc, aupr, fpr

def get_localoutlierfactor_scores(val, test, out_scores):
    scorer = sklearn.neighbors.LocalOutlierFactor(novelty=True)
    print("fitting validation set")
    start = time.time()
    scorer.fit(val)
    end = time.time()
    print("fitting took ", end - start)
    val = np.asarray(val)
    test = np.asarray(test)
    out_scores = np.asarray(out_scores)
    print(val.shape, test.shape, out_scores.shape)
    return scorer.score_samples(np.vstack((test, out_scores)))


def get_isolationforest_scores(val, test, out_scores):
    scorer = sklearn.ensemble.IsolationForest()
    print("fitting validation set")
    start = time.time()
    scorer.fit(val)
    end = time.time()
    print("fitting took ", end - start)
    val = np.asarray(val)
    test = np.asarray(test)
    out_scores = np.asarray(out_scores)
    print(val.shape, test.shape, out_scores.shape)
    return scorer.score_samples(np.vstack((test, out_scores)))

