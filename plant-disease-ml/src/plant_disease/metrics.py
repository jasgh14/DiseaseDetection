from typing import Dict, Tuple
import numpy as np

def _precision_recall_f1(tp, fp, fn, eps=1e-9):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    return precision, recall, f1

def multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    assert y_true.shape == y_pred.shape
    tp = (y_true * y_pred).sum(axis=0)
    fp = ((1 - y_true) * y_pred).sum(axis=0)
    fn = (y_true * (1 - y_pred)).sum(axis=0)

    p_c, r_c, f1_c = _precision_recall_f1(tp, fp, fn)
    macro_p = float(np.mean(p_c))
    macro_r = float(np.mean(r_c))
    macro_f1 = float(np.mean(f1_c))

    TP = tp.sum()
    FP = fp.sum()
    FN = fn.sum()
    micro_p, micro_r, micro_f1 = _precision_recall_f1(TP, FP, FN)

    # subset accuracy
    subset_acc = float((y_true == y_pred).all(axis=1).mean())

    # Hamming loss
    hamming = float((y_true != y_pred).mean())

    return {
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "subset_accuracy": float(subset_acc),
        "hamming_loss": float(hamming),
    }
