import numpy as np
from sklearn.metrics import roc_auc_score, hamming_loss
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score

def macro_aucroc(y_true, y_pred_prob):
    aucs = []
    for c in range(y_true.shape[1]):
        aucs.append(roc_auc_score(y_true[:,c], y_pred_prob[:,c]))
    return np.mean(aucs), aucs

def exact_match_ratio(y_true_bin, y_pred_bin):
    cnt = 0
    for gt,pd in zip(y_true_bin, y_pred_bin):
        if np.allclose(gt,pd):
            cnt += 1
    return cnt / len(y_true_bin)

def calc_all_metrics(y_true_bin, y_pred_prob, threshold=0.5):
    y_pred_bin = (y_pred_prob >= threshold).astype(int)
    macro_auc, per_auc = macro_aucroc(y_true_bin, y_pred_prob)
    emr = exact_match_ratio(y_true_bin, y_pred_bin)
    hl = hamming_loss(y_true_bin, y_pred_bin)
    return {
        "macro_auc":macro_auc,
        "per_class_auc":per_auc,
        "exact_match":emr,
        "hamming_loss":hl
    }

def ordinal_metrics(y_true_ord, y_pred_ord):
    qwk = cohen_kappa_score(y_true_ord, np.rint(y_pred_ord).astype(int), weights="quadratic")
    rho,_ = spearmanr(y_true_ord, y_pred_ord)
    mae = np.mean(np.abs(y_true_ord - y_pred_ord))
    return {"qwk":qwk, "spearman_rho":rho, "mae":mae}
