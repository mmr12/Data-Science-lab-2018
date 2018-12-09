import numpy as np


# here we'll take a probability distribution over the outputs (ie predict_proba)
# check if one of the top n FAQs match the ground truth
#
# inputs
# what_to_predict:
#   0: F1 score
#   1: precision
#   2: recall
#  99: precision, recall, F1-score


def multilabel_prec(y, y_pred_proba, what_to_predict=1, nvals=5):
    y_preds = np.argsort(y_pred_proba, axis=1)[:, -nvals:]
    TP = np.sum([is_in(y_preds[i, :], y[i]) for i in range(len(y))])
    FP = [np.sum([is_in(y_preds[i, :], j) for i in range(len(y))]) for j in range(len(y))]
    FN = [np.sum([is_not_in(y_preds[i, :], j) for i in range(len(y))]) for j in range(len(y))]
    precision = TP / (TP + np.mean(FP))
    recall = TP / (TP + np.mean(FN))
    F1 = 2 * precision * recall / (precision + recall)
    if what_to_predict == 0:
        return F1
    elif what_to_predict == 1:
        return precision
    elif what_to_predict == 2:
        return recall
    elif what_to_predict == 99:
        return precision, recall, F1


def is_in(y_hat, x):
    return min(np.sum(np.isin(y_hat, x)), 1)


def is_not_in(y_hat, x):
    return min(np.sum(np.isin(y_hat, x, invert=True)), 1)
