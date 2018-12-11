import numpy as np
import parmap
from sklearn.model_selection import KFold


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
    # predictions:
    y_preds = np.argsort(y_pred_proba, axis=1)[:, -nvals:] - 1
    if np.sum(y_pred_proba) == 0:
        y_preds = np.zeros(y_preds.shape) - 1
    # classes to check:
    TP = [np.sum([is_in(y_preds[i, :], y[j]) for i in range(len(y))]) for j in range(len(y))]
    # print("weighted mean TP", np.mean(TP), "nans", np.sum(np.isnan(TP)))
    FP = [np.sum([is_in(y_preds[i, :], j) for i in range(len(y))]) for j in range(len(y))]
    #print("weighted mean FP", np.mean(FP), "nans", np.sum(np.isnan(FP)))
    FN = [np.sum([is_not_in(y_preds[i, :], j) for i in range(len(y))]) for j in range(len(y))]
    # print("weighted mean FN", np.mean(FN))
    precision = np.round(np.nansum([TP[i] / (TP[i] + FP[i]) for i in range(len(y))]) / len(y), 3)
    recall = np.round(np.nansum([TP[i] / (TP[i] + FN[i]) for i in range(len(y))]) / len(y), 3)
    F1 = np.round(2 * precision * recall / (precision + recall), 3)
    if what_to_predict == 0:
        return F1
    elif what_to_predict == 1:
        return precision
    elif what_to_predict == 2:
        return recall
    elif what_to_predict == 99:
        return precision, recall, F1


# support functions

# inputs
#   y_hat: array
#       x: value
# output
#       1: x is in y_hat
#       0: x is not in y_hat
def is_in(y_hat, x):
    return min(np.sum(np.isin(y_hat, x)), 1)


# inputs
#   y_hat: array
#       x: value
# output
#       1: x is not in y_hat
#       0: x is in y_hat
def is_not_in(y_hat, x):
    return min(np.sum(np.isin(y_hat, x, invert=True)), 1)


# custom cross validation
# cross validates over a probability distribution over classes
# input
#    estimator: classification function. Allows for .fit and .predict_proba
#            X: x-values n x d
#            y: x-values n x 1
#      scoring: assumed to be multilabel_prec atm
# scoring_arg1: 0: F1, 1:precision, 2:recall, 99: precision, recall, F1-score
# scoring_arg2: how many FAQs will be returned
#     n_splits: cv splits
# output
# cv of scoring_arg1
def cross_val_proba_score(estimator, X, y, scoring=multilabel_prec, scoring_arg1=1, scoring_arg2=5, n_splits=5):
    kf = KFold(n_splits, shuffle=True)
    if scoring_arg1 == 99:
        score = np.zeros(3)
    else:
        score = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        estimator.fit(X_train, y_train)
        y_hat = estimator.predict_proba(X_test)
        score += scoring(y_test, y_hat, what_to_predict=scoring_arg1, nvals=scoring_arg2)

    score = score / n_splits
    return score


def cross_val_proba_score_parmap(estimator, X, y, scoring=multilabel_prec, scoring_arg1=1, scoring_arg2=5, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    scores = parmap.starmap(parmap_wrap, kf.split(X),
                            X, y, estimator, scoring=scoring, scoring_arg1=scoring_arg1, scoring_arg2=scoring_arg2)
    cv_score = np.asarray(scores).mean()
    return cv_score


def parmap_wrap(train_index, test_index, X, y, RF, scoring=multilabel_prec, scoring_arg1=1, scoring_arg2=5):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    RF.fit(X_train, y_train)
    y_hat = RF.predict_proba(X_test)
    score = scoring(y_test, y_hat, what_to_predict=scoring_arg1, nvals=scoring_arg2)
    return score
