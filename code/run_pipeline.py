from embedding.embedding import embedding
from classifier.classifier import classifier
from similarity.similarity import similarity
from prediction.test import *
import numpy as np

MODEL = 'tfidf'
# thresholds
if MODEL == "tfidf":
    thresh = 0.2  # 75% quantile 0.24
elif MODEL == "word2vec":
    thresh = 0.96  #75% quantile 0.96
elif MODEL == "doc2vec":
    thresh = 0.98  #75% quantile 0.98
elif MODEL == "tfidf_w2v":
    thresh = 0.96  # 75% quantile 0.96
elif MODEL == "tfidf_w2v_top5a":
    thresh = .965  # 75% quantile 0.965
elif MODEL == "tfidf_w2v_top5w":
    thresh = .965  # 75% quantile 0.965
else:
    print("Please select valid model")
SIM_THRESH = thresh
DATA_PREFIX = '../data/12-08-'
PRECLASSIFIER = 1
SCORE = 99  # 0: F1, 1:precision, 2:recall, 99: precision, recall, F1-score
NFAQS = 3  # n FAQs to be considered for the top answer
if __name__== "__main__" :
    # embedding(model=MODEL, data_prefix=DATA_PREFIX)
    # similarity(model=MODEL, thresh=SIM_THRESH)
    # classifier(model=MODEL, scoring=SCORE, n_FAQs=NFAQS, pre=PRECLASSIFIER)

    # predict()
    scores = np.zeros((5, 3))
    MODEL = "tfidf"
    classifier(model=MODEL, scoring=SCORE, n_FAQs=NFAQS, pre=PRECLASSIFIER)
    scores[0, :] = test(MODEL, data_prefix=DATA_PREFIX, scoring=SCORE, n_FAQs=NFAQS, pre=PRECLASSIFIER)

    MODEL = "doc2vec"
    classifier(model=MODEL, scoring=SCORE, n_FAQs=NFAQS, pre=PRECLASSIFIER)
    scores[1, :] = test(MODEL, data_prefix=DATA_PREFIX, scoring=SCORE, n_FAQs=NFAQS, pre=PRECLASSIFIER)

    MODEL = "word2vec"
    classifier(model=MODEL, scoring=SCORE, n_FAQs=NFAQS, pre=PRECLASSIFIER)
    scores[2, :] = test(MODEL, data_prefix=DATA_PREFIX, scoring=SCORE, n_FAQs=NFAQS, pre=PRECLASSIFIER)

    MODEL = "tfidf_w2v"
    classifier(model=MODEL, scoring=SCORE, n_FAQs=NFAQS, pre=PRECLASSIFIER)
    scores[3, :] = test(MODEL, data_prefix=DATA_PREFIX, scoring=SCORE, n_FAQs=NFAQS, pre=PRECLASSIFIER)

    print(scores)
    # TODO: test(doc2vec) outputs scores (nan, 0.0, nan)
    # TODO: link test and word2vec
    # TODO: (future) link tfidf_w2v with test
    # TODO: mirror test with validate
