from embedding.embedding import embedding
from classifier.classifier import classifier
from similarity.similarity import similarity
from prediction.test import *
import numpy as np

MODEL = 'word2vec'
# thresholds
if MODEL == "tfidf":
    thresh = 0.2  # 75% quantile 0.24
elif MODEL == "word2vec":
    thresh = 0.96  #75% quantile 0.96
elif MODEL == "doc2vec":
    thresh = 0.98  #75% quantile 0.98
elif MODEL == "tfidf_w2v":
    thresh = 0.96  # 75% quantile 0.96
else:
    print("Please select valid model")
SIM_THRESH = thresh
DATA_PREFIX = '../data/12-08-'

SCORE = 99  # 0: F1, 1:precision, 2:recall, 99: precision, recall, F1-score
NFAQS = 3  # n FAQs to be considered for the top answer
if __name__== "__main__" :
    # embedding(model=MODEL, data_prefix=DATA_PREFIX)
    # similarity(model=MODEL, thresh=SIM_THRESH)
    # classifier(model=MODEL, scoring=SCORE, n_FAQs=NFAQS)
    # test(model=MODEL, data_prefix=DATA_PREFIX, scoring=SCORE, n_FAQs=NFAQS)

    scores = np.zeros((4, 3))
    classifier(model="doc2vec", scoring=SCORE, n_FAQs=NFAQS)
    scores[0, :] = test(model="doc2vec", data_prefix=DATA_PREFIX, scoring=SCORE, n_FAQs=NFAQS)
    classifier(model="tfidf", scoring=SCORE, n_FAQs=NFAQS)
    scores[1, :] = test(model="tfidf", data_prefix=DATA_PREFIX, scoring=SCORE, n_FAQs=NFAQS)
    classifier(model="word2vec", scoring=SCORE, n_FAQs=NFAQS)
    scores[2, :] = test(model="word2vec", data_prefix=DATA_PREFIX, scoring=SCORE, n_FAQs=NFAQS)
    classifier(model="tfidf_w2v", scoring=SCORE, n_FAQs=NFAQS)
    scores[3, :] = test(model="tfidf_w2v", data_prefix=DATA_PREFIX, scoring=SCORE, n_FAQs=NFAQS)
    print(scores)
    # TODO: mirror test with validate
