from embedding.embedding import embedding
from classifier.classifier import classifier
from similarity.similarity import similarity
from prediction.test import *

MODEL = 'tfidf_w2v'
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
    thresh = .90 # what is it its 75% quantile??
elif MODEL == "tfidf_w2v_top5w":
    thresh = .90 # what is it its 75% quantile??
else:
    print("Please select valid model")
SIM_THRESH = thresh
DATA_PREFIX = '../data/12-08-'

SCORE = 99  # 0: F1, 1:precision, 2:recall, 99: precision, recall, F1-score
NFAQS = 3  # n FAQs to be considered for the top answer
if __name__== "__main__" :
    # embedding(model=MODEL, data_prefix=DATA_PREFIX)
    # similarity(model=MODEL, thresh=SIM_THRESH)
    classifier(model=MODEL, scoring=SCORE, n_FAQs=NFAQS)

    # predict()
    #test(MODEL, data_prefix=DATA_PREFIX, scoring=SCORE, n_FAQs=NFAQS)

    # TODO: test(doc2vec) outputs scores (nan, 0.0, nan)
    # TODO: link test and word2vec
    # TODO: (future) link tfidf_w2v with test
    # TODO: mirror test with validate
