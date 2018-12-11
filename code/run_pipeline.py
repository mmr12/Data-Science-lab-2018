from classifier.classifier import classifier
from embedding.embedding import embedding
from prediction.test import *
from similarity.similarity import similarity

MODEL = 'tfidf_w2v'
# thresholds
if MODEL == "tfidf":
    thresh = 0.2
elif MODEL == "word2vec":
    thresh = 0.96
elif MODEL == "doc2vec":
    thresh = 0.98
elif MODEL == "tfidf_w2v":
    thresh = 0
else:
    print("Please select valid model")
SIM_THRESH = thresh
DATA_PREFIX = '../data/12-08-'

SCORE = 99  # 0: F1, 1:precision, 2:recall, 99: precision, recall, F1-score
NFAQS = 5  # n FAQs to be considered for the top answer
if __name__== "__main__" :
    embedding(model=MODEL, data_prefix=DATA_PREFIX)
    similarity(model=MODEL, thresh=SIM_THRESH)
    classifier(model=MODEL, scoring=SCORE, n_FAQs=NFAQS)

    # predict()
    #test(MODEL, data_prefix=DATA_PREFIX)

    # TODO: test(doc2vec) outputs scores (nan, 0.0, nan)
    # TODO: link test and word2vec
    # TODO: (future) link tfidf_w2v with test
    # TODO: mirror test with validate
