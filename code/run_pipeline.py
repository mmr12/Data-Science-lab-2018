from prediction.test import *

MODEL = 'tfidf'  # doc2vec, tfidf, word2vec,
SIM_THRESH = 0.2
DATA_PREFIX = '../data/12-08-'
CLASS_TYPE = 'cv'  # 'cv', 'val', 'test'
SCORE = 99  # 0: F1, 1:precision, 2:recall, 99: precision, recall, F1-score
NFAQS = 5  # n FAQs to be considered for the top answer
if __name__== "__main__" :
    # embedding(model=MODEL, data_prefix=DATA_PREFIX)
    # similarity(model=MODEL, thresh=SIM_THRESH)
    # classifier(model=MODEL, scoring=SCORE, n_FAQs=NFAQS)

    # predict()
    test(MODEL, data_prefix=DATA_PREFIX)

    # TODO: test(doc2vec) outputs scores (nan, 0.0, nan)
    # TODO: link test and word2vec
    # TODO: (future) link tfidf_w2v with test
    # TODO: mirror test with validate
