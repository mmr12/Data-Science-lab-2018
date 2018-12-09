from prediction.test import *

MODEL = 'doc2vec'  # doc2vec, tfidf, word2vec,
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
