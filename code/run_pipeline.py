from classifier.classifier import classifier
from embedding.embedding import embedding
from prediction.test import *
from similarity.similarity import similarity

MODEL = 'doc2vec'
SIM_THRESH = 0.2
DATA_PREFIX = '../data/12-08-'
CLASS_TYPE = 'cv'  # 'cv', 'val', 'test'

if __name__== "__main__" :

    embedding(model=MODEL, data_prefix=DATA_PREFIX)
    similarity(model=MODEL, thresh=SIM_THRESH)
    classifier(model=MODEL)

    # predict()
    # test(MODEL, data_prefix=DATA_PREFIX)