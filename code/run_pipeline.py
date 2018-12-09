from embedding.embedding import embedding
from similarity.similarity import similarity
from classifier.classifier import classifier
from prediction.predict import predict

MODEL = 'tfidf'
SIM_THRESH = 0.2
DATA_PREFIX = '../data/12-04-'

if __name__== "__main__" :

    embedding(model=MODEL, data_prefix=DATA_PREFIX)
    similarity(model=MODEL, thresh=SIM_THRESH)
    classifier(model=MODEL)
    #predict() # TODO: fix (/replace) this