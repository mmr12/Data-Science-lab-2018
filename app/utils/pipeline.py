from joblib import load, dump
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
import numpy as np


def load_classifier(model='tfidf'):

    # Load and return the classifier
    if model == 'tfidf':
        print('Loading TFiDF Classifier')
        classifier = load('code/classifier/models/RF_TFiDF.joblib')
    elif model == 'word2vec':
        print('Loading word2vec Classifier')
        classifier = load('code/classifier/models/RF_word2vec.joblib')
    elif model == 'doc2vec':
        print('Loading doc2vec Classifier')
        classifier = load('code/classifier/models/RF_doc2vec.joblib')
    else:
        print('No Model {}'.format(model))
        classifier = None

    return classifier


def load_embedder(model='tfidf'):

    # Load and return the classifier
    if model == 'tfidf':
        print('Loading TFiDF Embedder')
        embedder = load('code/embedding/models/TF-IFD-ticket-ques.joblib')
    elif model == 'word2vec':
        print('Loading word2vec Embedding')
        embedder = Word2Vec.load('code/embedding/models/word2vec_ticket_ques.model')
    elif model == 'doc2vec':
        print('Loading doc2vec Embedding')
        embedder = Doc2Vec.load('code/embedding/models/doc2vec_ticket_ques.model')
    else:
        print('No Model {}'.format(model))
        embedder = None

    return embedder

def predict(text, embedder, classifier, model = 'tfidf'):

    if model =='tfidf':

        embedding = embedder.transform([text])
        probs = classifier.predict_proba(embedding)
        labels = classifier.classes_

        return probs, labels

    # TODO: other model types
    else:
        print('Model {} not found'.format(model))
        return None



