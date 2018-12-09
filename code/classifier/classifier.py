from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from .doc2vec import *
from .tfidf import RF
from .word2vec import *


# Unpickle the document data, the ticket to FAQ map
def classifier(model):
    # get embeddings
    if model == 'tfidf':
        X_train, y_train = RF()

    elif model == 'word2vec':
        with open("embedding/models/doc_data/ticket_ques_prepro.txt", "rb") as fp:
            ticket_ques_prepro = pickle.load(fp)

        X_train, y_train = word_embedding(ticket_ques_prepro)

    elif model == 'doc2vec':
        with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
            id_dict = pickle.load(fp)
        X_train, y_train = document_embedding(id_dict)

    else:
        print('Model {} not found'.format(model))
        return 0

    # train
    print('Training Classifier...')
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    print('Running CV on Classifier...')
    classifier_CV = RandomForestClassifier()
    cv_score = cross_val_proba_score(classifier_CV, X_train, y_train,
                                     scoring=multilabel_prec, scoring_arg1=1, scoring_arg2=5, n_splits=5)

    print('Cross Val Score: {0}'.format(cv_score))

    if model == 'tfidf':
        dump(classifier, 'classifier/models/RF_TFiDF.joblib')

    elif model == 'word2vec':
        dump(classifier, 'classifier/models/RF_word2vec.joblib')

    elif model == 'doc2vec':
        dump(classifier, 'classifier/models/RF_doc2vec.joblib')


# TODO: included files here and not in directory file

if __name__ == "__main__":
    classifier()