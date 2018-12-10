from .doc2vec import *
from .tfidf import RF
from .word2vec import *
from .tfidf_w2v import *


# Unpickle the document data, the ticket to FAQ map
def classifier(model, scoring=1, n_FAQs=5):
    # get embeddings
    if model == 'tfidf':
        X_train, y_train = RF()

    elif model == 'word2vec':

        with open("embedding/models/doc_data/all_docs_prepro.txt", "rb") as fp:
            all_docs_prepro = pickle.load(fp)
        with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
            id_dict = pickle.load(fp)
        X_train, y_train = word_embedding(all_docs_prepro, id_dict)

    elif model == 'tfidf_w2v':

        with open("embedding/models/doc_data/all_docs_prepro.txt", "rb") as fp:
            all_docs_prepro = pickle.load(fp)
        with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
            id_dict = pickle.load(fp)

        tfidf_w2v(all_docs_prepro, id_dict)

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
                                     scoring=multilabel_prec,
                                     scoring_arg1=scoring,
                                     scoring_arg2=n_FAQs,
                                     n_splits=5)

    print('Cross Val Score: {0}'.format(cv_score))

    if model == 'tfidf':
        dump(classifier, 'classifier/models/RF_TFiDF.joblib')

    elif model == 'word2vec':
        dump(classifier, 'classifier/models/RF_word2vec.joblib')

    elif model == 'doc2vec':
        dump(classifier, 'classifier/models/RF_doc2vec.joblib')


# TODO: included files here and not in directory file

if __name__ == "__main__":
    classifier('tfidf_w2v')