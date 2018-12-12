from .doc2vec import *
from .tfidf import RF
from .tfidf_w2v import *
from .tfidf_w2v_top5a import *
from .tfidf_w2v_top5w import *
from .word2vec import *


# Unpickle the document data, the ticket to FAQ map
def classifier(model, scoring=1, n_FAQs=5, pre=0):
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

        X_train, y_train = tfidf_w2v(all_docs_prepro, id_dict)

    elif model == 'tfidf_w2v_top5a':
        with open("embedding/models/doc_data/all_docs_prepro.txt", "rb") as fp:
            all_docs_prepro = pickle.load(fp)
        with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
            id_dict = pickle.load(fp)

        X_train, y_train = tfidf_w2v_top5a(all_docs_prepro, id_dict)

    elif model == 'tfidf_w2v_top5w':
        with open("embedding/models/doc_data/all_docs_prepro.txt", "rb") as fp:
            all_docs_prepro = pickle.load(fp)
        with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
            id_dict = pickle.load(fp)

        X_train, y_train = tfidf_w2v_top5w(all_docs_prepro, id_dict)

    elif model == 'doc2vec':
        with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
            id_dict = pickle.load(fp)
        X_train, y_train = document_embedding(id_dict)

    else:
        print('Model {} not found'.format(model))
        return 0

    # shapp we train a preclassiifer
    if pre == 1:
        # train pre-classifier
        print('Training pre-classifier...')
        y_pretrain = np.copy(y_train)
        y_pretrain[y_pretrain > -1] = 0
        preclassifier = RandomForestClassifier(n_estimators=100)
        preclassifier.fit(X_train, y_pretrain)
        print('Training classifier...')
        y_postrain = y_train[y_train > -1]
        X_postrain = X_train[y_train > -1]
        classifier = RandomForestClassifier(n_estimators=100)
        classifier.fit(X_postrain, y_postrain)
        print("Saving results")
        if model == 'tfidf':
            dump(preclassifier, 'classifier/models/RF_TFiDFpre.joblib')
            dump(classifier, 'classifier/models/RF_TFiDFpost.joblib')

        elif model == 'word2vec':
            dump(preclassifier, 'classifier/models/RF_word2vecpre.joblib')
            dump(classifier, 'classifier/models/RF_word2vecpost.joblib')

        elif model == 'tfidf_w2v':
            dump(preclassifier, 'classifier/models/RF_tfidf_w2vpre.joblib')
            dump(classifier, 'classifier/models/RF_tfidf_w2vpost.joblib')

        elif model == 'tfidf_w2v_top5a':
            dump(preclassifier, 'classifier/models/RF_tfidf_w2v_5apre.joblib')
            dump(classifier, 'classifier/models/RF_tfidf_w2v_5apost.joblib')

        elif model == 'tfidf_w2v_top5w':
            dump(preclassifier, 'classifier/models/RF_tfidf_w2v_5wpre.joblib')
            dump(classifier, 'classifier/models/RF_tfidf_w2v_5wpost.joblib')

        elif model == 'doc2vec':
            dump(preclassifier, 'classifier/models/RF_doc2vecpre.joblib')
            dump(classifier, 'classifier/models/RF_doc2vecpost.joblib')

    else:
        # train
        print('Training Classifier...')
        classifier = RandomForestClassifier(n_estimators=100)
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

        elif model == 'tfidf_w2v':
            dump(classifier, 'classifier/models/RF_tfidf_w2v.joblib')

        elif model == 'tfidf_w2v_top5a':
            dump(classifier, 'classifier/models/RF_tfidf_w2v_5a.joblib')

        elif model == 'tfidf_w2v_top5w':
            dump(classifier, 'classifier/models/RF_tfidf_w2v_5w.joblib')

        elif model == 'doc2vec':
            dump(classifier, 'classifier/models/RF_doc2vec.joblib')


# TODO: included files here and not in directory file

if __name__ == "__main__":
    classifier('tfidf_w2v_top5a')
