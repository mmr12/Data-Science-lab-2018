import pickle

from gensim.models import Word2Vec
from joblib import dump
from sklearn.ensemble import RandomForestClassifier

from .utils import *


def word_embedding(ticket_ques_prepro):

    # Load the Word2Vec model
    model_path = 'embedding/models/word2vec_ticket_ques.model'
    model = Word2Vec.load(model_path)

    with open('similarity/mappings/ticket_faq_map_word2vec.pkl', 'rb') as fp:
        Classes = pickle.load(fp)

    mapping = Classes['mapping']

    def doc_emb(dat):
        mean_ans = np.empty((len(dat), 128), dtype=float)
        for j in range(len(dat)):
            sentence = dat[j]
            words = np.empty((len(sentence), 128), dtype=float)
            for i in range(len(sentence)):
                words[i] = model[sentence[i]]
            mean_ans[j] = np.apply_along_axis(np.mean, 0, words)
        return mean_ans

    ticket_question_embeddings = doc_emb(ticket_ques_prepro)

    print('Running CV on Classifier...')
    classifier_CV = RandomForestClassifier()
    cv_score = cross_val_proba_score(classifier_CV, ticket_question_embeddings, mapping,
                                     scoring=multilabel_prec, scoring_arg1=1, scoring_arg2=5, n_splits=5)
    # scores = cross_val_score(classifier_CV, ticket_question_embeddings, mapping, cv=5)
    # cv_score = scores.mean()

    print('Training Classifier...')
    classifier = RandomForestClassifier()
    classifier.fit(X=ticket_question_embeddings, y=mapping)
    dump(classifier, 'classifier/models/RF_word2vec.joblib')
    y_pred_proba = classifier.predict_proba(ticket_question_embeddings)
    train_score = multilabel_prec(y=mapping, y_pred_proba=y_pred_proba, what_to_predict=1, nvals=5)
    #train_score = classifier.score(X=ticket_question_embeddings, y=mapping)

    print('Training Score: {0} \n Cross Val Score: {1}'.format(train_score, cv_score))
    print('###############')