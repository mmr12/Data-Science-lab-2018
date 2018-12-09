import pickle

from gensim.models import Word2Vec
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from .utils import *

def doc_emb_one(name, id_dict, all_docs_prepro, model):
    mean_ans = np.empty((len(id_dict[name]), 128), dtype=float)
    for j in id_dict[name]:
        sentence = all_docs_prepro[j]
        words = np.empty((len(sentence), 128), dtype=float)
        for i in range(len(sentence)):
            words[i] = model[sentence[i]]
        mean_ans[j - id_dict[name][0]] = np.apply_along_axis(np.mean, 0, words)
    return mean_ans


def classification(mean_ticket_ques, mapping):
    # RANDOM FOREST CLASSIFIER
    print('RANDOM FOREST CLASSIFIER')
    print('Running CV on Classifier...')
    classifier_CV = RandomForestClassifier()
    cv_score = cross_val_proba_score(classifier_CV, mean_ticket_ques, mapping,
                                     scoring=multilabel_prec, scoring_arg1=1, scoring_arg2=5, n_splits=5)
    #scores = cross_val_score(classifier_CV, mean_ticket_ques, mapping, cv=5)
    #cv_score = scores.mean()
    print('Training Classifier...')
    classifier = RandomForestClassifier()
    classifier.fit(X=mean_ticket_ques, y=mapping)
    y_pred_proba = classifier.predict_proba(mean_ticket_ques)
    dump(classifier, 'classifier/models/RF_word2vec.joblib')
    train_score = multilabel_prec(y=mapping, y_pred_proba=y_pred_proba, what_to_predict=1, nvals=5)
    #train_score = classifier.score(X=mean_ticket_ques, y=mapping)
    print('Training Score: {0} \nCross Val Score: {1}'.format(train_score, cv_score))

    '''
    print('GRADIENT BOOSTING CLASSIEIR')
    print('Running CV on Classifier...')
    Bclassifier_CV = GradientBoostingClassifier()
    scores = cross_val_score(Bclassifier_CV, mean_ticket_ques, mapping, cv=5)
    cv_score = scores.mean()
    print('Training Classifier...')
    Bclassifier = GradientBoostingClassifier()
    Bclassifier.fit(X=mean_ticket_ques, y=mapping)
    # dump(classifier, 'classifier/models/RF_word2vec.joblib')
    train_score = Bclassifier.score(X=mean_ticket_ques, y=mapping)
    print('Training Score: {0} \nCross Val Score: {1}'.format(train_score, cv_score))
    '''

def word_embedding(all_docs_prepro, id_dict):

    # Load the Word2Vec model
    model_path = 'embedding/models/word2vec_ticket_ques.model'
    model = Word2Vec.load(model_path)

    with open('similarity/mappings/ticket_faq_map_word2vec.pkl', 'rb') as fp:
        Classes = pickle.load(fp)

    mapping = Classes['mapping']

    ticket_question_embeddings = doc_emb_one('ticket_ques_prepro',id_dict, all_docs_prepro, model)

    classification(ticket_question_embeddings, mapping)