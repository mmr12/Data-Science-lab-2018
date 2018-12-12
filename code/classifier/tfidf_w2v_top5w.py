import pickle

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from operator import itemgetter

from .utils import *


def top5_average(dat, corpus, dct, model_w2v, model_tfidf, id_dict, all_docs_prepro):
    if dat == 'faq_ans':
        ind = id_dict['faq_ans'][0]
        leng = len(id_dict['faq_ans'])
        dat = all_docs_prepro[ind:leng]
    elif dat == 'ticket_ans':
        ind = id_dict['ticket_ans'][0]
        leng = len(id_dict['ticket_ans'])
        dat = all_docs_prepro[ind:leng]
    else:
        ind = id_dict['ticket_ques'][0]
        leng = len(id_dict['ticket_ques'])
        dat = all_docs_prepro[ind:leng]
    mean_ans = np.empty((leng, 128), dtype=float)
    for i in range(leng):
        vector = model_tfidf[corpus[ind]]
        vector_s = sorted(vector, key=itemgetter(1), reverse=True)
        top5 = vector_s[:5]
        top5 = np.asarray(top5, dtype=float)
        words = np.empty((len(top5), 128), dtype=float)
        for j in range(len(top5)):
            words[j] = model_w2v[dct[int(top5[j,0])]]
        mean_ans[i] = np.average(words, 0, weights=top5[:,1])
        ind += 1
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
    dump(classifier, 'classifier/models/RF_tfidf_word2vec_5w.joblib')
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

def tfidf_w2v_top5w(all_docs_prepro, id_dict):
    with open('../code/similarity/mappings/map_w2v_tfidf_5w.pkl', 'rb') as fp:
        Classes = pickle.load(fp)
    mapping = Classes['mapping']

    print('Loading Word2vec model')
    model_path = 'embedding/models/word2vec_all.model'
    model_w2v = Word2Vec.load(model_path)

    print('Loading Tfidf model')
    model_path = 'embedding/models/tfidf_all.model'
    model_tfidf = TfidfModel.load(model_path)

    dct = Dictionary(all_docs_prepro)
    corpus = [dct.doc2bow(line) for line in all_docs_prepro]

    mean_ticket_ques = top5_average('ticket_ques', corpus=corpus, dct=dct, model_w2v=model_w2v,
                                    model_tfidf=model_tfidf, id_dict=id_dict, all_docs_prepro=all_docs_prepro)

    return (mean_ticket_ques, mapping)
    # classification(mean_ticket_ques, mapping)
