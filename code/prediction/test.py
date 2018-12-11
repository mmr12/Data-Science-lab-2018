import pickle

from gensim.models import Word2Vec, Doc2Vec, TfidfModel
from gensim.corpora import Dictionary
from joblib import load

from .utils import *


def test(model, data_prefix='../data/12-08-', scoring=99, n_FAQs=6):
    with open(data_prefix + 'val-test.pkl', "rb") as fp:
        test_dic = pickle.load(fp)

    y = test_dic["y_test"]

    if model == 'tfidf':
        x = test_dic["x_test"]
        TFiDF = load('embedding/models/TF-IFD-ticket-ques.joblib')
        X_test = TFiDF.transform(x)
        classifier = load('classifier/models/RF_TFiDF.joblib')

    elif model == 'word2vec':
        # load data
        with open("embedding/models/doc_data/ticket_test.txt", "rb") as fp:
            test_prepo = pickle.load(fp)
        model_path = 'embedding/models/word2vec_all.model'
        model = Word2Vec.load(model_path)
        X_test = doc_emb_new_one(test_prepo, model)
        classifier = load('classifier/models/RF_word2vec.joblib')

    elif model == 'doc2vec':
        # load data
        with open("embedding/models/doc_data/ticket_test.txt", "rb") as fp:
            test_prepo = pickle.load(fp)
        # load model
        model = Doc2Vec.load('embedding/models/' + 'doc2vec_ticket_ques.model')
        # embed data
        X_test = np.array([model.infer_vector(test_prepo[i]) for i in range(len(test_prepo))])
        classifier = load('classifier/models/RF_doc2vec.joblib')

    elif model == 'tfidf_w2v':
        # load data
        with open("embedding/models/doc_data/ticket_test.txt", "rb") as fp:
            test_prepo = pickle.load(fp)
        # some embedding processing
        dct = Dictionary(test_prepo)
        corpus = [dct.doc2bow(line) for line in test_prepo]

        # load models
        print('Loading Word2vec model')
        model_path = 'embedding/models/word2vec_all.model'
        model_w2v = Word2Vec.load(model_path)

        print('Loading Tfidf model')
        model_path = 'embedding/models/tfidf_all.model'
        model_tfidf = TfidfModel.load(model_path)
        X_test = all_avg(ind_start=0,
                         ind_end=len(test_prepo),
                         corpus=corpus,
                         dct=dct,
                         model_w2v=model_w2v,
                         model_tfidf=model_tfidf)
        classifier = load('classifier/models/RF_tfidf_w2v.joblib')

    else:
        print('Model {} not found'.format(model))
        return 0

    y_hat = classifier.predict_proba(X_test)
    scores = multilabel_prec(y, y_hat, what_to_predict=scoring, nvals=n_FAQs)
    print("precision, recall, F1-score", scores)
    y_hat = np.zeros(y_hat.shape)
    scores = multilabel_prec(y, y_hat, what_to_predict=scoring, nvals=n_FAQs)
    print("When predicting all -1, precision, recall, F1-score", scores)
    return scores



# word2vec support function
def doc_emb_new_one(doc_prepo, model):
    length = len(doc_prepo)
    mean_ans = np.empty((length, 128), dtype=float)
    # extract vocabulary
    word_vectors = model.wv
    for j in range(length):
        sentence = doc_prepo[j]
        # let's go a little old school
        words = np.empty(128, dtype=float)
        counter = 0
        for i in range(len(sentence)):
            if sentence[i] in word_vectors.vocab:
                words += model[sentence[i]]
                counter += 1
        mean_ans[j] = words / counter
    return mean_ans


if __name__ == "__main__":
    test()
