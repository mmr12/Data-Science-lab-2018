import pickle

from gensim.models import Word2Vec, Doc2Vec
from joblib import load

from .utils import *


def test(model, data_prefix='../data/12-08-'):
    with open(data_prefix + 'val-test.pkl', "rb") as fp:
        test_dic = pickle.load(fp)

    y = test_dic["y_test"]

    if model == 'tfidf':
        x = test_dic["x_test"]
        TFiDF = load('embedding/models/TF-IFD-ticket-ques.joblib')
        X_test = TFiDF.transform(x)
        classifier = load('classifier/models/RF_TFiDF.joblib')

    elif model == 'word2vec':
        W2V(y)

    elif model == 'doc2vec':
        # load data
        with open("embedding/models/doc_data/ticket_test.txt", "rb") as fp:
            test_prepo = pickle.load(fp)
        # load model
        model = Doc2Vec.load('embedding/models/' + 'doc2vec_ticket_ques.model')
        # embed data
        X_test = np.array([model.infer_vector(test_prepo[i]) for i in range(len(test_prepo))])
        classifier = load('classifier/models/RF_doc2vec.joblib')

    else:
        print('Model {} not found'.format(model))
        return 0

    y_hat = classifier.predict_proba(X_test)
    scores = multilabel_prec(y, y_hat, what_to_predict=99, nvals=5)
    print("precision, recall, F1-score", scores)
    return scores


def W2V(y):
    with open("embedding/models/doc_data/ticket_test.txt", "rb") as fp:
        test_prepo = pickle.load(fp)
    # Load the Word2Vec model
    model_path = 'embedding/models/word2vec_ticket_ques.model'
    model = Word2Vec.load(model_path)
    X_test = doc_emb(test_prepo, model)
    classifier = load('classifier/models/RF_word2vec.joblib')
    y_hat = classifier.predict_proba(X_test)
    scores = multilabel_prec(y, y_hat, what_to_predict=99, nvals=5)
    return scores


# word2vec support function
def doc_emb(dat, model):
    mean_ans = np.empty((len(dat), 128), dtype=float)
    for j in range(len(dat)):
        sentence = dat[j]
        words = np.empty((len(sentence), 128), dtype=float)
        for i in range(len(sentence)):
            words[i] = model[sentence[i]]
        mean_ans[j] = np.apply_along_axis(np.mean, 0, words)
    return mean_ans


if __name__ == "__main__":
    test()
