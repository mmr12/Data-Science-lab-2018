import pickle

from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

from .utils import *


def RF():
    # load classes
    with open("similarity/mappings/ticket_faq_map_TF-IDF.pkl", "rb") as fp:
        Classes = pickle.load(fp)

    # nclasses = Classes['classes']
    mapping = Classes['mapping']

    # load embeddings
    TFiDF = load('embedding/models/TF-IFD-ticket-ques.joblib')

    # load data
    with open("embedding/models/doc_data/all_docs_sep.pkl", "rb") as fp:
        all_docs_sep = pickle.load(fp)

    ticket_ques = all_docs_sep['ticket_ques']

    # extract features
    matrix = TFiDF.transform(ticket_ques)

    # debug
    # print(matrix.shape, mapping.shape)

    RF = RandomForestClassifier()
    print('Running CV on Classifier...')
    cv_score = cross_val_proba_score(RF, matrix, mapping, scoring=multilabel_prec, scoring_arg1=99, scoring_arg2=5,
                                     n_splits=5)
    # scores = cross_val_score(RF, matrix, mapping, cv=5)
    # cv_score = scores.mean()

    print('CV score on RF ', np.around(cv_score, 4))

    # save model
    RF_final = RandomForestClassifier()
    RF_final.fit(matrix, mapping)
    dump(RF_final, 'classifier/models/RF_TFiDF.joblib')
    # dump(RF_final.predict(matrix), '/Users/margheritarosnati/Desktop/temp.joblib') #debug


def nn_classifier():
    # load classes
    with open("similarity/mappings/ticket_faq_map_TF-IDF_cosine.pkl", "rb") as fp:
        Classes = pickle.load(fp)

    # nclasses = Classes['classes']
    FAQ_per_ticket = Classes['mapping']

    # load embeddings
    TFiDF = load('embedding/models/TF-IFD-ticket-ques.joblib')

    # load data
    with open("embedding/models/doc_data/all_docs_sep.pkl", "rb") as fp:
        all_docs_sep = pickle.load(fp)

    ticket_ques = all_docs_sep['ticket_ques']

    # extract features
    matrix = TFiDF.transform(ticket_ques)

    # calculate similarities
    sim_matrix = cosine_similarity(matrix)
    for i in range(len(sim_matrix)):
        sim_matrix[i, i] = 0

    # Cross validation set up:
    index = np.arange(0, len(ticket_ques))
    np.random.shuffle(index)
    splits = np.array_split(index, 5)
    scores = np.zeros(5, )

    # Cross validation:
    for i in range(5):
        temp = sim_matrix[list(set(index) - set(splits[i])), :]
        mapping = np.argmax(temp[:, splits[i]], axis=0)
        scores[i] = np.sum(FAQ_per_ticket[mapping] == FAQ_per_ticket[splits[i]]) / len(splits[i])
    cv_score = np.mean(scores)
    print('CV score on nearest neighbour ', round(cv_score, 4))


if __name__ == "__main__":
    RF()
