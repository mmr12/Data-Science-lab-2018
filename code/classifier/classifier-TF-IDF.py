import pickle

from joblib import load, dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def classifier():
    # load classes
    with open("similarity/mappings/ticket_faq_map_TF-IDF_cosine.pkl", "rb") as fp:
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
    scores = cross_val_score(RF, matrix, mapping, cv=5)
    cv_score = scores.mean()

    print('CV score on RF ', round(cv_score, 2))

    # save model
    RF_final = RandomForestClassifier()
    RF_final.fit(matrix, mapping)
    dump(RF_final, 'classifier/models/RF_TFiDF.joblib')


if __name__ == "__main__":
    classifier()
