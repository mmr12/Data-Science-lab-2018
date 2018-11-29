import pickle
import random

import numpy as np
from joblib import load


def predict():
    print("Please paste the path of the .txt file containing the question you would like to answer")
    print("Otherwise, if you would like us to pick a random question, just leave the input blank")
    path = input("Path: ")

    # load data
    with open("embedding/models/doc_data/all_docs_sep.pkl", "rb") as fp:
        all_docs_sep = pickle.load(fp)

    if path == "":
        ticket_ques = all_docs_sep['ticket_ques']

        # pick one Q at random
        Q = ticket_ques[random.sample(range(len(ticket_ques)), 1)[0]]
    else:
        # read file
        file = open(path, "r")
        Q = ""
        for line in file:
            Q += line

    # load embeddings
    TFiDF = load('embedding/models/TF-IFD-ticket-ques.joblib')
    # embed new Q
    Q_vec = TFiDF.transform([Q])
    # print(Q_vec.shape) # debug
    # load classifier
    RF = load('classifier/models/RF_TFiDF.joblib')
    pred = RF.predict_proba(Q_vec)
    prob = np.max(pred)
    nFAQ = np.argmax(pred)

    print("The FAQ associated with \n\n", Q, "\nwith probability ", round(prob, 2), " is\n")
    print("FAQ question:\t\t", all_docs_sep['faq_ques'][nFAQ], "\n\nFAQ ans:\t\t", all_docs_sep['faq_ans'][nFAQ])


if __name__ == "__main__":
    predict()
