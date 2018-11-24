import pandas as pd
import pickle
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from joblib import dump, load


def similarity(mod):
    # load model
    TFiDF = load('embedding/models/TF-IFD-ans.joblib')

    # load data
    with open("embedding/models/doc_data/all_docs_sep.pkl", "wb") as fp:
        all_docs_sep = pickle.load(fp)
    with open("embedding/models/doc_data/id_dict_sep.pkl", "wb") as fp:
        id_dict_sep = pickle.load(fp)

    n_faq_ans = len(id_dict_sep['faq_ans'])

    # make matrix
    matrix = TFiDF.transform(all_docs_sep['ans'])
    sim_matrix = cosine_similarity(matrix[:n_faq_ans,:], matrix[n_faq_ans:,:])
    # TODO: how do we want to utilise this information?

if __name__== "__main__":
    similarity(mod)