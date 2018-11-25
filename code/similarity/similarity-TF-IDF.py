import pickle

import numpy as np
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity


def similarity():
    # load model
    TFiDF = load('../embedding/models/TF-IFD-ans.joblib')

    # load data
    with open("../embedding/models/doc_data/all_docs_sep.pkl", "rb") as fp:
        all_docs_sep = pickle.load(fp)
    with open("../embedding/models/doc_data/id_dict_sep.pkl", "rb") as fp:
        id_dict_sep = pickle.load(fp)

    n_faq_ans = len(id_dict_sep['faq_ans'])

    # make matrix
    matrix = TFiDF.transform(all_docs_sep['ans'])
    sim_matrix = cosine_similarity(matrix[:n_faq_ans,:], matrix[n_faq_ans:,:])

    # mapping
    FAQ_per_ticket = np.argmax(sim_matrix, axis=0)
    strength_FAQ_ticket = np.max(sim_matrix, axis=0)

    # assign weak mappings to -1
    thres = 0.2
    FAQ_per_ticket[strength_FAQ_ticket < thres] = -1

    # some stats
    n_unique = len(np.unique(FAQ_per_ticket))
    n_nonassigned = np.shape(FAQ_per_ticket[strength_FAQ_ticket < thres])[0]
    n_tickets = len(FAQ_per_ticket)

    output = {
        'classes': n_tickets,
        'mapping': FAQ_per_ticket
    }

    print(n_unique, 'classes, with ', round(n_nonassigned / n_tickets, 2), '% non assigned tickets')
    with open("mappings/ticket_faq_map_TF-IDF_cosine.pkl", "wb") as fp:
        pickle.dump(output, fp)

if __name__== "__main__":
    similarity()
