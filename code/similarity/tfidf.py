import pickle
import numpy as np
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def tfidf(faq_ans, ticket_ans):
    print('Loading TF-IDF Model...')
    TFiDF = load('embedding/models/TF-IFD-ans.joblib')

    # make matrix
    FAQ_matrix = TFiDF.transform(faq_ans)
    ticket_matrix = TFiDF.transform(ticket_ans)
    print('Computing TF-IDF Similarities...')
    sim_matrix = cosine_similarity(FAQ_matrix, ticket_matrix)

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
    # How many tickets each FAQ is assigned
    counts_per_faq = pd.Series(FAQ_per_ticket).value_counts()
    print(counts_per_faq)

    output = {
        'classes': n_tickets,
        'mapping': FAQ_per_ticket
    }
    print(n_unique, 'classes, with ', round(n_nonassigned / n_tickets, 2), '% non assigned tickets')

    with open("similarity/mappings/ticket_faq_map_TF-IDF.pkl", "wb") as fp:
        pickle.dump(output, fp)