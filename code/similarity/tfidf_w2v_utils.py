import numpy as np
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle


# AVERAGE 5 MOST IMPORTANT WORDS
def top5(all_ans_prepro, corpus, dct, model_w2v, model_tfidf):
    print(len(all_ans_prepro))
    mean_ans = np.empty((len(all_ans_prepro), 128), dtype=float)
    for i in range(len(corpus)):
        vector = model_tfidf[corpus[i]]
        vector_s = sorted(vector, key=itemgetter(1), reverse=True)
        top5 = vector_s[:5]
        top5 = np.asarray(top5, dtype=int)[:, 0]
        words = np.empty((len(top5), 128), dtype=float)
        for j in range(len(top5)):
            words[j] = model_w2v[dct[top5[j]]]
        mean_ans[i] = np.apply_along_axis(np.mean, 0, words)
    return mean_ans

# WEIGHTED AVERAGE OVER 5 MOST IMPORTANT WORDS
def top5_average(all_ans_prepro, corpus, dct, model_w2v, model_tfidf):
    mean_ans = np.empty((len(all_ans_prepro), 128), dtype=float)
    for i in range(len(corpus)):
        vector = model_tfidf[corpus[i]]
        vector_s = sorted(vector, key=itemgetter(1), reverse=True)
        top5 = vector_s[:5]
        top5 = np.asarray(top5, dtype=float)
        words = np.empty((len(top5), 128), dtype=float)
        for j in range(len(top5)):
            words[j] = model_w2v[dct[int(top5[j, 0])]]
        mean_ans[i] = np.average(words, 0, weights=top5[:, 1])
    return mean_ans

# WEIGHTED AVERAGE OVER ALL VECTORS
def all_average(all_ans_prepro, corpus, dct, model_w2v, model_tfidf):
    mean_ans = np.empty((len(all_ans_prepro), 128), dtype=float)
    for i in range(len(corpus)):
        vector = np.asarray(model_tfidf[corpus[i]], dtype=float)
        words = np.empty((len(vector), 128), dtype=float)
        for j in range(len(vector)):
            words[j] = model_w2v[dct[int(vector[j, 0])]]
        mean_ans[i] = np.average(words, 0, weights=vector[:, 1])
    return mean_ans

def compute_sim(mean_ticket_ans, mean_faq_ans):
    print('Computing word2vec similarity')

    # create matrix with cosine distances from all ticket ans to all faq ans
    sim_matrix = cosine_similarity(mean_faq_ans, mean_ticket_ans)

    # most similar faq - ticket mapping
    FAQ_per_ticket = np.argmax(sim_matrix, axis=0)
    strength_FAQ_ticket = np.max(sim_matrix, axis=0)

    # small similarities are set to a separate class
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

    with open("similarity/mappings/ticket_faq_map_word2vec.pkl", "wb") as fp:
        pickle.dump(output, fp)
