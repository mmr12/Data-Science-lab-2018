from gensim.models import Word2Vec
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def word_embedding(all_ans_prepro, faq_ans):
    print('Loading Word2vec model')
    model_path = 'embedding/models/word2vec_ans.model'
    model = Word2Vec.load(model_path)

    def doc_emb(dat):
        mean_ans = np.empty((len(dat), 128), dtype=float)
        for j in range(len(dat)):
            sentence = dat[j]
            words = np.empty((len(sentence), 128), dtype=float)
            for i in range(len(sentence)):
                words[i] = model[sentence[i]]
            mean_ans[j] = np.apply_along_axis(np.mean, 0, words)
        return mean_ans

    print('Computing word2vec similarity')
    #create doc vector for tickets answers i.e. average over each ticket ans the word2vec vector for each word
    mean_ticket_ans = doc_emb(all_ans_prepro[len(faq_ans):len(all_ans_prepro)])
    #create doc vector for faq ans i.e. average over each faq ans the word2vec vector for each word
    mean_faq_ans = doc_emb(all_ans_prepro[0:len(faq_ans)])

    #create matrix with cosine distances from all ticket ans to all faq ans
    sim_matrix = cosine_similarity(mean_faq_ans, mean_ticket_ans)

    #most similar faq - ticket mapping
    FAQ_per_ticket = np.argmax(sim_matrix, axis=0)
    strength_FAQ_ticket = np.max(sim_matrix, axis=0)

    #small similarities are set to a separate class
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