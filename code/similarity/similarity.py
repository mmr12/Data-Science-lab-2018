from .tfidf import *
from .word2vec import *
from .doc2vec import *
from .tfidf_w2v import *
import pickle
import numpy as np


def similarity():

    print('Loading Document Data...')
    with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
        id_dict = pickle.load(fp)
    # Unpickle the document data
    with open("embedding/models/doc_data/all_ans_prepro.txt", "rb") as fp:
        all_ans_prepro = pickle.load(fp)
    with open("embedding/models/doc_data/all_docs_sep.pkl", "rb") as fp:
        all_docs_sep = pickle.load(fp)

    ticket_ans_ids = np.array(id_dict['ticket_ans'])
    all_faq_ans = id_dict['faq_ans']

    faq_ans = all_docs_sep['faq_ans']
    ticket_ans = all_docs_sep['ticket_ans']

    #########################################

    #word2vec
    #word_embedding(all_ans_prepro, faq_ans)

    #########################################

    #tfidf + word2vec
    tfidf_w2v(all_ans_prepro, faq_ans)

    #########################################

    #doc2vec
    #document_embedding(all_faq_ans, ticket_ans_ids)

    #########################################

    #TFIDF
    #tfidf(faq_ans, ticket_ans)

if __name__== "__main__":
    similarity()
