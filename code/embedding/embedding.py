import pandas as pd
from .preprocessing import *
from .tfidf import *
from .doc2vec import *
from .word2vec import *
import pickle

def embedding():

    # Read in the data
    ticket_dat = pd.read_csv('../data/12-04-ticket_dat.csv')
    faq_dat = pd.read_csv('../data/12-04-faq_dat.csv')
    # Replace the NaNs
    ticket_dat.fillna('', inplace=True)
    faq_dat.fillna('', inplace=True)

    # FAQ question
    faq_ques = list(faq_dat.question)
    n_faq_ques = len(faq_ques)
    # FAQ answer
    faq_ans = list(faq_dat.answer_title + " " + faq_dat.answer)
    n_faq_ans = len(faq_ans)
    #ticket question
    ticket_ques = list(ticket_dat.question)
    n_ticket_ques = len(ticket_ques)
    #ticket ans
    ticket_ans = list(ticket_dat.answer)
    n_ticket_ans = len(ticket_ans)

    # Model assumption: same embedding for all
    all_docs = faq_ques + faq_ans + ticket_ques + ticket_ans
    # Model assumption: different embeddings
    all_ans = faq_ans + ticket_ans

    # create a dictionary storing the cut points for the four datasets so we can re-split them after.
    # use like all_docs[id_dict['faq_ques']] to get all faq questions.
    id_dict = {
        'faq_ques': range(0, n_faq_ques),
        'faq_ans': range(n_faq_ques, n_faq_ques + n_faq_ans),
        'ticket_ques': range(n_faq_ques + n_faq_ans, n_faq_ques + n_faq_ans + n_ticket_ques),
        'ticket_ans': range(n_faq_ques + n_faq_ans + n_ticket_ques, n_faq_ques + n_faq_ans + n_ticket_ques + n_ticket_ans)
    }
    all_docs_sep = {
        'faq_ques': faq_ques,
        'faq_ans': faq_ans,
        'ticket_ques': ticket_ques,
        'ticket_ans': ticket_ans}

    # Need to save this list and id dictionary as a pickle so we can decode IDs when we test things
    with open("embedding/models/doc_data/all_docs.txt", "wb") as fp:
        pickle.dump(all_docs, fp)
    with open("embedding/models/doc_data/id_dict.txt", "wb") as fp:
        pickle.dump(id_dict, fp)
    with open("embedding/models/doc_data/all_docs_sep.pkl", "wb") as fp:
        pickle.dump(all_docs_sep, fp)

    #preprocessed data to be saved
    all_ans_prepro = preprocess_docs_fn(all_ans)
    with open("embedding/models/doc_data/all_ans_prepro.txt", "wb") as fp:
        pickle.dump(all_ans_prepro, fp)

    ticket_ques_prepro = preprocess_docs_fn(ticket_ques)
    with open("embedding/models/doc_data/ticket_ques_prepro.txt", "wb") as fp:
        pickle.dump(ticket_ques_prepro, fp)

    ############################################

    # Model assumption: word2vec
    word_embedding(all_ans_prepro, ticket_ques_prepro)
    print('Word2vec training done')

    ############################################

    # Model assumption: doc2vec
    #document_embedding(all_ans, ticket_ques)
    #print('Doc2vec training done')

    #############################################

    # Model assumption: TF-IDF
    #print('Training TF-IDF Model')
    #tfidf(all_ans, ticket_ques)
    #print('Trained')

if __name__== "__main__":
    embedding()