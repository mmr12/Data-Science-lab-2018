import os
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

def tfidf_w2v(all_ans_prepro, ticket_ques_prepro, all_docs_prepro):

    '''word2vec models are run and saved from the embedding/wor2vec.py folder
        so make sure you have run word2vec model before running the tfidf_w2v.py model'''

    #tfidf ALL ANSWERS
    # checking if embedding model already exists
    exists = os.path.isfile('embedding/models/tfidf_ans.model')
    if exists:
        print('Tfidf all_ans embedding model already existing')
    # Create word embedding model
    else:
        print('Training tfidf on all answers')
        path = "embedding/models/tfidf_ans.model"
        dct = Dictionary(all_ans_prepro)  # fit dictionary
        corpus = [dct.doc2bow(line) for line in all_ans_prepro]  # convert corpus to BoW format
        model_tfidf = TfidfModel(corpus)  # fit model
        model_tfidf.save(path)


    #tfidf TICKET QUESTIONS
    exists = os.path.isfile('embedding/models/tfidf_ticket_ques.model')
    if exists:
        print('Tfidf all_ans embedding model already existing')
    else:
        print('Training tfidf on ticket questions')
        path = "embedding/models/tfidf_ticket_ques.model"
        dct = Dictionary(ticket_ques_prepro)  # fit dictionary
        corpus = [dct.doc2bow(line) for line in ticket_ques_prepro]  # convert corpus to BoW format
        model_tfidf = TfidfModel(corpus)  # fit model
        model_tfidf.save(path)


    #tfidf ALL DOCS
    exists = os.path.isfile('embedding/models/tfidf_all.model')
    if exists:
        print('Tfidf all_ans embedding model already existing')
    else:
        print('Training tfidf on all documents')
        path = "embedding/models/tfidf_all.model"
        dct = Dictionary(all_docs_prepro)  # fit dictionary
        corpus = [dct.doc2bow(line) for line in all_docs_prepro]  # convert corpus to BoW format
        model_tfidf = TfidfModel(corpus)  # fit model
        model_tfidf.save(path)