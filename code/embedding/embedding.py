import os
import pickle

import pandas as pd
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import TfidfModel
from gensim.corpora import Dictionary

from .preprocessing import *


def embedding(model, data_prefix='../data/12-04-'):

    # Read in the data
    ticket_dat = pd.read_csv(data_prefix + 'train_ticket.csv')
    faq_dat = pd.read_csv(data_prefix + 'faq_dat.csv')
    with open(data_prefix + 'val-test.pkl', "rb") as fp:
        test_dic = pickle.load(fp)

    # Replace the NaNs
    ticket_dat.fillna('', inplace=True)
    faq_dat.fillna('', inplace=True)

    # FAQ question
    faq_ques = list(faq_dat.question)
    n_faq_ques = len(faq_ques)
    # FAQ answer
    faq_ans = list(faq_dat.answer_title + " " + faq_dat.answer)
    n_faq_ans = len(faq_ans)
    # ticket question
    ticket_ques = list(ticket_dat.question)
    n_ticket_ques = len(ticket_ques)
    ticket_ids = list(ticket_dat.ticket_id)
    # ticket ans
    ticket_ans = list(ticket_dat.answer)
    n_ticket_ans = len(ticket_ans)


    # Model assumption: same embedding for all
    all_docs = faq_ques + faq_ans + ticket_ques + ticket_ans
    # Model assumption: different embeddings
    all_ans = faq_ans + ticket_ans
    # For the preclassifier
    ticket_ques_and_faqs = faq_ans + ticket_ques

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
        'ticket_ans': ticket_ans,
        'ticket_val': test_dic["x_val"],
        'ticket_test': test_dic["x_test"],
        'faq_title': faq_dat.answer_title, # Just used for app
        'faq_raw_answer': faq_dat.answer # Just used for app
    }

    # Run the preprocessing
    all_docs_prepro = preprocess_docs_fn(all_docs)
    all_ans_prepro = preprocess_docs_fn(all_ans)
    ticket_ques_prepro = preprocess_docs_fn(ticket_ques)
    val_prepo = preprocess_docs_fn(test_dic["x_val"])
    test_prepo = preprocess_docs_fn(test_dic["x_test"])

    dump_documents(all_docs, id_dict, all_docs_sep, all_ans, ticket_ques, ticket_ids, val_prepo, test_prepo,
                   all_docs_prepro)

    # Take model argument and train which ever model is selected
    if model == 'tfidf':
        tfidf(all_ans, ticket_ques_and_faqs)
    elif model == 'word2vec':
        word_embedding(all_docs_prepro)
    elif model == 'tfidf_w2v':
        tfidf_w2v(all_docs_prepro)
    elif model == 'tfidf_w2v_top5a':
        tfidf_w2v_top5a(all_docs_prepro)
    elif model == 'tfidf_w2v_top5a':
        tfidf_w2v_top5w(all_docs_prepro)
    elif model == 'doc2vec':
        document_embedding(all_ans_prepro, ticket_ques_prepro)
    else:
        print('Model {} not found'.format(model))

def tfidf(all_ans, ticket_ques_and_faqs):
    # ALL ANSWERS
    exists = os.path.isfile('embedding/models/TF-IFD-ans.joblib')
    if exists:
        print('TFiDF ans embedding model already existing')
    else:
        vectoriser = TfidfVectorizer(strip_accents='unicode', lowercase=True, analyzer='word')
        # create matrix: rows = all ans; cols = TI-IDF weighted word vector
        vectoriser.fit(all_ans)
        dump(vectoriser, 'embedding/models/TF-IFD-ans.joblib')

    # TICKET QUESTIONS
    exists = os.path.isfile('embedding/models/TF-IFD-ticket-ques.joblib')
    if exists:
        print('TFiDF ques embedding model already existing')
    else:
        vec2 = TfidfVectorizer(strip_accents='unicode', lowercase=True, analyzer='word')
        vec2.fit(ticket_ques_and_faqs)
        dump(vec2, 'embedding/models/TF-IFD-ticket-ques.joblib')

def word_embedding(all_docs_prepro):

    # checking if embedding model already exists
    exists = os.path.isfile('embedding/models/word2vec_all.model')
    if exists:
        print('Word2vec embedding model already existing')
    # Create word embedding model
    else:
        print('Training word2vec on all answers')
        word_path = "embedding/models/word2vec_all.model"
        word_tempfile = get_tmpfile(word_path)
        word_model = Word2Vec(all_docs_prepro, size=128, window=5, min_count=1, workers=4)
        word_model.save(word_path)

def tfidf_w2v(all_docs_prepro):

    #TFIDF MODEL
    exists = os.path.isfile('embedding/models/tfidf_all.model')
    if exists:
        print('Tfidf embedding model already existing')
    else:
        dct = Dictionary(all_docs_prepro)  # fit dictionary
        corpus = [dct.doc2bow(line) for line in all_docs_prepro]  # convert corpus to BoW format
        model_tfidf = TfidfModel(corpus)
        word_path = 'embedding/models/tfidf_all.model'
        model_tfidf.save(word_path)

    # WORD2VEC MODEL
    exists = os.path.isfile('embedding/models/word2vec_all.model')
    if exists:
        print('Word2vec embedding model already existing')
    else:
        print('Training word2vec on all answers')
        word_path = "embedding/models/word2vec_all.model"
        word_tempfile = get_tmpfile(word_path)
        word_model = Word2Vec(all_docs_prepro, size=128, window=5, min_count=1, workers=4)
        word_model.save(word_path)


def tfidf_w2v_top5a(all_docs_prepro):
    # TFIDF MODEL
    exists = os.path.isfile('embedding/models/tfidf_all.model')
    if exists:
        print('Tfidf embedding model already existing')
    else:
        dct = Dictionary(all_docs_prepro)  # fit dictionary
        corpus = [dct.doc2bow(line) for line in all_docs_prepro]  # convert corpus to BoW format
        model_tfidf = TfidfModel(corpus)
        word_path = 'embedding/models/tfidf_all.model'
        model_tfidf.save(word_path)

    # WORD2VEC MODEL
    exists = os.path.isfile('embedding/models/word2vec_all.model')
    if exists:
        print('Word2vec embedding model already existing')
    else:
        print('Training word2vec on all answers')
        word_path = "embedding/models/word2vec_all.model"
        word_tempfile = get_tmpfile(word_path)
        word_model = Word2Vec(all_docs_prepro, size=128, window=5, min_count=1, workers=4)
        word_model.save(word_path)


def tfidf_w2v_top5w(all_docs_prepro):
    # TFIDF MODEL
    exists = os.path.isfile('embedding/models/tfidf_all.model')
    if exists:
        print('Tfidf embedding model already existing')
    else:
        dct = Dictionary(all_docs_prepro)  # fit dictionary
        corpus = [dct.doc2bow(line) for line in all_docs_prepro]  # convert corpus to BoW format
        model_tfidf = TfidfModel(corpus)
        word_path = 'embedding/models/tfidf_all.model'
        model_tfidf.save(word_path)

    # WORD2VEC MODEL
    exists = os.path.isfile('embedding/models/word2vec_all.model')
    if exists:
        print('Word2vec embedding model already existing')
    else:
        print('Training word2vec on all answers')
        word_path = "embedding/models/word2vec_all.model"
        word_tempfile = get_tmpfile(word_path)
        word_model = Word2Vec(all_docs_prepro, size=128, window=5, min_count=1, workers=4)
        word_model.save(word_path)


def document_embedding(all_ans_prepro, ticket_ques_prepro):

    #ALL ANSWERS
    # checking if embedding model already exists
    exists = os.path.isfile('embedding/models/doc2vec_all.model')
    if exists:
        print('Doc2vec embedding model already existing')
    # Create Doc2Vec model in case it doesn't exists
    else:
        print('Training doc2vec on all answers')
        doc_path = "embedding/models/doc2vec_ans.model"
        doc_tempfile = get_tmpfile(doc_path)
        # DOC2VEC Model. 1 is distributed memory, 0 is distributed bag of words
        MODEL = 1
        tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_ans_prepro)]
        doc_model = Doc2Vec(tagged_docs, vector_size=128, window=5, min_count=1, workers=4, dm=MODEL)
        doc_model.save(doc_path)

    #TICKET QUESTIONS
    exists = os.path.isfile("embedding/models/doc2vec_ticket_ques.model")
    if exists:
        print('Doc2vec embedding model on ans already existing')
    else:
        print('Training doc2vec on ticket questions')
        doc_path = "embedding/models/doc2vec_ticket_ques.model"
        doc_tempfile = get_tmpfile(doc_path)
        # DOC2VEC Model. 1 is distributed memory, 0 is distributed bag of words
        MODEL = 1
        tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(ticket_ques_prepro)]
        doc_model = Doc2Vec(tagged_docs, vector_size=128, window=5, min_count=1, workers=4, dm=MODEL)
        doc_model.save(doc_path)


def dump_documents(all_docs, id_dict, all_docs_sep, all_ans_prepro, ticket_ques_prepro, ticket_ids, val_prepo,
                   test_prepo, all_docs_prepro):

    # Need to save this list and id dictionary as a pickle so we can decode IDs when we test things
    with open("embedding/models/doc_data/all_docs.txt", "wb") as fp:
        pickle.dump(all_docs, fp)

    with open("embedding/models/doc_data/id_dict.txt", "wb") as fp:
        pickle.dump(id_dict, fp)

    with open("embedding/models/doc_data/all_docs_sep.pkl", "wb") as fp:
        pickle.dump(all_docs_sep, fp)

    # preprocessed data to be saved
    with open("embedding/models/doc_data/all_ans_prepro.txt", "wb") as fp:
        pickle.dump(all_ans_prepro, fp)

    with open("embedding/models/doc_data/ticket_ques_prepro.txt", "wb") as fp:
        pickle.dump(ticket_ques_prepro, fp)

    # Also save the ticket_ids so can match with labelled data later
    with open("embedding/models/doc_data/ticket_ids.txt", "wb") as fp:
        pickle.dump(ticket_ids, fp)

    with open("embedding/models/doc_data/ticket_val.txt", "wb") as fp:
        pickle.dump(val_prepo, fp)

    with open("embedding/models/doc_data/ticket_test.txt", "wb") as fp:
        pickle.dump(test_prepo, fp)

    with open("embedding/models/doc_data/all_docs_prepro.txt", "wb") as fp:
        pickle.dump(all_docs_prepro, fp)


if __name__== "__main__":
    # TODO: allow argument passing
    embedding('tfidf_w2v_top5a', '../data/12-04-')
