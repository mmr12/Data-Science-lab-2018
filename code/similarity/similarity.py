import pickle
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity
from gensim.corpora import Dictionary

from .utils import *


def similarity(model, thresh):

    print('Loading Document Data...')

    with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
        id_dict = pickle.load(fp)
        print('Loaded Id Dict')

    with open("embedding/models/doc_data/all_ans_prepro.txt", "rb") as fp:
        all_ans_prepro = pickle.load(fp)
        print('Loaded All Answers')

    with open("embedding/models/doc_data/all_docs_sep.pkl", "rb") as fp:
        all_docs_sep = pickle.load(fp)
        print('Loaded Separated Answers')

    with open("embedding/models/doc_data/ticket_val.txt", "rb") as fp:
        val_prepo = pickle.load(fp)

    with open("embedding/models/doc_data/ticket_test.txt", "rb") as fp:
        test_prepo = pickle.load(fp)

    with open("embedding/models/doc_data/all_docs_prepro.txt", "rb") as fp:
        all_docs_prepro = pickle.load(fp)
        print('Loaded All Answers')

    print('Loading Completed')

    ticket_ans_ids = np.array(id_dict['ticket_ans'])
    all_faq_ans = id_dict['faq_ans']

    faq_ans = all_docs_sep['faq_ans']
    n_faq = len(faq_ans)
    ticket_ans = all_docs_sep['ticket_ans']
    n_ticket = len(ticket_ans)

    if model == 'tfidf':
        tfidf(faq_ans, ticket_ans, thresh)
    elif model == 'word2vec':
        word_embedding(all_docs_prepro, id_dict, thresh)
    elif model == 'tfidf_w2v':
        tfidf_w2v(all_docs_prepro, id_dict)
    elif model == 'doc2vec':
        document_embedding(n_faq, n_ticket, thresh)
    else:
        print('No model {} found'.format(model))


def tfidf(faq_ans, ticket_ans, thresh):
    print('Loading TF-IDF Model...')
    TFiDF = load('embedding/models/TF-IFD-ans.joblib')

    # make matrix
    FAQ_matrix = TFiDF.transform(faq_ans)
    ticket_matrix = TFiDF.transform(ticket_ans)
    print('Computing TF-IDF Similarities...')
    sim_matrix = cosine_similarity(FAQ_matrix, ticket_matrix)

    output = similarities(sim_matrix, thresh)

    with open("similarity/mappings/ticket_faq_map_TF-IDF.pkl", "wb") as fp:
        pickle.dump(output, fp)


def word_embedding(all_docs_prepro, id_dict, thresh):
    print('Loading Word2vec model')
    model_path = 'embedding/models/word2vec_all.model'
    model = Word2Vec.load(model_path)

    print('Computing word2vec similarity')
    #create doc vector for tickets answers i.e. average over each ticket ans the word2vec vector for each word
    mean_ticket_ans = doc_emb_one(name='ticket_ans', id_dict=id_dict, all_docs_prepro=all_docs_prepro,
                                  model=model)
    #create doc vector for faq ans i.e. average over each faq ans the word2vec vector for each word
    mean_faq_ans = doc_emb_one(name='faq_ans', id_dict=id_dict, all_docs_prepro=all_docs_prepro, model=model)

    output = compute_sim(mean_ticket_ans=mean_ticket_ans, mean_faq_ans=mean_faq_ans, thresh=thresh)

    with open("similarity/mappings/ticket_faq_map_word2vec.pkl", "wb") as fp:
        pickle.dump(output, fp)

def tfidf_w2v(all_docs_prepro, id_dict):

    print('Loading Word2vec model')
    model_path = 'embedding/models/word2vec_all.model'
    model_w2v = Word2Vec.load(model_path)

    print('Loading Word2vec model')
    model_path = 'embedding/models/tfidf_all.model'
    model_tfidf = Word2Vec.load(model_path)

    dct = Dictionary(all_docs_prepro)
    corpus = [dct.doc2bow(line) for line in all_docs_prepro]

    mean_ticket_ans = all_average(dat='ticket_ans', corpus=corpus, dct=dct, model_w2v=model_w2v,
                                  model_tfidf=model_tfidf, id_dict=id_dict, all_docs_prepro=all_docs_prepro)
    mean_faq_ans = all_average(dat='faq_ans', corpus=corpus, dct=dct, model_w2v=model_w2v, model_tfidf=model_tfidf)

    output = compute_sim(mean_ticket_ans, mean_faq_ans)

    with open("../code/similarity/mappings/map_w2v_tfidf_all.pkl", "wb") as fp:
        pickle.dump(output, fp)


def document_embedding(all_faq_ans, ticket_ans_ids, thresh):
    print('Loading Doc2Vec Model...')
    print('WARNING: doc2vec has not been investigated in some time as data size may not sufficient')
    model_path = 'embedding/models/doc2vec_ans.model'
    model = Doc2Vec.load(model_path)

    # Presently compute distances to all and then filter to FAQs after, specifying other_docs = all_faq_ans doesn't seem
    # to work for some reason.
    def sim_to_faq(doc_id):
        # Computes similarity to all faqs for a given doc_id
        dists = model.docvecs.distances(doc_id, other_docs=())
        return np.array(dists[all_faq_ans])

    sim_to_faq_vec = np.vectorize(sim_to_faq, otypes=[object])

    print('Computing doc2vec Similarities...')
    ticket_faq_dists = np.stack(sim_to_faq_vec(ticket_ans_ids)) #array w/ similarity btw each ticket_ans and each faq ans
    ticket_faq_map = np.argmin(ticket_faq_dists, axis=1)

    # We should threshold the distances so that if the minimum distance is not below a certain value then we assign it an
    # unknown class
    big_dist = [ticket_faq_dists.min(axis=1) > 1- thresh]
    ticket_faq_map[big_dist] = -1 # Set all thresholded distances to have label -1

    with open("similarity/mappings/ticket_faq_map_doc2vec.txt", "wb") as fp:
        pickle.dump(ticket_faq_map, fp)


if __name__== "__main__":
    # TODO: allow command line arguments or something
    similarity('word2vec', 0.5)
