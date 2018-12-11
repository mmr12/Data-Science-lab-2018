import pickle

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from joblib import load

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
        tfidf_w2v(all_docs_prepro, id_dict, thresh)
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

    output = compute_sim(ticket_matrix, FAQ_matrix, thresh)

    with open("similarity/mappings/ticket_faq_map_TF-IDF.pkl", "wb") as fp:
        pickle.dump(output, fp)


def word_embedding(all_docs_prepro, id_dict, thresh):
    print('Loading Word2vec model')
    model_path = 'embedding/models/word2vec_all.model'
    model = Word2Vec.load(model_path)

    print('Computing word2vec similarity')
    # create doc vector for tickets answers i.e. average over each ticket ans the word2vec vector for each word
    mean_ticket_ans = doc_emb_one(name='ticket_ans', id_dict=id_dict, all_docs_prepro=all_docs_prepro,
                                  model=model)
    # create doc vector for faq ans i.e. average over each faq ans the word2vec vector for each word
    mean_faq_ans = doc_emb_one(name='faq_ans', id_dict=id_dict, all_docs_prepro=all_docs_prepro, model=model)

    output = compute_sim(mean_ticket_ans=mean_ticket_ans, mean_faq_ans=mean_faq_ans, thresh=thresh)

    with open("similarity/mappings/ticket_faq_map_word2vec.pkl", "wb") as fp:
        pickle.dump(output, fp)


def tfidf_w2v(all_docs_prepro, id_dict, thresh):

    print('Loading Word2vec model')
    model_path = 'embedding/models/word2vec_all.model'
    model_w2v = Word2Vec.load(model_path)

    print('Loading Tfidf model')
    model_path = 'embedding/models/tfidf_all.model'
    model_tfidf = TfidfModel.load(model_path)

    dct = Dictionary(all_docs_prepro)
    corpus = [dct.doc2bow(line) for line in all_docs_prepro]

    mean_ticket_ans = all_average(dat='ticket_ans', corpus=corpus, dct=dct, model_w2v=model_w2v,
                                  model_tfidf=model_tfidf, id_dict=id_dict, all_docs_prepro=all_docs_prepro)
    mean_faq_ans = all_average(dat='faq_ans', corpus=corpus, dct=dct, model_w2v=model_w2v, model_tfidf=model_tfidf,
                               id_dict=id_dict, all_docs_prepro=all_docs_prepro)

    output = compute_sim(mean_ticket_ans, mean_faq_ans, thresh)

    with open("../code/similarity/mappings/map_w2v_tfidf_all.pkl", "wb") as fp:
        pickle.dump(output, fp)


def document_embedding(n_faq, n_ticket, thresh):
    print('Loading Doc2Vec Model...')
    print('FIXED by MR - contact her if this stops working close to 09Dec')
    model_path = 'embedding/models/doc2vec_ans.model'
    model = Doc2Vec.load(model_path)

    # new strategy
    faq_em = np.array([model.docvecs[i] for i in range(n_faq)])
    ticket_em = np.array([model.docvecs[i] for i in range(n_faq, n_faq + n_ticket)])
    output = compute_sim(ticket_em, faq_em, thresh)

    with open("similarity/mappings/ticket_faq_map_doc2vec.txt", "wb") as fp:
        pickle.dump(output, fp)


if __name__== "__main__":
    # TODO: allow command line arguments or something
    similarity('tfidf_w2v', 0.5)
