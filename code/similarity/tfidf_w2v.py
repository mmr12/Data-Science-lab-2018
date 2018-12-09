from gensim.models import Word2Vec
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from .tfidf_w2v_utils import *

def tfidf_w2v(all_ans_prepro, faq_ans):

    ### Loading all the data ###

    print('Loading Word2vec model')
    model_path = 'embedding/models/word2vec_ans.model'
    w2v_ans = Word2Vec.load(model_path)

    print('Loading tfidf model')
    model_path = 'embedding/models/tfidf_ans.model'
    tfidf_ans = TfidfModel.load(model_path)

    dct = Dictionary(all_ans_prepro)  # fit dictionary
    corpus = [dct.doc2bow(line) for line in all_ans_prepro]

    ### Computing similarity ###

    #AVERAGE 5 MOST IMPORTANT WORDS
    print('AVERAGE 5 MOST IMPORTANT WORDS')
    # create doc vector for tickets answers i.e. average over each ticket ans the word2vec vector for each word
    mean_ticket_ans_top5 = top5(all_ans_prepro[len(faq_ans):len(all_ans_prepro)], corpus, dct,
                                w2v_ans, tfidf_ans)
    # create doc vector for faq ans i.e. average over each faq ans the word2vec vector for each word
    mean_faq_ans_top5 = top5(all_ans_prepro[0:len(faq_ans)], corpus, dct,
                                w2v_ans, tfidf_ans)
    compute_sim(mean_ticket_ans_top5, mean_faq_ans_top5)
    print('\n')


    #WEIGHTED AVERAGE OVER 5 MOST IMPORTANT WORDS
    print('WEIGHTED AVERAGE OVER 5 MOST IMPORTANT WORDS')
    mean_ticket_ans_top5a = top5_average(all_ans_prepro[len(faq_ans):len(all_ans_prepro)], corpus, dct,
                                w2v_ans, tfidf_ans)
    mean_faq_ans_top5a = top5_average(all_ans_prepro[0:len(faq_ans)], corpus, dct,
                                w2v_ans, tfidf_ans)
    compute_sim(mean_ticket_ans_top5a, mean_faq_ans_top5a)
    print('\n')

    #WEIGHTED AVERAGE OVER ALL WORDS
    print('WEIGHTED AVERAGE OVER ALL WORDS')
    mean_ticket_ans_all = all_average(all_ans_prepro[len(faq_ans):len(all_ans_prepro)], corpus, dct,
                                w2v_ans, tfidf_ans)
    mean_faq_ans_all = all_average(all_ans_prepro[0:len(faq_ans)], corpus, dct,
                                w2v_ans, tfidf_ans)
    compute_sim(mean_ticket_ans_all, mean_faq_ans_all)
    print('\n')
