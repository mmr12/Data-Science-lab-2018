from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
import os

def word_embedding(all_ans_prepro, ticket_ques_prepro):

    #ALL ANSWERS
    # checking if embedding model already exists
    exists = os.path.isfile('embedding/models/word2vec_ans.model')
    if exists:
        print('Word2vec embedding model already existing')
    # Create word embedding model
    else:
        print('Training word2vec on all answers')
        word_path = "embedding/models/word2vec_ans.model"
        word_tempfile = get_tmpfile(word_path)
        word_model = Word2Vec(all_ans_prepro, size=128, window=5, min_count=1, workers=4)
        word_model.save(word_path)

    #TICKET QUESTIONS
    exists = os.path.isfile('embedding/models/word2vec_ticket_ques.model')
    if exists:
        print('Word2vec embedding model already existing')
    else:
        #not checking if already exists because if the first doesn't this won't either
        print('Training word2vec on ticket questions')
        word_path = "embedding/models/word2vec_ticket_ques.model"
        word_tempfile = get_tmpfile(word_path)
        word_model = Word2Vec(ticket_ques_prepro, size=128, window=5, min_count=1, workers=4)
        word_model.save(word_path)