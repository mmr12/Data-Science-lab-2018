import sys
from gensim.models import Word2Vec
from preprocessing import *

# Function for testing a word2vec model from the command line.
# run python test_w2v_model.py [model name].model [test word] to print the most similar words

fname = sys.argv[1]
query_word = sys.argv[2]

model = Word2Vec.load('models/' + fname)

prepro_query = preprocess_sentence_fn(query_word)
print('Results for "{0}", processed to {1}'.format(query_word, prepro_query))
print(model.wv.most_similar(positive=prepro_query))
