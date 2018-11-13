import pandas as pd
from preprocessing import *
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
from gensim.models import KeyedVectors

# Read in the data
ticket_dat = pd.read_csv('../../data/ticket_dat.csv')
faq_dat = pd.read_csv('../../data/faq_dat.csv')

# Replace the NaNs
ticket_dat.fillna('', inplace=True)
faq_dat.fillna('', inplace=True)

# Make sentences into
faq_ques = list(faq_dat.ques_content_translation)
faq_ques_docs = preprocess_docs_fn(faq_ques)

faq_ans = list(faq_dat.ans_content_translated)
faq_ans_docs = preprocess_docs_fn(faq_ans)

ticket_content = list(ticket_dat.content_translated)
ticket_content_docs = preprocess_docs_fn(ticket_content)

all_docs = faq_ques_docs + faq_ans_docs + ticket_content_docs


# Create embedding model
path = get_tmpfile("models/word2vec.model")

print('Training')
model = Word2Vec(all_docs, size=300, window=5, min_count=1, workers=4)
model.save("models/word2vec.model")

# Load a pretrained google model and train on it some more
# print('Loading Pretrained Model')
# model = KeyedVectors.load_word2vec_format('models/GoogleNews-vectors-negative300.bin', binary=True)
# # Train on our data
# print('Training...')
# model.train(all_docs, total_examples=len(all_docs), epochs=20)


query_words = "password"
prepro_query = preprocess_sentence_fn(query_words)
print('Results for "{0}", processed to {1}'.format(query_words, prepro_query))
print(model.wv.most_similar(positive=prepro_query))