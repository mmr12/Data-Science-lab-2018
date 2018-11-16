import pandas as pd
from preprocessing import *
from gensim.test.utils import get_tmpfile
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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


# Create word embedding model
word_path = "models/word2vec.model"
word_tempfile = get_tmpfile(word_path)

print('Training Word Model')
word_model = Word2Vec(all_docs, size=128, window=5, min_count=1, workers=4)
word_model.save(word_path)
print('Trained')


# Create Doc2Vec model
doc_path = "models/doc2vec.model"
doc_tempfile = get_tmpfile(doc_path)

# DOC2VEC Model. 1 is distributed memory, 0 is distributed bag of words
MODEL = 1

print('Training Doc Model')
tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_docs)]
doc_model = Doc2Vec(tagged_docs, vector_size=128, window=5, min_count=1, workers=4, dm=MODEL)
doc_model.save(doc_path)
print('Trained')