from code.embedding.preprocessing import *
from visualization_utils import *
from gensim.models import Word2Vec
import sys
import pandas as pd

# Function for visualizing embedding model
# run python visualization.py [model name].model

#save from input
fname = sys.argv[1]  #embedding model

#PARAMETERS for TSNE
perp = [5, 10] #tsne perplexity
#[5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
n_iter = 250 #tsne iterations
n_comp = 2 #do you want a 2D or 3D visualization?


#load the embedding model
model = Word2Vec.load('models/' + fname)

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

#unwrapping the embedding model
labels = []
tokens = []
col = [0]*len(model.wv.vocab)
print('Storing from which dataset words come from...')
i = 0
for word in model.wv.vocab:
    #print(i)
    tokens.append(model[word])
    labels.append(word)
    col[i] = belongs_to(word, faq_ques_docs, faq_ans_docs, ticket_content_docs)
    i += 1

#compute and plot PCA
print('Working on PCA plot')
pca_plot(tokens=tokens, col=col, fname=fname, n_comp=n_comp)

#compute and plot TSNE
for i in perp:
    print('CURRENTLY WORKING ON TSNE WITH PERPLEXITY = ' + str(i))
    tsne_plot(tokens=tokens, col=col, fname=fname, perp=i, n_iter=n_iter, n_comp=n_comp)






