import sys
from gensim.models import Word2Vec
from gensim.models.doc2vec import Doc2Vec
from preprocessing import *
import pickle

# Function for testing a doc2vec word2vec model from the command line.
# run python test_model.py [model name].model [model_type] [test word] to print the most similar docs or words

fname = sys.argv[1]
model_type = sys.argv[2]
query = sys.argv[3]


prepro_query = preprocess_sentence_fn(query)
print('Results for "{0}", processed to {1}'.format(query, prepro_query))

if model_type == "word":
    model = Word2Vec.load('embedding/models/' + fname)
    print(model.wv.most_similar(positive=prepro_query))

elif model_type == "doc":
    # Unpickle the tagged docs
    with open("embedding/models/doc_data/all_docs.txt", "rb") as fp:
        all_docs = pickle.load(fp)

    model = Doc2Vec.load('embedding/models/' + fname)
    query_doc_vec = model.infer_vector(prepro_query)
    most_similar_docs = model.docvecs.most_similar([query_doc_vec])

    for similar_doc in most_similar_docs[0:4]:
        print('[{}] '.format(round(similar_doc[1], 5))+all_docs[similar_doc[0]] + "\n")


else:
    print("No Model Type: {}".format(model_type))