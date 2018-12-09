from .doc2vec import *
from .tfidf import RF
from .word2vec import *
from .tfidf_w2v import *


# Unpickle the document data, the ticket to FAQ map
def classifier(model):

    if model == 'tfidf':
        RF()

    elif model == 'word2vec':

        with open("embedding/models/doc_data/all_docs_prepro.txt", "rb") as fp:
            all_docs_prepro = pickle.load(fp)
        with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
            id_dict = pickle.load(fp)

        word_embedding(all_docs_prepro, id_dict)

    elif model == 'tfidf_w2v':

        with open("embedding/models/doc_data/all_docs_prepro.txt", "rb") as fp:
            all_docs_prepro = pickle.load(fp)
        with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
            id_dict = pickle.load(fp)

        tfidf_w2v(all_docs_prepro, id_dict)

    elif model == 'doc2vec':
        with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
            id_dict = pickle.load(fp)
        document_embedding(id_dict)

    else:
        print('Model {} not found'.format(model))


# TODO: included files here and not in directory file

if __name__ == "__main__":
    classifier('tfidf_w2v')