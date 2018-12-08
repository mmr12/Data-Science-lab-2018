from .word2vec import *
from .doc2vec import *
from .tfidf import RF


# Unpickle the document data, the ticket to FAQ map
def classifier(model):
    with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
        id_dict = pickle.load(fp)

    with open("embedding/models/doc_data/ticket_ques_prepro.txt", "rb") as fp:
        ticket_ques_prepro = pickle.load(fp)

    if model == 'tfidf':
        RF()
    elif model == 'word2vec':
        word_embedding(ticket_ques_prepro)
    elif model == 'doc2vec':
        document_embedding(id_dict)
    else:
        print('Model {} not found'.format(model))


# TODO: included files here and not in directory file

if __name__ == "__main__":
    classifier()