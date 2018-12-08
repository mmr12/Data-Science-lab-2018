from .word2vec import *
from .doc2vec import *
from .tfidf import RF


# Unpickle the document data, the ticket to FAQ map
def classifier():
    with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
        id_dict = pickle.load(fp)

    with open("embedding/models/doc_data/ticket_ques_prepro.txt", "rb") as fp:
        ticket_ques_prepro = pickle.load(fp)


    ############################

    #word2vec
    word_embedding(ticket_ques_prepro)

    ############################

    #doc2vec
    #document_embedding(id_dict)

    ############################

    #tfidf
    #RF()


if __name__ == "__main__":
    classifier()