import pickle

from gensim.models.doc2vec import Doc2Vec

from .utils import *


def document_embedding(id_dict):

    print('DOC2VEC PERFORMANCE')

    with open("similarity/mappings/ticket_faq_map_doc2vec.txt", "rb") as fp:
        Classes = pickle.load(fp)

    # nclasses = Classes['classes']
    ticket_faq_map = Classes['mapping']

    # Load the Doc2Vec model
    model = Doc2Vec.load('embedding/models/' + 'doc2vec_ticket_ques.model')

    # Get the embeddings of the tickets
    r = range(len(id_dict['ticket_ques']))
    ticket_question_embeddings = np.array([model.docvecs[x] for x in r])

    return ticket_question_embeddings, ticket_faq_map
