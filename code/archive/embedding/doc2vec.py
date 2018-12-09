from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import os


def document_embedding(all_ans, ticket_ques):

    #ALL ANSWERS
    # checking if embedding model already exists
    exists = os.path.isfile('embedding/models/doc2vec_ans.model')
    if exists:
        print('Doc2vec embedding model already existing')
    # Create Doc2Vec model in case it doesn't exists
    else:
        print('Training doc2vec on all answers')
        doc_path = "embedding/models/doc2vec_ans.model"
        doc_tempfile = get_tmpfile(doc_path)
        # DOC2VEC Model. 1 is distributed memory, 0 is distributed bag of words
        MODEL = 1
        tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_ans)]
        doc_model = Doc2Vec(tagged_docs, vector_size=128, window=5, min_count=1, workers=4, dm=MODEL)
        doc_model.save(doc_path)

    #TICKET QUESTIONS
    print('Training doc2vec on ticket questions')
    doc_path = "embedding/models/doc2vec_ticket_ques.model"
    doc_tempfile = get_tmpfile(doc_path)
    # DOC2VEC Model. 1 is distributed memory, 0 is distributed bag of words
    MODEL = 1
    tagged_docs = [TaggedDocument(doc, [i]) for i, doc in enumerate(ticket_ques)]
    doc_model = Doc2Vec(tagged_docs, vector_size=128, window=5, min_count=1, workers=4, dm=MODEL)
    doc_model.save(doc_path)