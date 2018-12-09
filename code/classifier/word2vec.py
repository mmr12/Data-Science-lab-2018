import pickle

from gensim.models import Word2Vec

from .utils import *


def word_embedding(ticket_ques_prepro):

    # Load the Word2Vec model
    model_path = 'embedding/models/word2vec_ticket_ques.model'
    model = Word2Vec.load(model_path)

    with open('similarity/mappings/ticket_faq_map_word2vec.pkl', 'rb') as fp:
        Classes = pickle.load(fp)

    mapping = Classes['mapping']

    def doc_emb(dat):
        mean_ans = np.empty((len(dat), 128), dtype=float)
        for j in range(len(dat)):
            sentence = dat[j]
            words = np.empty((len(sentence), 128), dtype=float)
            for i in range(len(sentence)):
                words[i] = model[sentence[i]]
            mean_ans[j] = np.apply_along_axis(np.mean, 0, words)
        return mean_ans

    ticket_question_embeddings = doc_emb(ticket_ques_prepro)

    return ticket_question_embeddings, mapping
