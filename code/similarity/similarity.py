from gensim.models.doc2vec import Doc2Vec
from gensim.models import Word2Vec
import pickle
import numpy as np
from scipy.spatial import distance

def similarity(mod):

    with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
        id_dict = pickle.load(fp)

    ticket_ans_ids = np.array(id_dict['ticket_ans'])
    all_faq_ans = id_dict['faq_ans']

    print('Loading Document Data...')
    # Unpickle the document data
    with open("embedding/models/doc_data/all_docs.txt", "rb") as fp:
        all_docs = pickle.load(fp)

    if mod == "word2vec":
        with open("embedding/models/doc_data/all_docs_prepro.txt", "rb") as fp:
            all_docs_prepro = pickle.load(fp)

        print('Loading Model...')
        model_path = 'embedding/models/word2vec.model'
        model = Word2Vec.load(model_path)

        #create doc vector for tickets answers i.e. average over each ticket ans the word2vec vector for each word
        mean_ticket_ans = np.empty((len(id_dict['ticket_ans']), 128), dtype=float)
        for j in id_dict['ticket_ans']:
            sentence = all_docs_prepro[j]
            words = np.empty((len(sentence), 128), dtype=float)
            for i in range(len(sentence)):
                words[i] = model[sentence[i]]
            mean_ticket_ans[j - id_dict['ticket_ans'][0]] = np.apply_along_axis(np.mean, 0, words)

        #create doc vector for faq ans i.e. average over each faq ans the word2vec vector for each word
        mean_faq_ans = np.empty((len(id_dict['faq_ans']), 128), dtype=float)
        for j in id_dict['faq_ans']:
            sentence = all_docs_prepro[j]
            words = np.empty((len(sentence), 128), dtype=float)
            for i in range(len(sentence)):
                words[i] = model[sentence[i]]
            mean_faq_ans[j - id_dict['faq_ans'][0]] = np.apply_along_axis(np.mean, 0, words)

        #create matrix with cosine distances from all ticket ans to all faq ans
        ticket_faq_dists = np.empty((len(mean_ticket_ans), len(mean_faq_ans)), dtype=float)
        for i in range(len(mean_ticket_ans)):
            for j in range(len(mean_faq_ans)):
                ticket_faq_dists[i, j] = distance.cosine(mean_ticket_ans[i], mean_faq_ans[j])

        #most similar faq - ticket mapping
        ticket_faq_map = np.argmin(ticket_faq_dists, axis=1)

        #too big distances are set to a separate class
        big_dist = [ticket_faq_dists.min(axis=1) > 0.7]
        ticket_faq_map[big_dist] = -1  # Set all thresholded distances to have label -1

        with open("similarity/mappings/ticket_faq_map_word2vec_cosine.txt", "wb") as fp:
            pickle.dump(ticket_faq_map, fp)

    elif mod == "doc2vec":
        print('Loading Model...')
        model_path = 'embedding/models/doc2vec.model'
        model = Doc2Vec.load(model_path)

        # Presently compute distances to all and then filter to FAQs after, specifying other_docs = all_faq_ans doesn't seem
        # to work for some reason.

        def sim_to_faq(doc_id):
            # Computes similarity to all faqs for a given doc_id
            dists = model.docvecs.distances(doc_id, other_docs=())
            return np.array(dists[all_faq_ans])


        sim_to_faq_vec = np.vectorize(sim_to_faq, otypes=[object])

        print('Computing Similarities...')
        ticket_faq_dists = np.stack(sim_to_faq_vec(ticket_ans_ids)) #array w/ similarity btw each ticket_ans and each faq ans
        ticket_faq_map = np.argmin(ticket_faq_dists, axis=1)

        # We should threshold the distances so that if the minimum distance is not below a certain value then we assign it an
        # unknown class
        big_dist = [ticket_faq_dists.min(axis=1) > 0.7]
        ticket_faq_map[big_dist] = -1 # Set all thresholded distances to have label -1
        #to do: correct this assignment

        with open("similarity/mappings/ticket_faq_map_doc2vec_cosine.txt", "wb") as fp:
            pickle.dump(ticket_faq_map, fp)

if __name__== "__main__":
    similarity(mod)



