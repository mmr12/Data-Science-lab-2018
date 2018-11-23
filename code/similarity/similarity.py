from gensim.models.doc2vec import Doc2Vec
import pickle
import numpy as np

def similarity():
    print('Loading Model...')
    model_path = 'embedding/models/doc2vec.model'
    model = Doc2Vec.load(model_path)
    #TODO: it would be nice to be able to select the embedding before computing similarities

    print('Loading Document Data...')
    # Unpickle the document data
    with open("embedding/models/doc_data/all_docs.txt", "rb") as fp:
        all_docs = pickle.load(fp)

    with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
        id_dict = pickle.load(fp)

    ticket_ans_ids = np.array(id_dict['ticket_ans'])
    all_faq_ans = id_dict['faq_ans']

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
    big_dist = [ticket_faq_dists.min(axis=1) > 0.5]
    ticket_faq_map[big_dist] = -1 # Set all thresholded distances to have label -1

    with open("similarity/mappings/ticket_faq_map.txt", "wb") as fp:
        pickle.dump(ticket_faq_map, fp)

if __name__== "__main__":
    similarity()



