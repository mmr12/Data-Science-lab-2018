from gensim.models.doc2vec import Doc2Vec
import pickle
import numpy as np

print('Loading Model...')
model_path = '../embedding/models/doc2vec.model'
model = Doc2Vec.load(model_path)

print('Loading Document Data...')
# Unpickle the document data
with open("../embedding/models/doc_data/all_docs.txt", "rb") as fp:
    all_docs = pickle.load(fp)

with open("../embedding/models/doc_data/id_dict.txt", "rb") as fp:
    id_dict = pickle.load(fp)

test_document_ids = np.array(id_dict['ticket_ans'])
all_faq_ans = id_dict['faq_ans']

# Presently compute distances to all and then filter to FAQs after, specifying other_docs = all_faq_ans doesn't seem
# to work for some reason.

def sim_to_faq(doc_id):
    # Computes similarity to all faqs for a given doc_id
    dists = model.docvecs.distances(doc_id, other_docs=())
    return np.array(dists[all_faq_ans])


sim_to_faq_vec = np.vectorize(sim_to_faq, otypes=[object])

print('Computing Similarities...')
d = np.stack(sim_to_faq_vec(test_document_ids))

print(d.shape)
print(d)




