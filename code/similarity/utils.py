import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from operator import itemgetter


# compute sentence embedding for word2vec
def doc_emb_one(name, id_dict, all_docs_prepro, model):
    mean_ans = np.empty((len(id_dict[name]), 128), dtype=float)
    for j in id_dict[name]:
        sentence = all_docs_prepro[j]
        words = np.empty((len(sentence), 128), dtype=float)
        for i in range(len(sentence)):
            words[i] = model[sentence[i]]
        mean_ans[j - id_dict[name][0]] = np.apply_along_axis(np.mean, 0, words)
    return mean_ans

# compute sentence embedding for tfidf_w2v
def all_average(dat, corpus, dct, model_w2v, model_tfidf, id_dict, all_docs_prepro):
    if dat == 'faq_ans':
        ind = id_dict['faq_ans'][0]
        leng = len(id_dict['faq_ans'])
        dat = all_docs_prepro[ind:leng]
    elif dat == 'ticket_ans':
        ind = id_dict['ticket_ans'][0]
        leng = len(id_dict['ticket_ans'])
        dat = all_docs_prepro[ind:leng]
    else:
        ind = id_dict['ticket_ques'][0]
        leng = len(id_dict['ticket_ques'])
        dat = all_docs_prepro[ind:leng]
    mean_ans = np.empty((leng, 128), dtype=float)
    for i in range(leng):
        vector = np.asarray(model_tfidf[corpus[ind]], dtype=float)
        words = np.empty((len(vector), 128), dtype=float)
        for j in range(len(vector)):
            words[j] = model_w2v[dct[int(vector[j,0])]]
        mean_ans[i] = np.average(words, 0, weights=vector[:,1])
        ind += 1
    return mean_ans

def top5(dat, corpus, dct, model_w2v, model_tfidf, id_dict, all_docs_prepro):
    if dat == 'faq_ans':
        ind = id_dict['faq_ans'][0]
        leng = len(id_dict['faq_ans'])
        dat = all_docs_prepro[ind:leng]
    elif dat == 'ticket_ans':
        ind = id_dict['ticket_ans'][0]
        leng = len(id_dict['ticket_ans'])
        dat = all_docs_prepro[ind:leng]
    else:
        ind = id_dict['ticket_ques'][0]
        leng = len(id_dict['ticket_ques'])
        dat = all_docs_prepro[ind:leng]
    mean_ans = np.empty((leng, 128), dtype=float)
    for i in range(leng):
        vector = model_tfidf[corpus[ind]]
        vector_s = sorted(vector, key=itemgetter(1), reverse=True)
        top5 = vector_s[:5]
        top5 = np.asarray(top5, dtype=int)[:,0]
        words = np.empty((len(top5), 128), dtype=float)
        for j in range(len(top5)):
            words[j] = model_w2v[dct[top5[j]]]
        mean_ans[i] = np.apply_along_axis(np.mean, 0, words)
        ind += 1
    return mean_ans

def top5_average(dat, corpus, dct, model_w2v, model_tfidf, id_dict, all_docs_prepro):
    if dat == 'faq_ans':
        ind = id_dict['faq_ans'][0]
        leng = len(id_dict['faq_ans'])
        dat = all_docs_prepro[ind:leng]
    elif dat == 'ticket_ans':
        ind = id_dict['ticket_ans'][0]
        leng = len(id_dict['ticket_ans'])
        dat = all_docs_prepro[ind:leng]
    else:
        ind = id_dict['ticket_ques'][0]
        leng = len(id_dict['ticket_ques'])
        dat = all_docs_prepro[ind:leng]
    mean_ans = np.empty((leng, 128), dtype=float)
    for i in range(leng):
        vector = model_tfidf[corpus[ind]]
        vector_s = sorted(vector, key=itemgetter(1), reverse=True)
        top5 = vector_s[:5]
        top5 = np.asarray(top5, dtype=float)
        words = np.empty((len(top5), 128), dtype=float)
        for j in range(len(top5)):
            words[j] = model_w2v[dct[int(top5[j,0])]]
        mean_ans[i] = np.average(words, 0, weights=top5[:,1])
        ind += 1
    return mean_ans

# compute similarity automatically
def compute_sim(mean_ticket_ans, mean_faq_ans, thresh):
    print('Computing similarity')

    # create matrix with cosine distances from all ticket ans to all faq ans
    sim_matrix = cosine_similarity(mean_faq_ans, mean_ticket_ans)

    # most similar faq - ticket mapping
    FAQ_per_ticket = np.argmax(sim_matrix, axis=0)
    strength_FAQ_ticket = np.max(sim_matrix, axis=0)

    # small similarities are set to a separate class
    FAQ_per_ticket[strength_FAQ_ticket < thresh] = -1

    # some stats
    uniques, counts = np.unique(FAQ_per_ticket, return_counts=True)
    n_unique = len(uniques)
    solo_classes = np.sum(counts == 1)
    n_nonassigned = np.shape(FAQ_per_ticket[strength_FAQ_ticket < thresh])[0]
    n_tickets = len(FAQ_per_ticket)
    # How many tickets each FAQ is assigned
    # counts_per_faq = pd.Series(FAQ_per_ticket).value_counts()
    #print(counts_per_faq)

    output = {
        'classes': n_tickets,
        'mapping': FAQ_per_ticket
    }
    print(n_unique, '\tclasses out of 200')
    print(round(n_nonassigned / n_tickets, 2), '\tnon assigned tickets (should be ~0.75)')
    print(solo_classes, "\tclasses with one ticket only")
    return output