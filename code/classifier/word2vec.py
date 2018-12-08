from gensim.models import Word2Vec
from joblib import dump
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

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

    print('Running CV on Classifier...')
    classifier_CV = RandomForestClassifier()
    scores = cross_val_score(classifier_CV, ticket_question_embeddings, mapping, cv=5)
    cv_score = scores.mean()

    print('Training Classifier...')
    classifier = RandomForestClassifier()
    classifier.fit(X=ticket_question_embeddings, y=mapping)
    dump(classifier, 'classifier/models/RF_word2vec.joblib')
    train_score = classifier.score(X=ticket_question_embeddings, y=mapping)

    print('Training Score: {0} \n Cross Val Score: {1}'.format(train_score, cv_score))
    print('###############')