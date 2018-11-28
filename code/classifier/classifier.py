from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import Word2Vec
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# Unpickle the document data, the ticket to FAQ map
def classifier(mod):
    with open("embedding/models/doc_data/all_docs.txt", "rb") as fp:
        all_docs = pickle.load(fp)

    with open("embedding/models/doc_data/id_dict.txt", "rb") as fp:
        id_dict = pickle.load(fp)

    if mod == "word2vec":

        with open("similarity/mappings/ticket_faq_map_word2vec_cosine.txt", "rb") as fp:
            ticket_faq_map = pickle.load(fp)

        with open("embedding/models/doc_data/all_docs_prepro.txt", "rb") as fp:
            all_docs_prepro = pickle.load(fp)

        # Load the Word2Vec model
        model_path = 'embedding/models/word2vec.model'
        model = Word2Vec.load(model_path)

        # embedding for the ticket questions
        ticket_question_embeddings = np.empty((len(id_dict['ticket_ques']), 128), dtype=float)
        for j in id_dict['ticket_ques']:
            sentence = all_docs_prepro[j]
            words = np.empty((len(sentence), 128), dtype=float)
            for i in range(len(sentence)):
                words[i] = model[sentence[i]]
            ticket_question_embeddings[j - id_dict['ticket_ques'][0]] = np.apply_along_axis(np.mean, 0, words)

        most_freq_class = pd.Series(ticket_faq_map).value_counts().index[0]
        print('{0} class classification. {1} from uniform random guessing.'.format(len(np.unique(ticket_faq_map)),
                                                                                   1 / len(np.unique(ticket_faq_map))))
        print(
            'Guessing Most Common Class ({1}): {0}'.format(sum(ticket_faq_map == most_freq_class) / len(ticket_faq_map),
                                                           most_freq_class))

    elif mod == "doc2vec":

        with open("similarity/mappings/ticket_faq_map_doc2vec_cosine.txt", "rb") as fp:
            ticket_faq_map = pickle.load(fp)

        # Load the Doc2Vec model
        model = Doc2Vec.load('embedding/models/' + 'doc2vec.model')

        # Get the embeddings of the tickets
        ticket_question_embeddings = np.array([model.docvecs[x] for x in id_dict['ticket_ques']])

        most_freq_class = pd.Series(ticket_faq_map).value_counts().index[0]
        print('{0} class classification. {1} from uniform random guessing.'.format(len(np.unique(ticket_faq_map)),
                                                                                   1 / len(np.unique(ticket_faq_map))))
        print(
            'Guessing Most Common Class ({1}): {0}'.format(sum(ticket_faq_map == most_freq_class) / len(ticket_faq_map),
                                                           most_freq_class))

    print('Training Classifier...')
    classifier = RandomForestClassifier()
    classifier.fit(X=ticket_question_embeddings, y=ticket_faq_map)
    train_score = classifier.score(X=ticket_question_embeddings, y=ticket_faq_map)

    print('Running CV on Classifier...')
    classifier_CV = RandomForestClassifier()
    scores = cross_val_score(classifier_CV, ticket_question_embeddings, ticket_faq_map, cv=5)
    cv_score = scores.mean()

    # Some classes only appear once maybe we should assign them a -1

    print('Training Score: {0} \nCross Val Score: {1}'.format(train_score, cv_score))


if __name__ == "__main__":
    classifier(mod)