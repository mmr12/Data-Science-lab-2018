from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Unpickle the document data, the ticket to FAQ map

with open("../embedding/models/doc_data/all_docs.txt", "rb") as fp:
    all_docs = pickle.load(fp)

with open("../embedding/models/doc_data/id_dict.txt", "rb") as fp:
    id_dict = pickle.load(fp)

with open("../similarity/mappings/ticket_faq_map.txt", "rb") as fp:
    ticket_faq_map = pickle.load(fp)

# Load the Doc2Vec model
model = Doc2Vec.load('../embedding/models/' + 'doc2vec.model')

# Get the embeddings of the tickets
ticket_question_embeddings = np.array([model.docvecs[x] for x in id_dict['ticket_ques']])

most_freq_class = pd.Series(ticket_faq_map).value_counts().index[0]
print('{0} class classification. {1} from uniform random guessing.'.format(len(np.unique(ticket_faq_map)), 1/len(np.unique(ticket_faq_map))))
print('Guessing Most Common Class ({1}): {0}'.format(sum(ticket_faq_map == most_freq_class)/len(ticket_faq_map), most_freq_class))

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