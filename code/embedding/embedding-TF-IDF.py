import pickle

import pandas as pd
from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer


def embedding():

    # Read in the data
    ticket_dat = pd.read_csv('../data/11-24-ticket_dat.csv')
    faq_dat = pd.read_csv('../data/11-24-faq_dat_cleaned.csv')

    # Replace the NaNs
    ticket_dat.fillna('', inplace=True)
    faq_dat.fillna('', inplace=True)

    # Make sentences into
    faq_ques = list(faq_dat.question)
    n_faq_ques = len(faq_ques)

    # FAQ answer is the answer and its title concatenated
    faq_ans = list(faq_dat.answer_title + " " + faq_dat.answer)
    n_faq_ans = len(faq_ans)


    ticket_ques = list(ticket_dat.question)
    n_ticket_ques = len(ticket_ques)


    ticket_ans = list(ticket_dat.answer)
    n_ticket_ans = len(ticket_ans)

    # Model assumption: two different embeddings
    all_ans = faq_ans + ticket_ans

    # create a dictionary storing the cut points for the four datasets so we can re-split them after.
    all_docs_sep = {
        'ticket_ans': ticket_ans,
        'FAQ_ans': faq_ans,
        'ticket_ques': ticket_ques,
        'FAQ_ques': faq_ques
    }

    # Need to save this list and id dictionary as a pickle so we can decode IDs when we test things
    with open("embedding/models/doc_data/all_docs_sep.pkl", "wb") as fp:
        pickle.dump(all_docs_sep, fp)

    # Model assumption: TF-IDF
    # initialise model
    vectoriser = TfidfVectorizer(strip_accents='unicode', lowercase=True, analyzer='word')
    # create matrix: rows = all ans; cols = TI-IDF weighted word vector
    vectoriser.fit(all_ans)
    dump(vectoriser, 'embedding/models/TF-IFD-ans.joblib')
    # train model on ans too
    # TODO: use this for classification?
    vec2 = TfidfVectorizer(strip_accents='unicode', lowercase=True, analyzer='word')
    vec2.fit(ticket_ques)
    dump(vec2, 'embedding/models/TF-IFD-ticket-ques.joblib')


if __name__== "__main__":
    embedding()
