import dash_html_components as html
import numpy as np
import pickle

def create_faq_boxes(faqs):

    # Generates an div containing FAQ boxes
    # faqs: a array of dictionaries with keys 'title', 'text' and 'prob' and 'index', one per FAQ

    children = []

    for faq in faqs:
        children.append(create_faq_box(faq))

    return html.Div(children)


def create_faq_box(faq):

    probability_style ={
        "background": '#57c5f7',
        "color": "white",
        "padding": "5px",
        "borderRadius": "5px"}

    # Generates an FAQ box
    # dictionary with keys 'title' and 'text' and 'prob' and 'index'

    box = html.Div([
        html.H3(faq['title']),
        html.Div([html.Span('Probability: {}'.format(faq['prob']), style=probability_style)]),
        html.Div([html.Span(faq['text'])])
    ],
    style={
        "boxShadow": "1px 2px 9px 0px",
        "padding": "10px",
        "margin": "5px 5px 20px 5px",
        "borderRadius": "10px"
    })

    return box


def faq_data_from_predictions(probs, labels):
    probs = probs[0]

    faqs = []

    # Create an array of FAQ dictionaries from the array of probs outputted from numpy predict_proba
    top_three_index = rev_sort(probs, 5)

    # These are the actual FAQ ids -1 etc. different from numpy index
    faq_indexs = labels[top_three_index]

    faq_raw_answer, faq_title = load_raw_faq_data()

    top_three_probs = probs[top_three_index]
    top_three_text = get_faq_texts(faq_indexs, faq_raw_answer)
    top_three_titles = get_faq_titles(faq_indexs, faq_title)


    for prob, text, title, index in zip(top_three_probs, top_three_text, top_three_titles, faq_indexs):
        faqs.append({
            'title': title,
            'text': text,
            'prob': prob,
            'index': index
        })

    print(top_three_index)
    return faqs

def load_raw_faq_data():
    # Load the raw title and content of FAQ for display
    with open("../code/embedding/models/doc_data/all_docs_sep.pkl", "rb") as fp:
        all_docs_sep = pickle.load(fp)
    return all_docs_sep['faq_raw_answer'], all_docs_sep['faq_title']

def get_faq_texts(faq_indexs, faq_raw_answer):
    texts = []
    for index in faq_indexs:
        if index == -1:
            texts.append('Please answer this manually')
        else:
            texts.append(faq_raw_answer[index])

    return texts

def get_faq_titles(faq_indexs, faq_title):
    titles = []
    for index in faq_indexs:
        if index == -1:
            titles.append('No FAQ in Dataset')
        else:
            titles.append(faq_title[index])

    return titles

def rev_sort(arr, n):
    # Return args of greatest n values in arr
    return arr.argsort()[-n:][::-1]

def button(text, id):
    return html.Div(html.Button(text, id=id, n_clicks_timestamp='0'), className="four columns", style={'textAlign':'center'})