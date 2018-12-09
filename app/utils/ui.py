import dash_html_components as html
import numpy as np

def create_faq_boxes(faqs):

    # Generates an div containing FAQ boxes
    # faqs: a array of dictionaries with keys 'title', 'text' and 'prob', one per FAQ

    children = []

    for faq in faqs:
        children.append(create_faq_box(faq))

    return html.Div(children)


def create_faq_box(faq):

    # Generates an FAQ box
    # dictionary with keys 'title' and 'text' and 'prob'

    box = html.Div([
        html.H3(faq['title']),
        html.P(faq['prob']),
        html.P(faq['text'])
    ])

    return box


def faq_data_from_predictions(probs, labels):
    probs = probs[0]

    faqs = []

    # Create an array of FAQ dictionaries from the array of probs outputted from numpy predict_proba
    top_three_index = rev_sort(probs, 5)

    # These are the actual FAQ ids -1 etc. different from numpy index
    faq_indexs = labels[top_three_index]

    top_three_probs = probs[top_three_index]
    top_three_text = get_faq_texts(faq_indexs)
    top_three_titles = get_faq_titles(faq_indexs)

    for prob, text, title in zip(top_three_probs, top_three_text, top_three_titles):
        faqs.append({
            'title':title,
            'text':text,
            'prob':prob
        })

    print(top_three_index)
    return faqs

def get_faq_texts(indexs):
    # TODO: have this use the FAQ dataset to return the text of the FAQs
    return np.repeat('test', len(indexs))

def get_faq_titles(indexs):
    # TODO: have this use the FAQ dataset to return the title of the FAQs
    return indexs

def rev_sort(arr, n):
    # Return args of greatest n values in arr
    return arr.argsort()[-n:][::-1]