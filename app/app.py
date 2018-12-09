# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from utils import ui
from utils import pipeline


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Input(id='input-1-keypress', type='text', value='Sample Ticket Text',
              style={
                  "width": '100%'
              }),
    html.Div(id='output-keypress')
])

# Inital Loading Steps (run before app)
MODEL = 'tfidf'
embedder = pipeline.load_embedder(model=MODEL)
classifier = pipeline.load_classifier(model=MODEL)


@app.callback(Output('output-keypress', 'children'),
              [Input('input-1-keypress', 'value')])
def update_output(input1):

    probs, labels = pipeline.predict(text=input1, embedder=embedder, classifier=classifier, model=MODEL)
    faqs = ui.faq_data_from_predictions(probs, labels)

    return ui.create_faq_boxes(faqs)


if __name__ == '__main__':
    app.run_server(debug=True)