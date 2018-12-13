# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from utils import ui
from utils import pipeline
from utils import example_text
import numpy as np


app = dash.Dash(__name__)

image_style = {
    'borderRadius':'100%',
    'display': 'inline-block',
    "width": "76%",
    "-webkitFilter":"grayscale(100%)"}

example_texts = example_text.load_example_texts()

images = [html.Div(html.Img(src='assets/brian.jpg', style=image_style),
                                style={'text-align':'center'}, className="four columns"),
                        html.Div(html.Img(src='assets/costanza_newer.jpeg', style=image_style),
                                                        style={'text-align':'center'}, className="four columns"),
                        html.Div(html.Img(src='assets/maggie_new.jpeg', style=image_style),
                                 style={'text-align':'center'}, className="four columns")
                        ]

name_style = style={'textAlign':'center', 'color':'white'}

names = [
    html.Div(html.Span('Brian Regan'), className="four columns", style=name_style),
    html.Div(html.Span('Costanza Calzolari'), className="four columns", style=name_style),
    html.Div(html.Span('Margherita Rosnati'), className="four columns", style=name_style)
]

app.layout = html.Div([

    # Links to fonts and icons
    html.Link(rel="stylesheet", href="https://use.fontawesome.com/releases/v5.6.0/css/all.css",integrity="sha384-aOkxzJ5uQz7WBObEZcHvV5JvRW3TUc2rNPA7pe3AwnsUohiw1Vj2Rgx2KSOkF5+h", crossOrigin="anonymous"),
    html.Link(rel="stylesheet", href="https://fonts.googleapis.com/css?family=Lato"),
    html.Link(rel="stylesheet", href="https://fonts.googleapis.com/css?family=Roboto"),

    html.H1('Natural Email Understanding',
            style={"textAlign":"center",
                   'paddingTop':"30px",
                   "fontFamily":"Roboto, Arial",
                   "fontStyle":"bold"}),
    html.H2('Automated FAQ Selection in IT Tickets Handling', style={"textAlign":"center"}),
    html.Div([
        dcc.Textarea(id='text-input',
                     placeholder='Write or paste Ticket Text here or select an example below',
                     style={'width':'100%', 'height':'200px'})
        ], style={
        "width": '80%',
        "margin": "auto",
        "marginTop": '10px',
        "marginBottom": '10px'
    }
    ),
    html.Div([
        html.Button([html.I(className="fas fa-eraser", style={'padding':"5px"}),'Clear'], id='clear-text', n_clicks_timestamp='0')
    ], style={'textAlign':'center', 'width':'80%', 'margin':'auto', 'marginBottom':'10px'}),
    html.Div([
        ui.button('Positive Sample', 'sample-load-1', icon='fas fa-check'),
        ui.button('Positive Sample', 'sample-load-2', icon='fas fa-check'),
        ui.button('Negative Sample', 'sample-load-3', icon='fas fa-times'),
        ui.button('Random Sample', 'sample-load-4', icon='fas fa-dice')
    ],
    style={
        'width':'80%',
        'margin':'auto'
    },
    className='row'),
    html.Div(id='faq-outputs',
             style={
                 "width": "80%",
                 "height": "1000px",
                 "overflow": "scroll",
                 "margin": "auto",
                 "marginTop": '10px',
                 "marginBottom": '10px'
             }),
    html.Div(id='team-info',
             style = {
                 "width":"100%",
                 "height": "400px",
                 "background": "#1269b0"
             },
             children= [
                 html.Div(images, style={'width':'80%', 'margin':'auto', "paddingTop":"30px"}, className="row"),
                 html.Div(names, style={'width':'80%', 'margin':'auto', "paddingTop":""}, className="row")

             ]
             ),
    html.Div([
        dcc.Markdown('This project is part of the [Data Science Lab](https://www.dslab.io/) at ETH. We would like to extend a thank you to Professors Zhang, Krause and Feuerriegel and Bernhard Kratzwald for the academic guidance, Mark Buschor for the time given to us and [food&lab](https://www.ethz.ch/en/campus/getting-to-know/cafes-restaurants-shops/gastronomy/restaurants-and-cafeterias/zentrum/food-lab.html) for sustaining all of our meetings'),
        html.A(html.I(className="fab fa-github", id="github-icon"),
               href="https://github.com/DS3Lab/DSL2018-Proj-ETH-IT",
               style={'color':'#999'})
    ],
        id='footer',
        style={
            "width": "60%",
            "margin":"auto",
            "height": "100px",
            "padding": "10px",
            "textAlign": "center",
            "fontSize": "14px",
            "color": "#999"
        })
])

# Inital Loading Steps (run before app)
MODEL = 'tfidf'
embedder = pipeline.load_embedder(model=MODEL)
classifier = pipeline.load_classifier(model=MODEL)


@app.callback(Output('faq-outputs', 'children'),
              [Input('text-input', 'value')])
def update_output(input1):

    probs, labels = pipeline.predict(text=input1, embedder=embedder, classifier=classifier, model=MODEL)
    faqs = ui.faq_data_from_predictions(probs, labels)

    return ui.create_faq_boxes(faqs)

@app.callback(Output('text-input', 'value'),
              [Input('sample-load-1', 'n_clicks_timestamp'),
               Input('sample-load-2', 'n_clicks_timestamp'),
               Input('sample-load-3', 'n_clicks_timestamp'),
               Input('sample-load-4', 'n_clicks_timestamp'),
               Input('clear-text', 'n_clicks_timestamp')])
def on_click(btn1, btn2, btn3, btn4, clear_btn):
    good_index1, good_index2, bad_index = 7, 98, 14

    timestamps = np.array([btn1, btn2, btn3, btn4, clear_btn]).astype(np.float)
    if np.max(timestamps) >0:
        max_index = np.argmax(timestamps)

        if max_index == 0:
            msg = example_texts[good_index1]
        elif max_index == 1:
            msg = example_texts[good_index2]
        elif max_index == 2:
            msg = example_texts[bad_index]
        elif max_index == 3:
            msg = example_text.random_ticket(example_texts)
        else:
            msg = ''

        return msg
    else:
        return ''



if __name__ == '__main__':
    app.run_server(debug=True)