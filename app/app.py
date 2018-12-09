# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from utils import ui
from utils import pipeline


app = dash.Dash(__name__)

image_style = {
    'borderRadius':'100%',
    'display': 'inline-block',
    "width": "76%"}

images = [html.Div(html.Img(src='assets/brian.jpg', style=image_style),
                                style={'text-align':'center'}, className="four columns"),
                        html.Div(html.Img(src='assets/costanza.jpeg', style=image_style),
                                                        style={'text-align':'center'}, className="four columns"),
                        html.Div(html.Img(src='assets/maggie.jpeg', style=image_style),
                                 style={'text-align':'center'}, className="four columns")
                        ]

name_style = style={'textAlign':'center', 'color':'white'}

names = [
    html.Div(html.Span('Brian Regan'), className="four columns", style=name_style),
    html.Div(html.Span('Costanza Calzolari'), className="four columns", style=name_style),
    html.Div(html.Span('Margherita Rosnati'), className="four columns", style=name_style)
]

app.layout = html.Div([

    html.H1('Natural Language Misunderstanding', style={"textAlign":"center", 'paddingTop':"30px"}),
    html.H2('An Interesting Subtitle', style={"textAlign":"center"}),
    html.Div([
        dcc.Input(id='text-input', type='text', value='Sample Ticket Text', style={'width':'100%', 'height':'200px'})
        ], style={
        "width": '80%',
        "margin": "auto",
        "marginTop": '10px',
        "marginBottom": '50px'
    }
    ),
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
                 "background": "#333"
             },
             children= [
                 html.Div(images, style={'width':'80%', 'margin':'auto', "paddingTop":"30px"}, className="row"),
                 html.Div(names, style={'width':'80%', 'margin':'auto', "paddingTop":""}, className="row")

             ]
             ),
    html.Div(id='footer',
             style={
                 "width": "100%",
                 "height": "100px"
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


if __name__ == '__main__':
    app.run_server(debug=True)