import pandas as pd

def load_example_texts():

    example_tables = pd.read_csv('assets/example_texts.csv')
    return list(example_tables.text)