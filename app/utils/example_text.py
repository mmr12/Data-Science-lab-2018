import pickle
import random

def load_example_texts():

    with open('../data/12-08-val-test.pkl', 'rb') as f:
        val_set_tickets = pickle.load(f)

        return val_set_tickets['x_test']

def random_ticket(tickets):

    return random.choice(tickets)