# Function to delete all model associate static files

import os

dir_to_clean = ['classifier/models',
                'embedding/models',
                'embedding/models/doc_data',
                'prediction/test_data',
                'similarity/mappings']

for dir in dir_to_clean:
    print('Cleaning Directory {}'.format(dir))
    for file in os.listdir(dir):
        name = dir + '/' + file
        if os.path.isfile(name):
            os.remove(name)
            print('     Removed file {}'.format(os.path.join(name)))
