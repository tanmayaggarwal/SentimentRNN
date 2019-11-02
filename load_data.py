import numpy as np

def load_data(reviews_path, labels_path):
    # read data from text files
    with open(reviews_path, 'r') as f:
        reviews = f.read()
    with open(labels_path, 'r') as f:
        labels = f.read()

    print (reviews[:2000])
    print()
    print(labels[:20])
    return reviews, labels
