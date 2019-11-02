import numpy as np
import torch
from collections import Counter

def encode_data(words, reviews_split, labels):
    word_counts = Counter(words)
    # sorting the words from most to least frequent
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionary
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab, 1)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    # convert reviews to integers
    reviews_ints = []
    for review in reviews_split:
        reviews_ints.append([vocab_to_int[word] for word in review.split()])

    # encoding the labels
    labels_split = labels.split('\n')
    encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])

    return reviews_ints, encoded_labels, int_to_vocab, vocab_to_int
