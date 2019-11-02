import numpy as np
import torch
from string import punctuation
from collections import Counter

def pre_process(reviews):
    # remove all punctuation
    reviews = reviews.lower()
    all_text = ''.join([c for c in reviews if c not in punctuation])

    # split by new lines and spaces
    reviews_split = all_text.split('\n')
    all_text = ' '.join(reviews_split)

    # create a list of words
    words = all_text.split()

    return words, reviews_split

def remove_outliers(reviews_ints, encoded_labels):
    # removing outliers
    # getting rid of extremely long or short reviews

    # outlier review stats
    review_lens = Counter([len(x) for x in reviews_ints])
    print("Zero-length reviews: {}".format(review_lens[0]))
    print("Maximum review length: {}".format(max(review_lens)))

    # remove reviews with zero length and corresponding labels in encoded_labels

    print("Number of reviews before removing outliers: ", len(reviews_ints))

    # get indices of reviews with length 0
    non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

    # remove 0-length reviews and their labels
    reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
    encoded_labels = np.array([encoded_labels[ii] for ii in non_zero_idx])

    print("Number of reviews after removing outliers: ", len(reviews_ints))

    return reviews_ints, encoded_labels

# padding / truncating remaining data so that reviews are of the same length

def pad_features(reviews_ints, seq_length):
    # return features of review_ints, where each review is padded with 0's or truncated to the input seq_length

    features = np.zeros((len(reviews_ints), seq_length), dtype=int)

    for i in range(len(reviews_ints)):
        if len(reviews_ints[i]) < 200:
            review = np.array(reviews_ints[i])
            review = np.pad(review, (seq_length-len(review),0), 'constant', constant_values=0)
            features[i] = review
        elif len(reviews_ints[i]) > 200:
            review = np.array(reviews_ints[i])
            review = review[0:seq_length]
            features[i] = review

    return features
