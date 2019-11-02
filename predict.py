import numpy as np
import torch
import torch.nn as nn
from pre_process import pre_process, pad_features

def predict(train_on_gpu, net, test_review, vocab_to_int, sequence_length=200):
    # prints out whether a given a review is predicted to be positive or negative in sentiment using a trained model
    # parameters include net: a trained network, test_review = review made of normal text and punctuations, sequence_length = the padded length of a review

    # pre-process and tokenize the review
    words, reviews_split = pre_process(test_review)
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in words])

    # test sequence padding
    features = pad_features(test_ints, sequence_length)
    # convert to tensor
    feature_tensor = torch.from_numpy(features)

    net.eval()
    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size, train_on_gpu)

    if (train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    # get predicted output
    output, h = net(feature_tensor, h)

    # convert output probability to predicted class (0 or 1)
    pred = torch.round(output.squeeze())

    if (pred.item()==1):
        print("Positive review detected!")
    else:
        print("Negative review detected.")
