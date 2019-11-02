import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


# load the data

from load_data import load_data

reviews_path = 'data/reviews.txt'
labels_path = 'data/labels.txt'

reviews, labels = load_data(reviews_path, labels_path)

# pre-process the data

from pre_process import pre_process
words, reviews_split = pre_process(reviews)
print(words[:30])

# encoding the data

from encode_data import encode_data
reviews_ints, encoded_labels, int_to_vocab, vocab_to_int = encode_data(words, reviews_split, labels)
print('Unique words: ', len((vocab_to_int)))
print()

# print tokens in first review as test
print('Tokenized review: \n', reviews_ints[:1])

# removing outliers
from pre_process import remove_outliers
reviews_ints, encoded_labels = remove_outliers(reviews_ints, encoded_labels)

# padding the sequences to be of equal length
from pre_process import pad_features
seq_length = 200 # can be modified as needed
features = pad_features(reviews_ints, seq_length)

# creating data sets
from split_data import split_data
split_frac = 0.8

train_x, train_y, val_x, val_y, test_x, test_y = split_data(features, encoded_labels, split_frac)
print("\t\t\tFeature Shapes:")
print("Train set: \t\t{}".format(train_x.shape),
      "\nValidation set: \t{}".format(val_x.shape),
      "\nTest set: \t\t{}".format(test_x.shape))

# create data loaders
# create Tensor datasets
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

# data loaders
batch_size = 50

# shuffle data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

# obtain one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

print('Sample input size: ', sample_x.size())
print('Sample input: \n', sample_x)
print()
print('Sample label size: ', sample_y.size())
print('Sample label: \n', sample_y)

# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# define the network
from SentimentRNN import SentimentRNN

# instantiate the model with hyperparameters
vocab_size = len(vocab_to_int)+1
output_size = 1
embedding_dim = 200
hidden_dim = 512
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)

# training the model

from train import train
net, criterion = train(net, train_on_gpu, batch_size, train_loader, valid_loader)

# testing the model
from test import test
test_losses, num_correct = test(net, train_on_gpu, batch_size, test_loader, criterion)

# print stats
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))
# accuracy
test_acc = num_correct/len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

# inference on a test review
from predict import predict

# test reviews
test_review_neg = "This was a crappy movie. The story line was terrible."
test_review_pos = "I loved this movie. What a fantastic plot. The actors were great as well."

seq_length = 200

predict(train_on_gpu, net, test_review_neg, seq_length)
