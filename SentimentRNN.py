import numpy as np
import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    # The RNN model that will be used to perform Sentiment analysis

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, drop_prob=0.5):
        # initialize the model
        super(SentimentRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # define the layers
        self.embed = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        # perform a forward pass on some input and hidden state
        batch_size = x.size(0)
        x = self.embed(x)
        r_out, hidden = self.lstm(x, hidden)
        out = r_out.contiguous().view(-1, self.hidden_dim)
        out = self.dropout(out)
        out = self.fc(out)
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1] # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, hidden

    def init_hidden(self, batch_size, train_on_gpu):
        # initialize hidden state
        # create two new tensors with size n_layers x batch_size x hidden_dim
        # initialize to zero, for hidden state and cell state of LSTM

        weight = next(self.parameters()).data

        if(train_on_gpu):
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
