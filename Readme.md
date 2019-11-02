## Sentiment RNN

This application performs sentiment analysis using RNNs. Using an RNN (instead of a simple FFNN) increases the accuracy since the input can be a sequence of words.

The following steps are taken:
1. Load in text data
2. Pre-process the data, encoding characters as integers
3. Pad the data such that each review is a standard sequence length
4. Define an RNN with embedding and hidden LSTM layers that predicts the sentiment of a given review
5. Train the RNN
6. Test performance on test data

Network architecture:
- Input words will be passed to an embedding layer to create a more efficient representation for the input data than one-hot encoded vectors. The embedding layer is for dimensionality reduction.
- The new embeddings will be passed to LSTM cells. The LSTM cells add recurrent connections to the network and give the ability to include information about the sequence of words.
- THe LSTM outputs will go to a sigmoid output layer. The sigmoid function will predict sentiment values between 0 (negative) and 1 (positive).

Loss calculation:
- Loss is calculated by comparing the output at the last time step and the training label. The sigmoid outputs for the intermediate layers are ignored.
