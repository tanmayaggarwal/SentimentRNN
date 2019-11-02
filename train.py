import numpy as np
import torch
import torch.nn as nn

def train(net, train_on_gpu, batch_size, train_loader, valid_loader):
    # setting the data and training hyperparameters
    lr = 0.001
    epochs = 3
    clip = 5 # gradient clipping

    # loss and optimization function
    criterion = nn.BCELoss()  # use Binary Cross Entropy Loss given we are working with a single Sigmoid output (applies cross entropy loss to a single value between 0 and 1)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    counter = 0
    print_every = 100

    # move model to GPU, if available
    if(train_on_gpu):
        net.cuda()

    net.train()

    for e in range(epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size, train_on_gpu)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # creating new variables for the hidden state to avoid backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # clip gradient to prevent exploding gradient problem
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            optimizer.step()

            # loss statistics
            if counter % print_every == 0:
                # get validation loss
                val_h = net.init_hidden(batch_size, train_on_gpu)
                val_losses = []
                net.eval()
                for inputs, labels in valid_loader:
                    val_h = tuple([each.data for each in val_h])
                    if(train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                net.train()
                print("Epoch: {}/{}...".format(e+1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))

    return net, criterion
