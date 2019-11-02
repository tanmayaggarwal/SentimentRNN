import numpy as np
import torch
import torch.nn as nn

def test(net, train_on_gpu, batch_size, test_loader, criterion):

    # get test data loss and accuracy
    test_losses = []
    num_correct = 0

    # init hidden state
    h = net.init_hidden(batch_size, train_on_gpu)

    net.eval()
    #iterate over test data
    for inputs, labels in test_loader:
        h = tuple([each.data for each in h])

        if(train_on_gpu):
            inputs, labels = inputs.cuda(), labels.cuda()

        # get predicted outputs
        output, h = net(inputs, h)

        # calculate loss
        test_loss = criterion(output.squeeze(), labels.float())
        test_losses.append(test_loss.item())

        # convert output probabilities to predicted class (0 or 1)
        pred = torch.round(output.squeeze())

        # compare predictions to true labels
        correct_tensor = pred.eq(labels.float().view_as(pred))
        correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
        num_correct += np.sum(correct)

    return test_losses, num_correct
