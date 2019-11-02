import numpy as np
import torch
from string import punctuation
from collections import Counter

def split_data(features, encoded_labels, split_frac):
    split_idx = int(len(features)*0.8)
    train_x, remaining_x = features[:split_idx], features[split_idx:]
    train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

    test_idx = int(len(remaining_x)*0.5)
    val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
    val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]
    
    return train_x, train_y, val_x, val_y, test_x, test_y
