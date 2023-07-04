import numpy as np

def softmax(x):
    xmax = np.max(x)
    return np.exp(x - xmax) / np.sum(np.exp(x - xmax))