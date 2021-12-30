import numpy as np
from sklearn.metrics import average_precision_score


def get_map(labels, scores):
    average_precision = np.zeros(5)
    for i in range(5):
        average_precision[i] = average_precision_score(
            labels[:, i], scores[:, i])
    return average_precision
