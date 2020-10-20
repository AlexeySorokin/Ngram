import numpy as np


def get_perplexity(data, base=2):
    total_score, total_length = 0, 0
    for elem in data:
        total_score += -elem["total_prob"]
        total_length += len(elem["probs"])
    answer = total_score / total_length
    answer /= np.log10(base)
    return answer