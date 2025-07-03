import numpy as np

def topk_idx(v, k):
    return np.argsort(np.abs(v))[-k:][::-1]

def feature_disagreement(a, b, k=8):
    A, B = set(topk_idx(a, k)), set(topk_idx(b, k))
    return 1.0 - len(A & B) / k

def sign_disagreement(a, b, k=8):
    A, B = set(topk_idx(a, k)), set(topk_idx(b, k))
    overlap = A & B
    mismatches = sum(np.sign(a[i]) != np.sign(b[i]) for i in overlap)
    return (k - len(overlap) + mismatches) / k

def euclidean(a, b):
    a_n, b_n = a / np.linalg.norm(a), b / np.linalg.norm(b)
    return np.linalg.norm(a_n - b_n)

def euclidean_abs(a, b):
    a_n, b_n = np.abs(a), np.abs(b)
    a_n, b_n = a_n / np.linalg.norm(a_n), b_n / np.linalg.norm(b_n)
    return np.linalg.norm(a_n - b_n)