import numpy as np
import torch

def euclidean_distance_matrix(motions, texts):
    """
    Calculate the Euclidean distance matrix between motion embeddings and text embeddings.
    Both motions and texts are assumed to have shape (B, D).
    """
    if isinstance(motions[0], list) or isinstance(motions[0], np.ndarray):
        motions = torch.stack([torch.tensor(m, dtype=torch.float32) for m in motions])  # Shape: (B, D)
    if isinstance(texts[0], list) or isinstance(texts[0], np.ndarray):
        texts = torch.stack([torch.tensor(t, dtype=torch.float32) for t in texts])  # Shape: (B, D)

    # Compute pairwise distances
    distances = torch.cdist(motions, texts, p=2)  # Shape: (B, B)
    return distances

def r_precision(distance_matrix, k=1):
    """
    Compute R-Precision for Top-k.
    """
    correct = 0
    for i, row in enumerate(distance_matrix):
        top_k_indices = torch.argsort(row)[:k]  # Indices of top-k closest
        if i in top_k_indices:
            correct += 1
    return correct / distance_matrix.size(0)

def similarity_fn(motion_embedding, text_embedding):
    """
    Custom similarity function between motion and text embedding.
    Replace with your actual model's similarity function.
    """
    return torch.dot(motion_embedding, text_embedding)

def chronological_accuracy_rate(motions, positive_texts, negative_texts):
    """
    Compute Chronological Accuracy Rate (CAR).
    """
    correct = 0
    total = len(motions)

    for motion, pos_text, neg_text in zip(motions, positive_texts, negative_texts):
        sim_pos = similarity_fn(motion, pos_text)
        sim_neg = similarity_fn(motion, neg_text)
        if sim_pos > sim_neg:
            correct += 1

    return correct / total