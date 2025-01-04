import torch
import torch.nn.functional as F
from typing import List, Optional
import clip

def compute_similarity(embedding_a, embedding_b, metric="cosine"):
    """
    Compute similarity between two embeddings using the specified metric.
    """
    if metric == "cosine":
        return F.cosine_similarity(embedding_a.unsqueeze(0), embedding_b.unsqueeze(0)).item()
    elif metric == "dot":
        return torch.dot(embedding_a, embedding_b).item()
    elif metric == "euclidean":
        return -torch.norm(embedding_a - embedding_b, p=2).item()
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def compute_avg_similarity(model, motion_emb, captions: List[str], device, metric="cosine") -> float:
    """
    Compute the average similarity between a motion embedding and a list of captions.
    """
    if not captions:
        return None  # No captions to process

    similarities = []
    for caption in captions:
        # Tokenize and encode text inputs
        text_inputs = clip.tokenize(caption).to(device)
        text_emb = model.text_encoder.encode_text(text_inputs).float()  
        text_emb = F.normalize(text_emb, p=2, dim=1)
        similarities.append(compute_similarity(motion_emb, text_emb.squeeze(), metric))
    
    return sum(similarities) / len(similarities)

def process_batch(model, motions, pos_caps_list, neg_caps_list, device, metric="cosine"):
    """
    Process a batch of motions, positive captions, and negative captions to compute
    average similarities.
    """
    pos_avg_sims = []
    neg_avg_sims = []

    for i, motion in enumerate(motions):
        # Positive similarities
        pos_avg_sims.append(compute_avg_similarity(model, motion, pos_caps_list[i], device, metric))

        # Negative similarities
        if neg_caps_list[i]:
            neg_avg_sims.append(compute_avg_similarity(model, motion, neg_caps_list[i], device, metric))
        else:
            neg_avg_sims.append(None)

    return pos_avg_sims, neg_avg_sims

def build_similarity_matrix(model, motion_embs, pos_caps_list, device, metric="cosine"):
    """
    Build a motion-to-caption similarity matrix.
    """
    num_motions = motion_embs.shape[0]
    similarity_matrix = torch.zeros(num_motions, num_motions, device=device)

    for i, motion in enumerate(motion_embs):
        for j, captions in enumerate(pos_caps_list):
            # Compute the average similarity between motion i and captions of motion j
            similarity_matrix[i, j] = compute_avg_similarity(
                model, motion, captions, device, metric
            )

    return similarity_matrix

def compute_car(pos_avg_sims, neg_avg_sims):
    """
    Compute CAR (Chronological Accuracy Rate).
    """
    num_correct = 0
    valid_samples = 0

    for pos_sim, neg_sim in zip(pos_avg_sims, neg_avg_sims):
        if neg_sim is not None:  # Only consider samples with valid negative captions
            valid_samples += 1
            if pos_sim > neg_sim:  # Positive similarity exceeds negative similarity
                num_correct += 1

    return num_correct / valid_samples if valid_samples > 0 else 0.0

def compute_r_precision(motion_cap_sim_matrix, k_values=[1, 3, 5]):
    """
    Compute R-Precision (Recall@k) from the similarity matrix.
    """
    num_samples = motion_cap_sim_matrix.shape[0]
    r_precision_values = {}

    for k in k_values:
        num_correct = 0

        for i in range(num_samples):
            # Retrieve similarities for motion i
            row = motion_cap_sim_matrix[i]

            # Sort indices by descending similarity
            topk_indices = torch.argsort(row, descending=True)[:k]

            # Check if the correct text index (i) is in the top-k indices
            if i in topk_indices:
                num_correct += 1

        # Compute R-Precision for the current value of k
        r_precision_values[f"r_precision_top_{k}"] = num_correct / num_samples

    return r_precision_values