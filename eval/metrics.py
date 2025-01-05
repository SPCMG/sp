import torch
import torch.nn.functional as F
import clip

def compute_avg_similarity_batched(model, motion_emb, captions, device):
    """
    Encodes all 'captions' at once, then averages the similarity with 'motion_emb'.
    """
    if not captions:
        return None

    tokens = clip.tokenize(captions).to(device)
    text_feat = model.text_encoder.encode_text(tokens).float()
    text_feat = F.normalize(text_feat, p=2, dim=1)

    # If we use cosine similarity, we can do a batched dot product
    # motion_emb is [D], text_feat is [num_caps, D]
    # => cos_sim = (motion_emb * text_feat).sum(dim=-1)
    # => or we can use F.cosine_similarity
    sims = F.cosine_similarity(
        motion_emb.unsqueeze(0),  # [1, D]
        text_feat,               # [num_caps, D]
        dim=-1
    )  # [num_caps]

    return sims.mean().item()

def process_batch(model, batch_motion_emb, pos_caps_list, neg_caps_list, device):
    """
    batch_motion_emb: [B, D] motion embeddings for this batch
    pos_caps_list: List[List[str]] (length B)
    neg_caps_list: List[List[str]] (length B)
    """
    pos_avg_sims = []
    neg_avg_sims = []

    for i, motion_emb in enumerate(batch_motion_emb):
        # Positive
        pos_avg_sims.append(
            compute_avg_similarity_batched(model, motion_emb, pos_caps_list[i], device)
        )
        # Negative
        if neg_caps_list[i]:
            neg_avg_sims.append(
                compute_avg_similarity_batched(model, motion_emb, neg_caps_list[i], device)
            )
        else:
            neg_avg_sims.append(None)

    return pos_avg_sims, neg_avg_sims

def compute_car_vectorized(pos_avg_sims, neg_avg_sims):
    """
    Vectorized version of CAR.
    pos_avg_sims: Tensor of shape [N]
    neg_avg_sims: Tensor of shape [N], with some entries possibly -inf 
                  if negative captions are missing.
    """
    # 1) Mask where we have valid negatives
    valid_mask = (neg_avg_sims != float('-inf'))
    
    # 2) Extract only valid samples
    valid_pos_sims = pos_avg_sims[valid_mask]
    valid_neg_sims = neg_avg_sims[valid_mask]
    
    # 3) Count how many positive similarities are greater than negative
    num_correct = (valid_pos_sims > valid_neg_sims).sum()
    
    # 4) Compute ratio
    total_valid = valid_pos_sims.shape[0]
    if total_valid == 0:
        return 0.0
    
    return (num_correct / total_valid).item()

# --------------------------------------------------------------
def encode_and_average_captions(model, all_caption_lists, device):
    """
    all_caption_lists: List[List[str]]
        - Outer list has length N (number of motions).
        - Inner list contains all positive captions for that motion.
    Returns:
        text_embs: Torch tensor of shape [N, D]
          (average-encoded & normalized).
    """
    # We'll store the average text embedding for each motion
    avg_embs = []

    for captions_for_one_motion in all_caption_lists:
        if len(captions_for_one_motion) == 0:
            # If no captions exist, just store a zero vector (or skip)
            avg_embs.append(torch.zeros(model.text_encoder.output_dim, device=device))
            continue

        # 1) Tokenize all captions at once
        tokens = clip.tokenize(captions_for_one_motion).to(device)

        # 2) Encode text in one forward pass
        text_feat = model.text_encoder.encode_text(tokens).float()
        text_feat = F.normalize(text_feat, p=2, dim=1)

        # 3) Average over all captions for this motion
        avg_text_feat = text_feat.mean(dim=0)

        avg_embs.append(avg_text_feat)

    # Stack into a single [N, D] tensor
    text_embs = torch.stack(avg_embs, dim=0)
    # Optionally normalize again after averaging (up to you)
    text_embs = F.normalize(text_embs, p=2, dim=1)
    return text_embs


def build_similarity_matrix_fast(motion_embs, text_embs):
    """
    Vectorized version of building an NxN similarity matrix via a single
    matrix multiplication, assuming motion_embs and text_embs are both
    [N, D] and already normalized.
    """
    # motion_embs: [N, D]
    # text_embs:   [N, D]
    # Return:      [N, N]
    return motion_embs @ text_embs.t()

def compute_r_precision(sim_matrix, k_values=[1, 3, 5]):
    """
    sim_matrix: [N, N], sim_matrix[i, j] = similarity of motion i to text j
    ...
    """
    num_samples = sim_matrix.shape[0]
    r_precision_values = {}

    for k in k_values:
        num_correct = 0
        for i in range(num_samples):
            # Descending sort
            row = sim_matrix[i]
            topk_indices = torch.argsort(row, descending=True)[:k]
            # Check if the diagonal index i is in topk
            if i in topk_indices:
                num_correct += 1
        r_precision_values[f"r_precision_top_{k}"] = num_correct / num_samples

    return r_precision_values

