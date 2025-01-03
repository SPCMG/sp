import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_sim(a, b):
    # a: (N, D), b: (M, D)
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    return torch.matmul(a_norm, b_norm.t())

def push_loss(anchor_emb, pos_embs, margin=0.2):
    """
    We want anchor and positives to have a similarity >= margin.
    So if cos_sim < margin, we pay a penalty.
    push_loss = mean( ReLU(margin - cos_sim) ).
    """
    if pos_embs.shape[0] == 0:
        return 0.0
    # anchor_emb: shape (1, D)
    cos = cosine_sim(anchor_emb, pos_embs) # (1, P)
    # We want cos to be at least margin => penalty if cos < margin
    # L = mean( ReLU( margin - cos ) )
    L = F.relu(margin - cos).mean()
    return L

def pull_loss(anchor_emb, neg_embs, margin=0.2):
    """
    We want anchor and negatives to have a similarity <= some low threshold.
    If cos_sim > (1 - margin?), or we can re-use the same margin logic in a symmetrical way:
    L = mean( ReLU( cos - (1 - margin) ) ) if you want them below (1 - margin).
    Alternatively, a simpler approach: L = mean( ReLU( cos - margin ) ) to push them below margin.
    """
    if neg_embs.shape[0] == 0:
        return 0.0
    cos = cosine_sim(anchor_emb, neg_embs) # (1, N)
    # If cos > margin, we pay a penalty
    # L = mean( ReLU( cos - margin ) )
    L = F.relu(cos - margin).mean()
    return L

class HumanML3DLoss(nn.Module):
    """
    We define four separate losses:

    L1: anchor close to positives_same_motion
    L2: anchor far from negatives_same_motion
    L3: anchor close to positives_other_motion
    L4: anchor far from negatives_other_motion

    total_loss = alpha1*L1 + alpha2*L2 + alpha3*L3 + alpha4*L4
    """
    def __init__(
        self,
        margin=0.2,
        alpha1=1.0,  # weight for L1
        alpha2=1.0,  # weight for L2
        alpha3=1.0,  # weight for L3
        alpha4=1.0   # weight for L4
    ):
        super().__init__()
        self.margin = margin
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4

    def forward(
        self,
        pos_same_emb,           # shape (P, D)
        neg_same_emb,           # shape (N_s, D)
        pos_other_emb,          # shape (P2, D)
        neg_other_emb           # shape (N_o, D)
    ):
        # Calculate the centroid of pos_same_emb as the anchor embedding
        anchor_emb = pos_same_emb.mean(dim=0, keepdim=True)  # shape (1, D)

        # L1: anchor close to positives_same_motion
        L1 = push_loss(anchor_emb, pos_same_emb, margin=self.margin)

        # L2: anchor far from negatives_same_motion
        L2 = pull_loss(anchor_emb, neg_same_emb, margin=self.margin)

        # L3: anchor close to positives_other_motion
        L3 = push_loss(anchor_emb, pos_other_emb, margin=self.margin)

        # L4: anchor far from negatives_other_motion
        L4 = pull_loss(anchor_emb, neg_other_emb, margin=self.margin)

        total_loss = self.alpha1 * L1 + self.alpha2 * L2 + self.alpha3 * L3 + self.alpha4 * L4
        # Return the total plus each partial for logging
        return total_loss, (L1, L2, L3, L4)