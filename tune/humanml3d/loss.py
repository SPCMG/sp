import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_sim(a, b):
    # a: (N, D), b: (M, D)
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    return torch.matmul(a_norm, b_norm.t())

class HumanML3DLoss(nn.Module):
    """
    Loss A: anchor <-> positives_same_motion vs. anchor <-> positives_of_other_motion
    (we're calling them 'negatives_other_motion' from the anchor's perspective).
    
    Loss B: anchor <-> positives_same_motion vs. anchor <-> negatives_same_motion (swapped actions).
    """
    def __init__(self, margin=0.2, alpha=1.0, beta=1.0):
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta

    def forward(self, anchor_emb, pos_embs, neg_other_embs, neg_same_embs):
        """
        anchor_emb: (1, D)
        pos_embs: (P, D)
        neg_other_embs: (N1, D)
        neg_same_embs: (N2, D)
        """
        # 1) Loss A: anchor closer to pos_embs than to neg_other_embs
        L_A = 0.0
        if pos_embs.shape[0] > 0 and neg_other_embs.shape[0] > 0:
            pos_sim = cosine_sim(anchor_emb, pos_embs)       # shape: (1, P)
            neg_sim = cosine_sim(anchor_emb, neg_other_embs) # shape: (1, N1)
            # Expand for pairwise
            pos_sim_expanded = pos_sim.transpose(0,1).repeat(1, neg_sim.shape[1])  # (P, N1)
            neg_sim_expanded = neg_sim.repeat(pos_sim.shape[1], 1)                 # (P, N1)
            L_A_matrix = F.relu(self.margin - pos_sim_expanded + neg_sim_expanded)
            L_A = L_A_matrix.mean()

        # 2) Loss B: anchor closer to pos_embs than to neg_same_embs
        L_B = 0.0
        if pos_embs.shape[0] > 0 and neg_same_embs.shape[0] > 0:
            pos_sim = cosine_sim(anchor_emb, pos_embs)      # shape: (1, P)
            neg_sim = cosine_sim(anchor_emb, neg_same_embs) # shape: (1, N2)
            pos_sim_expanded = pos_sim.transpose(0,1).repeat(1, neg_sim.shape[1])  # (P, N2)
            neg_sim_expanded = neg_sim.repeat(pos_sim.shape[1], 1)                 # (P, N2)
            L_B_matrix = F.relu(self.margin - pos_sim_expanded + neg_sim_expanded)
            L_B = L_B_matrix.mean()

        total_loss = self.alpha * L_A + self.beta * L_B
        return total_loss, (L_A, L_B)
