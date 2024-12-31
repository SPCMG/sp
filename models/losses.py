# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, margin=0.2):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, text_embeds, motion_embeds, labels):
        euclidean_distance = F.pairwise_distance(text_embeds, motion_embeds, keepdim=True)
        loss_contrastive = torch.mean((1 - labels) * torch.pow(euclidean_distance, 2) +
                                      labels * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive