import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import clip
from matplotlib import pyplot as plt
import numpy as np

from src.models.architectures.transformer import Encoder_TRANSFORMER

###########################################
# 1. Setup
###########################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Load CLIP model (text encoder)
clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
clip.model.convert_weights(clip_model)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# Parameters for motion encoder
params = {
    'modeltype': 'motionclip',
    'njoints': 22,
    'nfeats': 3,
    'num_frames': 200,
    'num_classes': 1,
    'translation': True,
    'pose_rep': 'rot6d',  # or 'xyz', ensure your data matches this
    'glob': True,
    'glob_rot': True,
    'latent_dim': 256,
    'ff_size': 1024,
    'num_layers': 4,
    'num_heads': 4,
    'dropout': 0.1,
    'activation': 'gelu'
}

motion_encoder = Encoder_TRANSFORMER(**params).to(device)


###########################################
# 2. Dummy Dataset
###########################################
# Replace this with your actual dataset. The dataset should return:
# - motion_data: tensor of shape (njoints, nfeats, nframes)
# - caption: a string describing the motion

class DummyMotionTextDataset(Dataset):
    def __init__(self, size=1000):
        self.size = size
        self.njoints = params['njoints']
        self.nfeats = params['nfeats']
        self.nframes = params['num_frames']

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Dummy motion: random tensor (njoints, nfeats, nframes)
        motion = torch.randn(self.njoints, self.nfeats, self.nframes)
        # Dummy caption
        caption = f"A person performing action number {idx}"
        return motion, caption

dataset = DummyMotionTextDataset()
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

###########################################
# 3. Loss and Optimizer
###########################################
# We'll use a contrastive loss similar to CLIP:
# For a batch of size B, we get motion embeddings and text embeddings,
# and form a BxB similarity matrix. We apply cross-entropy loss to align matched pairs.

optimizer = optim.Adam(motion_encoder.parameters(), lr=1e-4)
loss_ce = nn.CrossEntropyLoss()

###########################################
# 4. Training Loop
###########################################
epochs = 3

for epoch in range(epochs):
    motion_encoder.train()

    for batch_idx, (motion_data, captions) in enumerate(dataloader):
        motion_data = motion_data.to(device)  # (B, njoints, nfeats, nframes)
        # Tokenize captions for CLIP
        texts = clip.tokenize(captions).to(device)  # (B, seq_len)

        # Encode text
        with torch.no_grad():
            text_emb = clip_model.encode_text(texts).float()  # (B, 512)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Prepare batch dict for motion encoder
        B = motion_data.shape[0]
        y = torch.zeros(B, dtype=torch.long, device=device)
        lengths = torch.full((B,), params['num_frames'], dtype=torch.long, device=device)
        mask = torch.arange(params['num_frames'], device=device)[None, :] < lengths[:, None]
        batch_motion = {"x": motion_data, "y": y, "mask": mask}

        # Encode motion
        motion_out = motion_encoder(batch_motion)
        motion_emb = motion_out["mu"]  # (B, latent_dim)
        motion_emb = motion_emb / motion_emb.norm(dim=-1, keepdim=True)

        # Compute similarity matrix
        # Normalize and compute logit scaling if needed; here we just do dot product
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))  # CLIP default init
        logits_per_motion = (logit_scale.exp() * motion_emb @ text_emb.t())  # (B, B)
        logits_per_text = logits_per_motion.t()  # (B, B)

        ground_truth = torch.arange(B, dtype=torch.long, device=device)

        # Compute CLIP-like loss
        # This encourages motion i to match text i and vice versa
        loss_motion = loss_ce(logits_per_motion, ground_truth)
        loss_text = loss_ce(logits_per_text, ground_truth)
        loss = (loss_motion + loss_text) / 2.0

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] Batch {batch_idx}/{len(dataloader)} Loss: {loss.item():.4f}")

print("Training Complete!")
