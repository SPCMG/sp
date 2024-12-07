import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from models.architectures.transformer import Encoder_TRANSFORMER

# -----------------------------
# 1. LOAD CLIP TEXT ENCODER
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
clip.model.convert_weights(clip_model)  # ensures float16 precision if available

# Freeze CLIP weights if you do not want to train it
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

# -----------------------------
# 2. INITIALIZE MOTION ENCODER
# -----------------------------
# Example parameters for Encoder_TRANSFORMER:
params = {
    'modeltype': 'motionclip',
    'njoints': 22,        # adjust to your dataset
    'nfeats': 3,          # adjust if each joint is (x,y,z)
    'num_frames': 200,    # maximum number of frames per sequence
    'num_classes': 1,     # not really used here, just dummy
    'translation': True,
    'pose_rep': 'rot6d',  # or 'xyz' depending on how your data is represented
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

# -----------------------------
# 3. PREPARE INPUT DATA
# -----------------------------
# Suppose you have:
# - Motion batch of shape: (B, njoints, nfeats, nframes)
# - Corresponding text: A list of captions

# Example dummy data:
batch_size = 4
njoints = params['njoints']
nfeats = params['nfeats']
nframes = params['num_frames']

# Dummy motion: random tensor
motion_data = torch.randn(batch_size, njoints, nfeats, nframes, device=device)

# Dummy text: a list of strings
captions = ["A person walking forward", "A person running in place", "A person jumping up", "A person turning around"]

# For CLIP text encoding, we need to tokenize
texts = clip.tokenize(captions).to(device)  # shape: (B, seq_len)

# -----------------------------
# 4. ENCODE TEXT INTO EMBEDDINGS
# -----------------------------
with torch.no_grad():
    text_emb = clip_model.encode_text(texts).float()  # shape: (B, 512)
    # Normalize if needed
    text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
print("Text embedding shape:", text_emb.shape)  # (B, 512)

# -----------------------------
# 5. ENCODE MOTION INTO EMBEDDINGS
# -----------------------------
# The motion encoder expects a dictionary with "x", "y", "mask".
# "x": motion data (B, njoints, nfeats, nframes)
# "y": class labels (we can set them to 0 if unused)
# "mask": boolean mask indicating valid frames (B, nframes)

y = torch.zeros(batch_size, dtype=torch.long, device=device)
lengths = torch.full((batch_size,), nframes, dtype=torch.long, device=device)  # all sequences full length
mask = torch.arange(nframes, device=device)[None, :] < lengths[:, None]  # (B, nframes)

batch_motion = {"x": motion_data, "y": y, "mask": mask}

with torch.no_grad():
    motion_out = motion_encoder(batch_motion)
    motion_emb = motion_out["mu"]  # shape: (B, latent_dim)
    # Normalize if needed
    motion_emb = motion_emb / motion_emb.norm(dim=-1, keepdim=True)
print("Motion embedding shape:", motion_emb.shape)  # (B, 256)

# -----------------------------
# 6. USE YOUR OWN LOSS
# -----------------------------
# Now you have motion_emb and text_emb.
# For example, a simple cosine similarity loss:
cosine_sim = nn.CosineSimilarity(dim=1)
similarity = cosine_sim(motion_emb, text_emb)  # shape: (B,)
# Suppose you want them to match closely, you can define a loss as:
loss = (1 - similarity).mean()
print("Loss:", loss.item())

# This is just an example. You can define and incorporate your custom loss terms here.
