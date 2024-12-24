import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from models.architectures.transformer import Encoder_TRANSFORMER


#############################################
# 1) CLIP Text Encoder (same as MotionCLIP) #
#############################################
class ClipTextEncoder(nn.Module):
    """
    Exactly uses CLIP's text encoder from the official OpenAI CLIP model
    (as MotionCLIP does). We freeze the weights so we don't train them.
    """
    def __init__(self, clip_model_name="ViT-B/32", device="cuda"):
        super().__init__()
        self.device = device
        # Load the full CLIP model, get rid of the image encoder if you want
        # but typically we keep everything. We'll just not use the image encoder.
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        
        # Freeze all CLIP parameters
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def forward(self, text_tokenized):
        """
        text_tokenized: Already tokenized text (shape [bs, token_length])
        Returns: [bs, 512] text embedding
        """
        return self.clip_model.encode_text(text_tokenized).float()


###########################################
# 2) Transformer-based Motion Encoder (Modified)
###########################################
class MotionTransformerEncoder(nn.Module):
    """
    Modified to handle 263-dimensional input features directly.
    """
    def __init__(
        self,
        modeltype="transformer",
        n_features=263,          # Updated to 263
        num_frames=196,          # Maximum number of frames
        latent_dim=512,          # Embedding dimension
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        device="cuda",
        **kwargs
    ):
        super().__init__()
        self.device = device
        self.latent_dim = latent_dim

        # Instantiate the modified Encoder_TRANSFORMER
        self.encoder = Encoder_TRANSFORMER(
            modeltype=modeltype,
            n_features=n_features,
            num_frames=num_frames,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            **kwargs
        )

    def forward(self, motion_batch):
        """
        motion_batch: dictionary containing
            - "x": [bs, max_len, 263] (Tensor)
            - "mask": [bs, max_len] (bool Tensor)
        Returns:
            motion_emb: [bs, latent_dim]
        """
        motion_emb = self.encoder(motion_batch)  # [bs, latent_dim]
        return motion_emb


###########################################
# 3) Simple Contrastive Loss (CLIP-like)  #
###########################################
class ContrastiveLoss(nn.Module):
    """
    CLIP-like contrastive loss to align motion and text embeddings.
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, motion_emb, text_emb):
        # Normalize embeddings
        motion_emb = F.normalize(motion_emb, dim=-1)  # [bs, latent_dim]
        text_emb = F.normalize(text_emb, dim=-1)      # [bs, 512]

        # Ensure dimensions match
        if motion_emb.size(-1) != text_emb.size(-1):
            raise ValueError(f"Embedding dimensions must match. Got motion_emb: {motion_emb.size(-1)}, text_emb: {text_emb.size(-1)}")

        # Compute similarity matrices
        logits_per_motion = torch.matmul(motion_emb, text_emb.t()) / self.temperature  # [bs, bs]
        logits_per_text = logits_per_motion.t()                                        # [bs, bs]

        # Ground truth labels
        batch_size = motion_emb.size(0)
        ground_truth = torch.arange(batch_size, device=motion_emb.device)

        # Compute cross-entropy losses
        loss_motion = F.cross_entropy(logits_per_motion, ground_truth)
        loss_text = F.cross_entropy(logits_per_text, ground_truth)
        total_loss = (loss_motion + loss_text) / 2.0

        return total_loss


#########################################################
# 4) Combined MotionClipModel (Motion + Text)         #
#########################################################
class MotionClipModel(nn.Module):
    """
    Combined MotionCLIP model with a motion encoder, text encoder, and contrastive loss.
    """
    def __init__(
        self,
        # Motion encoder options
        njoints=22,                  # Number of joints
        n_feats=263,                 # Number of motion features per frame
        num_frames=196,              # Maximum number of frames
        latent_dim=512,              # Latent dimension for embeddings
        # CLIP text encoder
        clip_model_name="ViT-B/32",
        device="cuda"
    ):
        super().__init__()
        self.device = device

        # 1. Motion Encoder (Transformer)
        self.motion_encoder = MotionTransformerEncoder(
            modeltype="transformer",
            n_features=n_feats,
            num_frames=num_frames,
            latent_dim=latent_dim,
            ff_size=1024,
            num_layers=4,
            num_heads=4,
            dropout=0.1,
            activation="gelu",
            device=device
        )

        # 2. Text Encoder (CLIP)
        self.text_encoder = ClipTextEncoder(
            clip_model_name=clip_model_name, 
            device=device
        )

        # 3. Contrastive Loss
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)

    def forward(self, motion_batch, text_tokenized):
        """
        Args:
            motion_batch: dictionary containing:
                - "x": [bs, max_len, 263] (Tensor)
                - "mask": [bs, max_len] (bool Tensor)
                - "lengths": [bs] (int Tensor)
                - "y": [bs] (Tensor) - not used in contrastive loss
            text_tokenized: [bs, token_length] (Tensor)
        Returns:
            dict containing:
                - "loss": scalar loss
                - "motion_emb": [bs, latent_dim]
                - "text_emb": [bs, 512]
        """
        # Ensure motion_batch is on the correct device
        for k, v in motion_batch.items():
            if isinstance(v, torch.Tensor):
                motion_batch[k] = v.to(self.device)

        # Tokenize text is already done outside and passed as 'text_tokenized'
        text_tokenized = text_tokenized.to(self.device)

        # 1. Motion Encoder forward => motion embedding
        motion_emb = self.motion_encoder(motion_batch)  # [bs, latent_dim]

        # 2. Text Encoder forward => text embedding
        text_emb = self.text_encoder(text_tokenized)      # [bs, 512]

        # 3. Contrastive Loss
        loss = self.contrastive_loss(motion_emb, text_emb)

        return {
            "loss": loss,
            "motion_emb": motion_emb,
            "text_emb": text_emb
        }
