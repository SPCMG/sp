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


################################################
# 2) Transformer-based Motion Encoder (MotionCLIP)
#    from motionCLIP's Encoder_TRANSFORMER
################################################
class MotionTransformerEncoder(nn.Module):
    """
    Wraps around the Encoder_TRANSFORMER from the original MotionCLIP code.
    We'll instantiate it with all the necessary hyperparams. 
    """
    def __init__(
        self,
        modeltype="transformer", 
        njoints=24, 
        nfeats=6, 
        num_frames=60, 
        num_classes=1,
        translation=True, 
        pose_rep="xyz", 
        glob=True, 
        glob_rot=True,
        latent_dim=512,        # match CLIP's 512 dimension
        ff_size=1024,
        num_layers=4,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        ablation=None,
    ):
        super().__init__()
        # Instantiate the actual MotionCLIP Encoder_TRANSFORMER
        self.encoder = Encoder_TRANSFORMER(
            modeltype=modeltype,
            njoints=njoints,
            nfeats=nfeats,
            num_frames=num_frames,
            num_classes=num_classes,
            translation=translation,
            pose_rep=pose_rep,
            glob=glob,
            glob_rot=glob_rot,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            ablation=ablation,
            activation=activation
        )
        self.latent_dim = latent_dim

    def forward(self, motion_batch):
        """
        motion_batch is expected to be a dict with at least:
            {
              "x": [bs, njoints, nfeats, nframes],
              "y": [bs],            (or an int label)
              "mask": [bs, nframes], 
              ... 
            }
        We only need "mu" from the encoder (like MotionCLIP does).
        """
        # The encoder returns { "mu": <tensor>, ... }
        out = self.encoder(motion_batch)
        # We'll consider out["mu"] as our 512-dim motion embedding
        return out["mu"]


###########################################
# 3) Simple Contrastive Loss (CLIP-like)  #
###########################################
class ContrastiveLoss(nn.Module):
    """
    We can keep it simple: 
      - Normalize embeddings
      - CrossEntropy with diagonal as positives
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, motion_emb, text_emb):
        # motion_emb: [bs, dim]
        # text_emb:   [bs, dim]
        motion_emb = F.normalize(motion_emb, dim=-1)
        text_emb   = F.normalize(text_emb,   dim=-1)

        logits_per_motion = motion_emb @ text_emb.t() / self.temperature
        logits_per_text   = logits_per_motion.t()

        batch_size = motion_emb.shape[0]
        ground_truth = torch.arange(batch_size, device=motion_emb.device)

        # standard cross-entropy losses
        loss_motion = F.cross_entropy(logits_per_motion, ground_truth)
        loss_text   = F.cross_entropy(logits_per_text,   ground_truth)
        total_loss = (loss_motion + loss_text) / 2.0
        return total_loss


#########################################################
# 4) Combined Model: MotionCLIP-style (Text + Motion)   #
#########################################################
class MotionClipModel(nn.Module):
    def __init__(
        self,
        # Motion encoder options
        njoints=24,
        nfeats=6,
        num_frames=60,
        latent_dim=512,
        # CLIP text encoder
        clip_model_name="ViT-B/32",
        device="cuda"
    ):
        super().__init__()
        self.device = device

        # 4A) Motion Encoder (Transformer) 
        self.motion_encoder = MotionTransformerEncoder(
            njoints=njoints,
            nfeats=nfeats,
            num_frames=num_frames,
            latent_dim=latent_dim
        )

        # 4B) CLIP Text Encoder (frozen)
        self.text_encoder = ClipTextEncoder(
            clip_model_name=clip_model_name, 
            device=device
        )

        # 4C) Contrastive Loss
        self.contrastive_loss = ContrastiveLoss(temperature=0.07)

    def forward(self, motion_batch, text_tokenized):
        """
        motion_batch: dictionary containing x, y, mask, etc. (see above)
        text_tokenized: the result of clip.tokenize(...) 
                        or something already on device
        """
        # Move to device if not already
        # typical pattern: for k,v in motion_batch.items(): ...
        motion_batch_on_device = {}
        for k, v in motion_batch.items():
            if torch.is_tensor(v):
                motion_batch_on_device[k] = v.to(self.device)
            else:
                motion_batch_on_device[k] = v

        # text is also a tensor typically
        text_tokenized = text_tokenized.to(self.device)

        # 4A) Motion Encoder forward => motion embedding
        motion_emb = self.motion_encoder(motion_batch_on_device)  # shape [bs, latent_dim]

        # 4B) Text Encoder forward => text embedding
        text_emb = self.text_encoder(text_tokenized)              # shape [bs, 512]

        # 4C) Contrastive Loss
        loss = self.contrastive_loss(motion_emb, text_emb)

        return {
            "loss": loss,
            "motion_emb": motion_emb,
            "text_emb": text_emb
        }
