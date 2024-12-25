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
# 3) MultiCaption Contrastive Loss #
###########################################
# class ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, motion_emb, text_emb):
#         # Debug original embeddings
#         print("\nContrastive Loss Debug:")
#         print(f"Original motion_emb - shape: {motion_emb.shape}")
#         print(f"min: {motion_emb.min().item():.4f}, max: {motion_emb.max().item()}, mean: {motion_emb.mean().item():.4f}")
#         print(f"Original text_emb - shape: {text_emb.shape}")
#         print(f"min: {text_emb.min().item():.4f}, max: {text_emb.max().item()}, mean: {text_emb.mean().item():.4f}")

#         # Normalize embeddings
#         motion_emb = F.normalize(motion_emb, dim=-1)
#         text_emb = F.normalize(text_emb, dim=-1)

#         # Debug normalized embeddings
#         print("\nAfter normalization:")
#         print(f"Motion embedding norms: {motion_emb.norm(dim=-1)}")
#         print(f"Text embedding norms: {text_emb.norm(dim=-1)}")

#         # Compute similarity matrices
#         logits_per_motion = torch.matmul(motion_emb, text_emb.t()) / self.temperature
#         logits_per_text = logits_per_motion.t()

#         # Debug logits
#         print("\nLogits debug:")
#         print(f"Logits shape: {logits_per_motion.shape}")
#         print(f"Logits values - min: {logits_per_motion.min().item():.4f}, max: {logits_per_motion.max().item()}")
#         print(f"Temperature: {self.temperature}")

#         # Ground truth labels
#         batch_size = motion_emb.size(0)
#         ground_truth = torch.arange(batch_size, device=motion_emb.device)
        
#         print(f"\nGround truth labels: {ground_truth}")
#         print(f"Logits per motion:\n{logits_per_motion}")
#         print(f"Logits per text:\n{logits_per_text}")

#         # Compute cross-entropy losses
#         loss_motion = F.cross_entropy(logits_per_motion, ground_truth)
#         loss_text = F.cross_entropy(logits_per_text, ground_truth)
#         total_loss = (loss_motion + loss_text) / 2.0

#         print(f"\nLosses:")
#         print(f"Motion loss: {loss_motion.item():.4f}")
#         print(f"Text loss: {loss_text.item():.4f}")
#         print(f"Total loss: {total_loss.item():.4f}")

#         return total_loss

class MultiCaption_ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, motion_embs, text_embs_list, motion_ids):
        device = motion_embs.device
        
        # 1. Normalize embeddings (add small epsilon for numerical stability)
        motion_embs = F.normalize(motion_embs + 1e-8, dim=-1)
        
        all_embeddings = [motion_embs]
        all_cluster_ids = [motion_ids]
        
        # 2. Process text embeddings
        for bid, text_embs in enumerate(text_embs_list):
            # Add small epsilon and normalize
            text_embs = F.normalize(text_embs + 1e-8, dim=-1)
            all_embeddings.append(text_embs)
            all_cluster_ids.append(torch.full((text_embs.shape[0],), motion_ids[bid], device=device))
        
        # 3. Concatenate all embeddings and cluster IDs
        all_embeddings = torch.cat(all_embeddings, dim=0)  # [total_items, latent_dim]
        all_cluster_ids = torch.cat(all_cluster_ids, dim=0)  # [total_items]
        
        # 4. Compute similarity with gradient clipping
        similarity = torch.clamp(
            torch.matmul(all_embeddings, all_embeddings.t()) / self.temperature,
            min=-100,
            max=100
        )
        
        # 5. Create cluster-based labels with handling for empty clusters
        labels = (all_cluster_ids.unsqueeze(0) == all_cluster_ids.unsqueeze(1)).float()
        
        # 6. Remove self-similarity and create mask for valid pairs
        mask = (1 - torch.eye(labels.shape[0], device=device))
        labels = labels * mask
        
        # Add small epsilon to avoid log(0)
        similarity = similarity.masked_fill(torch.eye(similarity.shape[0], device=device).bool(), -1e9)
        
        # 7. Compute loss with numerical stability
        log_probs = F.log_softmax(similarity, dim=-1)
        
        # 8. Compute loss only for valid pairs
        valid_pairs = (labels.sum(dim=1) > 0)
        if valid_pairs.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)
            
        loss = -(log_probs * labels)[valid_pairs].sum(dim=-1).mean()
        
        return loss

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

        # 3. Multi-caption Loss
        self.multicap_loss = MultiCaption_ContrastiveLoss(temperature=0.07)

    def encode_motion(self, motion_batch):
        # Move everything to device
        for k, v in motion_batch.items():
            if torch.is_tensor(v):
                motion_batch[k] = v.to(self.device)
        return self.motion_encoder(motion_batch)  # shape [bs, latent_dim]

    def encode_texts_for_one_item(self, captions):
        """
        captions: list of raw caption strings
        Return: a single Tensor [num_captions, latent_dim]
        """
        text_emb_list = []
        for cap in captions:
            tokens = clip.tokenize([cap]).to(self.device)  # shape [1, token_len]
            emb = self.text_encoder(tokens)                # shape [1, 512]
            text_emb_list.append(emb)
        if len(text_emb_list) == 0:
            # if no captions, return an empty tensor
            return torch.empty(0, 512, device=self.device)
        return torch.cat(text_emb_list, dim=0)  # [num_captions, 512]

    def compute_multicap_loss(self, motion_embs, text_embs_list, motion_ids):
        """
        Wrap your MultiCaption_ContrastiveLoss
        """
        return self.multicap_loss(motion_embs, text_embs_list, motion_ids)