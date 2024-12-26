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
# 3) Motion Text Alignment Loss #
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

# class MultiCaption_ContrastiveLoss(nn.Module):
#     def __init__(self, temperature=0.07, margin=0.2):
#         super().__init__()
#         self.temperature = temperature
#         self.margin = margin

#     def forward(self, motion_embs, text_embs_list, motion_ids):
#         device = motion_embs.device
        
#         # 1. Normalize all embeddings
#         motion_embs = F.normalize(motion_embs + 1e-8, dim=-1)
        
#         all_embeddings = [motion_embs]
#         all_cluster_ids = [motion_ids]
        
#         # 2. Process text embeddings
#         for bid, text_embs in enumerate(text_embs_list):
#             text_embs = F.normalize(text_embs + 1e-8, dim=-1)
#             all_embeddings.append(text_embs)
#             all_cluster_ids.append(torch.full((text_embs.shape[0],), motion_ids[bid], device=device))
        
#         # 3. Concatenate all embeddings and cluster IDs
#         all_embeddings = torch.cat(all_embeddings, dim=0)    # [total_items, latent_dim]
#         all_cluster_ids = torch.cat(all_cluster_ids, dim=0)  # [total_items]
        
#         # 4. Compute similarity matrix
#         similarity = torch.matmul(all_embeddings, all_embeddings.t()) / self.temperature
        
#         # 5. Create positive/negative masks
#         labels = (all_cluster_ids.unsqueeze(0) == all_cluster_ids.unsqueeze(1)).float()
#         identity_mask = torch.eye(labels.shape[0], device=device).bool()
        
#         # Separate positive and negative pairs
#         positive_mask = (labels > 0.5) & (~identity_mask)  # Same motion/caption pairs, exclude self
#         negative_mask = (labels < 0.5) & (~identity_mask)  # Different motion/caption pairs
        
#         # 6. Compute InfoNCE loss with margin
#         total_loss = torch.tensor(0.0, device=device)
#         total_items = 0
        
#         for i in range(len(all_embeddings)):
#             positives = similarity[i][positive_mask[i]]
#             negatives = similarity[i][negative_mask[i]]
            
#             if len(positives) > 0 and len(negatives) > 0:
#                 # InfoNCE part: Pull positives together
#                 numerator = torch.exp(positives)
#                 denominator = numerator.sum() + torch.exp(negatives).sum()
#                 infonce_loss = -torch.log(numerator/denominator).mean()
                
#                 # Margin part: Push negatives apart with minimum margin
#                 margin_loss = torch.max(
#                     torch.zeros_like(negatives),
#                     negatives + self.margin  # negative similarities should be less than -margin
#                 ).mean()
                
#                 # Combine losses
#                 total_loss += infonce_loss + margin_loss
#                 total_items += 1
        
#         if total_items == 0:
#             return torch.tensor(0.0, device=device, requires_grad=True)
            
#         avg_loss = total_loss / total_items
        
#         # For monitoring
#         with torch.no_grad():
#             pos_sim = similarity[positive_mask].mean() if positive_mask.sum() > 0 else torch.tensor(0.0)
#             neg_sim = similarity[negative_mask].mean() if negative_mask.sum() > 0 else torch.tensor(0.0)
#             print(f"\nSimilarities - Positive: {pos_sim:.4f}, Negative: {neg_sim:.4f}")
#             print(f"Loss: {avg_loss:.4f}")
            
#             # Additional monitoring of margin violations
#             if negative_mask.sum() > 0:
#                 margin_violations = (similarity[negative_mask] > -self.margin).float().mean()
#                 print(f"Margin violations: {margin_violations:.2%}")
        
#         return avg_loss

class MotionTextAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07, margin=0.2):
        """
        Args:
            temperature: For scaling similarity in softmax or InfoNCE.
            margin:      Negative pairs must stay below -margin, penalizing those above.
        """
        super().__init__()
        self.temperature = temperature
        self.margin = margin

    def forward(self, motion_embs, text_embs_list, shuffled_text_embs_list, motion_ids):
        """
        motion_embs:       [num_motions, emb_dim]   Each row is a motion embedding.
        text_embs_list:    list of Tensors, each [num_captions_i, emb_dim] for motion i.
        shuffled_text_embs_list (optional): same shape as text_embs_list but shuffled.
        motion_ids:        [num_motions], cluster IDs for each motion i.

        We will re-map motion_ids if needed to ensure shuffled texts get separate IDs.
        """
        device = motion_embs.device

        # ------------------------------------------------------------
        # 1) Normalize all motion embeddings (adds tiny epsilon to avoid zero vector)
        # ------------------------------------------------------------
        motion_embs = F.normalize(motion_embs + 1e-8, dim=-1)

        # We'll build up these lists, then concatenate
        all_embeddings = []
        all_cluster_ids = []

        # First, add the motion embeddings
        all_embeddings.append(motion_embs)         # shape [num_motions, emb_dim]
        all_cluster_ids.append(motion_ids)         # shape [num_motions]

        # ------------------------------------------------------------
        # 2) Process original text embeddings
        #    Each text emb is assigned the SAME cluster as the motion
        # ------------------------------------------------------------
        for i, text_embs in enumerate(text_embs_list):
            # Normalize
            text_embs = F.normalize(text_embs + 1e-8, dim=-1)

            # Same cluster as motion i => motion_ids[i]
            cluster_id_i = motion_ids[i]  # e.g., 0,1,2,... for each motion

            # Repeat that cluster ID for each caption
            cluster_vec = torch.full(
                (text_embs.shape[0],),
                cluster_id_i,
                device=device,
                dtype=motion_ids.dtype
            )

            all_embeddings.append(text_embs)
            all_cluster_ids.append(cluster_vec)

        # ------------------------------------------------------------
        # 3) Process shuffled text embeddings
        #    Assign a NEW cluster ID => motion_ids[i] + 100000
        # ------------------------------------------------------------
        if shuffled_text_embs_list is not None:
            for i, shuf_text_embs in enumerate(shuffled_text_embs_list):
                # Normalize
                shuf_text_embs = F.normalize(shuf_text_embs + 1e-8, dim=-1)

                # Use a large offset so it's definitely different from original cluster
                new_cluster_id = motion_ids[i] + 100000
                new_cluster_vec = torch.full(
                    (shuf_text_embs.shape[0],),
                    new_cluster_id,
                    device=device,
                    dtype=motion_ids.dtype
                )

                all_embeddings.append(shuf_text_embs)
                all_cluster_ids.append(new_cluster_vec)

        # ------------------------------------------------------------
        # 4) Concatenate all embeddings & cluster IDs
        # ------------------------------------------------------------
        all_embeddings = torch.cat(all_embeddings, dim=0)     # shape [total_items, emb_dim]
        all_cluster_ids = torch.cat(all_cluster_ids, dim=0)   # shape [total_items]

        # ------------------------------------------------------------
        # 5) Compute similarity matrix
        # ------------------------------------------------------------
        similarity = torch.matmul(all_embeddings, all_embeddings.t()) / self.temperature
        num_items = all_embeddings.shape[0]

        # ------------------------------------------------------------
        # 6) Create positive/negative masks
        #    labels[i,j] = 1 if same cluster ID, else 0
        # ------------------------------------------------------------
        labels = (all_cluster_ids.unsqueeze(0) == all_cluster_ids.unsqueeze(1)).float()

        # Remove diagonal (self-comparison)
        identity_mask = torch.eye(num_items, device=device).bool()
        positive_mask = (labels > 0.5) & (~identity_mask)
        negative_mask = (labels < 0.5) & (~identity_mask)

        # ------------------------------------------------------------
        # 7) Compute InfoNCE with margin across all items
        # ------------------------------------------------------------
        total_loss = torch.tensor(0.0, device=device)
        total_count = 0

        for i in range(num_items):
            pos_indices = positive_mask[i]  # which j are positives for i
            neg_indices = negative_mask[i]  # which j are negatives for i

            positives = similarity[i][pos_indices]
            negatives = similarity[i][neg_indices]

            if positives.numel() > 0 and negatives.numel() > 0:
                # InfoNCE: pull positives up relative to negatives
                numerator = torch.exp(positives)
                denominator = numerator.sum() + torch.exp(negatives).sum()
                infonce_loss = -torch.log(numerator / denominator).mean()

                # Margin: push negatives below -margin
                margin_loss = torch.max(
                    torch.zeros_like(negatives),  # no penalty if neg < -margin
                    negatives + self.margin       # penalty if similarity is above -margin
                ).mean()

                total_loss += infonce_loss + margin_loss
                total_count += 1

        if total_count == 0:
            # Edge case: no valid pairs at all
            return torch.tensor(0.0, device=device, requires_grad=True)

        final_loss = total_loss / total_count

        # ------------------------------------------------------------
        # 8) Optional Monitoring
        # ------------------------------------------------------------
        with torch.no_grad():
            pos_sim = similarity[positive_mask].mean() if positive_mask.any() else 0
            neg_sim = similarity[negative_mask].mean() if negative_mask.any() else 0

            print(f"\nPos Sim = {pos_sim:.4f} | Neg Sim = {neg_sim:.4f}")
            print(f"Loss = {final_loss:.4f}")

            # Margin violation: negative pairs that are above -margin
            if negative_mask.any():
                margin_violations = (similarity[negative_mask] > -self.margin).float().mean()
                print(f"Margin violations = {margin_violations:.2%}")

        return final_loss

#########################################################
# 4) Combined MotionClipModel (Motion + Text)         #
#########################################################
class MotionClipModel(nn.Module):
    """
    Combined MotionCLIP model with motion encoder, text encoder, and alignment loss.
    """
    def __init__(self, njoints=22, n_feats=263, num_frames=196, latent_dim=512,
                 clip_model_name="ViT-B/32", device="cuda"):
        super().__init__()
        self.device = device

        # Motion Encoder (Transformer)
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

        # Text Encoder (CLIP)
        self.text_encoder = ClipTextEncoder(
            clip_model_name=clip_model_name, 
            device=device
        )

        # Alignment Loss
        self.motion_text_alignment_loss = MotionTextAlignmentLoss(
            temperature=0.07,
            margin=0.2
        )

    def encode_motion(self, motion_batch):
        for k, v in motion_batch.items():
            if torch.is_tensor(v):
                motion_batch[k] = v.to(self.device)
        return self.motion_encoder(motion_batch)

    def encode_texts(self, captions):
        """
        captions: list of raw text strings
        Returns: Tensor [num_captions, latent_dim]
        """
        text_embs = [self.text_encoder(clip.tokenize([cap]).to(self.device)) for cap in captions]
        return torch.cat(text_embs, dim=0) if text_embs else torch.empty(0, 512, device=self.device)

    def compute_motion_text_alignment_loss(self, motion_embs, text_embs_list, shuffled_text_embs_list, motion_ids):
        """
        Compute loss for motion-text alignment.
        """
        return self.motion_text_alignment_loss(
            motion_embs, text_embs_list, shuffled_text_embs_list, motion_ids
        )