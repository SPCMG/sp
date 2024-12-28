import torch
import torch.nn as nn
from transformers import CLIPModel

class ClipTextEncoder(nn.Module):
    def __init__(self, pretrained_model_name="openai/clip-vit-base-patch32"):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(pretrained_model_name)
        
        # Freeze the vision encoder
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.visual_projection.parameters():
            param.requires_grad = False

        # Now only text_model and text_projection are trainable.

    def forward(self, input_ids, attention_mask):
        """
        input_ids: (batch, seq_len)
        attention_mask: (batch, seq_len)
        returns: text_emb (batch, hidden_dim)
        """
        text_emb = self.clip_model.get_text_features(
            input_ids=input_ids, 
            attention_mask=attention_mask
        )
        return text_emb
