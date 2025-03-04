import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPConfig

class ClipTextEncoder(nn.Module):
    def __init__(
        self,
        hf_pretrained_name="openai/clip-vit-base-patch32",
        pretrained_ckpt_path=None,
        dropout=0.1
    ):
        super().__init__()
        # 1) Load base CLIP architecture
        config = CLIPConfig.from_pretrained(hf_pretrained_name)
        self.clip_model = CLIPModel(config)

        # 2) Optionally load local checkpoint weights (if you have a custom .pt file)
        if pretrained_ckpt_path is not None:
            state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")
            self.clip_model.load_state_dict(state_dict, strict=False)

        # 3) Freeze vision model but keep text model trainable
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = False 
        for param in self.clip_model.visual_projection.parameters():
            param.requires_grad = False
        
        # Unfreeze text model for fine-tuning
        for param in self.clip_model.text_model.parameters():
            param.requires_grad = True

        # Print trainable text parameters for debugging
        for name, param in self.clip_model.named_parameters():
            if "text" in name:
                print(f"{name}: requires_grad = {param.requires_grad}")

        # 4) Dropout layer to apply on the text embeddings
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        """ Returns text embeddings from the CLIP text encoder """
        text_emb = self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return self.dropout(text_emb)

    def encode_text(self, text_batch):
        """ Alternative forward pass using raw text_model """
        output = self.clip_model.text_model(**text_batch)
        return output.pooler_output
