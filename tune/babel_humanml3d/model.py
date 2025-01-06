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
        """
        hf_pretrained_name: Which Hugging Face CLIP variant to load for the skeleton architecture.
        pretrained_ckpt_path: Path to your local .pt checkpoint (e.g. laion400m_laclip.pt).
                              If provided, we'll overwrite the HF weights with these.
        """
        super().__init__()
        # 1) Load base architecture (ensures correct shapes/layers)
        config = CLIPConfig.from_pretrained(hf_pretrained_name)
        self.clip_model = CLIPModel(config)

        # 2) Overwrite weights with your .pt, if provided
        if pretrained_ckpt_path is not None:
            state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")
            self.clip_model.load_state_dict(state_dict, strict=False)

        # 2) Overwrite weights with your .pt, if provided
        # if pretrained_ckpt_path is not None:
        #     checkpoint = torch.load(pretrained_ckpt_path, map_location="cpu")
        #     # Determine if checkpoint is nested
        #     if 'clip_model_state_dict' in checkpoint:
        #         state_dict = checkpoint['clip_model_state_dict']
        #     elif 'state_dict' in checkpoint:
        #         # Assuming your checkpoint has nested 'clip_model' keys
        #         state_dict = {k.replace('clip_model.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('clip_model.')}
        #     else:
        #         state_dict = checkpoint  # Fallback

        #     # Load state_dict into clip_model
        #     self.clip_model.load_state_dict(state_dict, strict=False)

        # 3) Freeze vision if you only want to finetune text
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.visual_projection.parameters():
            param.requires_grad = False

        # Add dropout layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        """
        Returns text embeddings from the text encoder.
        """
        text_emb = self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Apply dropout to the text embeddings
        text_emb = self.dropout(text_emb)

        return text_emb
    
    def encode_text(self, text):
        output = self.clip_model.text_model(**text)
        pooled_output = output[1]
        return pooled_output