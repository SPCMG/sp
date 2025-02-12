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
        :param hf_pretrained_name: Name of the Hugging Face CLIP variant, e.g. "openai/clip-vit-base-patch32".
        :param pretrained_ckpt_path: Path to a local .pt checkpoint (e.g. laion400m_laclip.pt).
                                     If provided, we'll load those weights into the HF CLIP model.
        :param dropout: Dropout ratio to apply on the final text embedding.
        """
        super().__init__()
        # 1) Load base CLIP architecture
        config = CLIPConfig.from_pretrained(hf_pretrained_name)
        self.clip_model = CLIPModel(config)

        # 2) Optionally load local checkpoint weights (if you have a custom .pt file)
        if pretrained_ckpt_path is not None:
            state_dict = torch.load(pretrained_ckpt_path, map_location="cpu")
            # We set strict=False in case the checkpoint doesn't match 1:1 with the HF architecture
            self.clip_model.load_state_dict(state_dict, strict=False)

        # 3) Freeze vision if you only want to finetune text
        for param in self.clip_model.vision_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.visual_projection.parameters():
            param.requires_grad = False

        # 4) Dropout layer to apply on the text embeddings
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask):
        """
        Returns [batch_size, hidden_dim] text embeddings from the CLIP text encoder.
        """
        # CLIP provides a convenience method for text features
        text_emb = self.clip_model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Apply dropout
        text_emb = self.dropout(text_emb)
        return text_emb

    def encode_text(self, text_batch):
        """
        An alternative forward pass that uses the raw text_model. 
        This might be useful if you need the raw outputs from CLIP’s text model.
        """
        output = self.clip_model.text_model(**text_batch)
        # output is a BaseModelOutputWithPooling containing (last_hidden_state, pooler_output)
        pooled_output = output.pooler_output
        return pooled_output
