from models.losses import ContrastiveLoss
from models.architectures.transformer import TransformerEncoder
from models.architectures.mamba import MambaEncoder
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from tune.humanml3d.tokenizer import SimpleTokenizer

class MotionTextModel(nn.Module):
    def __init__(self, config):
        super(MotionTextModel, self).__init__()
        self.text_encoder_type = config.model.text_encoder
        self.motion_encoder_type = config.model.motion_encoder
        self.max_text_length = config.data.max_text_length
        self.device = config.device

        # Initialize the text encoder
        self.clip_model, _ = clip.load(config.model.clip_model_name, device=self.device, jit=False)
        if self.text_encoder_type == "clip":
            self.text_encoder = self.clip_model
        elif self.text_encoder_type == "laclip":
            # Load the LaCLIP weights from the checkpoint
            checkpoint = torch.load(config.model.laclip_ckpt_path, map_location=self.device)
            self.clip_model.load_state_dict(checkpoint["state_dict"], strict=False)
            self.text_encoder = self.clip_model
        elif self.text_encoder_type == "motionlaclip":
            # Load the MotionLaCLIP weights from the checkpoint
            checkpoint = torch.load(config.model.motionlaclip_ckpt_path, map_location=self.device)
            self.clip_model.load_state_dict(checkpoint, strict=False)
            self.text_encoder = self.clip_model
        elif self.text_encoder_type == "motionlaclipplus":
            # Load the MotionLaCLIP+ weights from the checkpoint
            checkpoint = torch.load(config.model.motionlaclipplus_ckpt_path, map_location=self.device)
            self.clip_model.load_state_dict(checkpoint, strict=False)
            self.text_encoder = self.clip_model
        else:
            raise ValueError("Unsupported text encoder type")
       

        # Freeze the text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Initialize the motion encoder
        if self.motion_encoder_type == "transformer":
            self.motion_encoder = TransformerEncoder(
                input_size=config.data.n_feats,
                hidden_size=config.model.latent_dim,
                output_size=config.model.latent_dim,
                num_layers=config.model.num_transformer_layers,
                num_heads=config.model.num_heads,
                ff_size=config.model.ff_size,
                dropout=config.model.dropout,
                activation=config.model.activation
            )
        elif self.motion_encoder_type == "mamba":
            self.motion_encoder = MambaEncoder(
                input_size=config.data.n_feats,
                hidden_size=config.model.latent_dim,
                output_size=config.model.latent_dim,
                num_layers=config.model.num_mamba_layers,  
                dropout=config.model.dropout         
            )
        else:
            raise ValueError("Unsupported motion encoder type")

        # Initialize the loss
        self.loss = ContrastiveLoss(temperature=config.loss.temperature, margin=config.loss.margin)

    def forward(self, text_inputs, motion_inputs, labels):
        # Tokenize and encode text inputs
        text_inputs = clip.tokenize(text_inputs).to(self.device)
        text_embeds = self.text_encoder.encode_text(text_inputs).float()  # Use encode_text explicitly

        # Encode motion inputs
        motion_embeds = self.motion_encoder(motion_inputs)

        text_embeds = F.normalize(text_embeds, p=2, dim=1)
        motion_embeds = F.normalize(motion_embeds, p=2, dim=1)

        # Compute the loss
        loss = self.loss(text_embeds, motion_embeds, labels)
        return loss
