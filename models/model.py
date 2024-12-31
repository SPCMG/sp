from losses import ContrastiveLoss
from architectures.transformer import TransformerEncoder
from architectures.mamba import MambaEncoder
import clip
import torch

class MotionTextModel(nn.Module):
    def __init__(self, config):
        super(MotionTextModel, self).__init__()
        self.text_encoder_type = config.model.text_encoder
        self.motion_encoder_type = config.model.motion_encoder
        self.device = config.device

        # Initialize the text encoder
        if self.text_encoder_type == "clip":
            self.clip_model, _ = clip.load(config.model.clip_model_name, device=self.device, jit=False)
            self.text_encoder = self.clip_model.encode_text
        elif self.text_encoder_type == "laclip":
            self.text_encoder = torch.load(config.model.laclip_ckpt_path, map_location=self.device)
        elif self.text_encoder_type == "motionlaclip":
            self.text_encoder = torch.load(config.model.motionlaclip_ckpt_path, map_location=self.device)
        else:
            raise ValueError("Unsupported text encoder type")

        # Freeze the text encoder parameters
        for param in self.text_encoder.parameters():
            param.requires_grad = False

        # Initialize the motion encoder
        if self.motion_encoder_type == "transformer":
            self.motion_encoder = TransformerEncoder(
                input_size=config.data.n_feats,
                latent_dim=config.model.latent_dim,
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
        if self.text_encoder_type == "clip":
            text_inputs = clip.tokenize(text_inputs).to(self.device)
            text_embeds = self.text_encoder(text_inputs).float()
        else:
            text_embeds = self.text_encoder(text_inputs)

        # Encode motion inputs
        motion_embeds = self.motion_encoder(motion_inputs)

        # Compute the loss
        loss = self.loss(text_embeds, motion_embeds, labels)
        return loss