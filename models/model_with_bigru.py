from models.losses import ContrastiveLoss
from models.architectures.transformer import TransformerEncoder
from models.architectures.mamba import MambaEncoder
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from tune.humanml3d.tokenizer import SimpleTokenizer
from models.architectures.bigru import TextEncoderBiGRUCo, MotionEncoderBiGRUCo

class MotionTextModel(nn.Module):
    def __init__(self, config):
        super(MotionTextModel, self).__init__()
        self.text_encoder_type = config.model.text_encoder
        self.motion_encoder_type = config.model.motion_encoder
        self.max_text_length = config.data.max_text_length
        self.device = config.device

        # Initialize the text encoder
        if self.text_encoder_type == "clip":
            self.clip_model, _ = clip.load(config.model.clip_model_name, device=self.device, jit=False)
            self.text_encoder = self.clip_model
        elif self.text_encoder_type == "laclip":
            checkpoint = torch.load(config.model.laclip_ckpt_path, map_location=self.device)
            self.clip_model.load_state_dict(checkpoint["state_dict"], strict=False)
            self.text_encoder = self.clip_model
        elif self.text_encoder_type == "motionlaclip":
            checkpoint = torch.load(config.model.motionlaclip_ckpt_path, map_location=self.device)
            self.clip_model.load_state_dict(checkpoint, strict=False)
            self.text_encoder = self.clip_model
        elif self.text_encoder_type == "motionlaclipplus":
            checkpoint = torch.load(config.model.motionlaclipplus_ckpt_path, map_location=self.device)
            self.clip_model.load_state_dict(checkpoint, strict=False)
            self.text_encoder = self.clip_model
        elif self.text_encoder_type == "bigru":
            self.text_encoder = TextEncoderBiGRUCo(
                word_size=300,
                pos_size=15,
                hidden_size=512,
                output_size=512,
                device=self.device
            )
        else:
            raise ValueError("Unsupported text encoder type")

        # Freeze the text encoder parameters for non-BiGRU encoders
        if self.text_encoder_type != "bigru":
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
        elif self.motion_encoder_type == "bigru":
            self.motion_encoder = MotionEncoderBiGRUCo(
                input_size=512,
                hidden_size=1024,
                output_size=512,
                device=self.device
            )
        else:
            raise ValueError("Unsupported motion encoder type")

        # Initialize the loss
        self.loss = ContrastiveLoss(temperature=config.loss.temperature, margin=config.loss.margin)

    def forward(self, text_inputs, motion_inputs, labels):
        # Tokenize and encode text inputs
        if self.text_encoder_type == "bigru":
            word_emb, pos_ohot, _, cap_lens = text_inputs
            word_emb = word_emb.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            text_embeds = self.text_encoder(word_emb, pos_ohot, cap_lens)
        else:
            text_inputs = clip.tokenize(text_inputs).to(self.device)
            text_embeds = self.text_encoder.encode_text(text_inputs).float()

        # Encode motion inputs
        if self.motion_encoder_type == "bigru":
            motions, m_lens = motion_inputs
            motions = motions.detach().to(self.device).float()
            m_lens = m_lens // 4
            motion_embeds = self.motion_encoder(motions, m_lens)
        else:
            motion_embeds = self.motion_encoder(motion_inputs)

        text_embeds = F.normalize(text_embeds, p=2, dim=1)
        motion_embeds = F.normalize(motion_embeds, p=2, dim=1)

        # Compute the loss
        loss = self.loss(text_embeds, motion_embeds, labels)
        return loss

    def encode_motion(self, motion_inputs):
        """
        Encodes motion inputs into motion embeddings.
        """
        with torch.no_grad():
            if self.motion_encoder_type == "bigru":
                motions, m_lens = motion_inputs
                motions = motions.detach().to(self.device).float()
                m_lens = m_lens // config.model.unit_length
                motion_embeds = self.motion_encoder(motions, m_lens)
            else:
                motion_embeds = self.motion_encoder(motion_inputs)
            motion_embeds = F.normalize(motion_embeds, p=2, dim=1)
        return motion_embeds

    def encode_text(self, text_inputs):
        """
        Encodes text inputs into text embeddings.
        """
        with torch.no_grad():
            if self.text_encoder_type == "bigru":
                word_emb, pos_ohot, _, cap_lens = text_inputs
                word_emb = word_emb.detach().to(self.device).float()
                pos_ohot = pos_ohot.detach().to(self.device).float()
                text_embeds = self.text_encoder(word_emb, pos_ohot, cap_lens)
            else:
                if isinstance(text_inputs, list) and isinstance(text_inputs[0], str):
                    text_inputs = clip.tokenize(text_inputs).to(self.device)
                text_embeds = self.text_encoder.encode_text(text_inputs).float()
            text_embeds = F.normalize(text_embeds, p=2, dim=1)
        return text_embeds
