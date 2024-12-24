import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: [max_len, batch_size, d_model]
        Returns:
            x: [max_len, batch_size, d_model] + positional encoding
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


# Only for ablation / not used in the final model
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1 / (lengths[..., None] - 1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # Add the time encoding
        x = x + time[..., None]
        return self.dropout(x)


class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, n_features, num_frames,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4,
                 dropout=0.1, activation="gelu", **kwargs):
        super().__init__()

        self.modeltype = modeltype
        self.n_features = n_features  # Now directly using n_features=263
        self.num_frames = num_frames
        self.latent_dim = latent_dim

        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.activation = activation

        # Input projection layer: maps from n_features to latent_dim
        self.input_projection = nn.Linear(self.n_features, self.latent_dim)

        # Positional Encoding
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                   nhead=self.num_heads,
                                                   dim_feedforward=self.ff_size,
                                                   dropout=self.dropout,
                                                   activation=self.activation)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer,
                                                         num_layers=self.num_layers)

    def forward(self, batch):
        """
        Args:
            batch: dict containing:
                - "x": [bs, max_len, 263] (Tensor)
                - "mask": [bs, max_len] (bool Tensor)
        Returns:
            motion_emb: [bs, latent_dim]
        """
        x, mask = batch["x"], batch["mask"]  # x: [bs, max_len, 263], mask: [bs, max_len]

        bs, max_len, D = x.shape
        assert D == self.n_features, f"Expected feature dimension {self.n_features}, got {D}"

        # Project input features to latent_dim
        x = self.input_projection(x)  # [bs, max_len, latent_dim]

        # Permute for Transformer: [max_len, bs, latent_dim]
        x = x.permute(1, 0, 2)

        # Apply positional encoding
        x = self.sequence_pos_encoder(x)  # [max_len, bs, latent_dim]

        # Create src_key_padding_mask for Transformer (True for padding)
        src_key_padding_mask = ~mask  # [bs, max_len]

        # Pass through Transformer Encoder
        transformer_output = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        # [max_len, bs, latent_dim]

        # Permute back to [bs, latent_dim, max_len]
        transformer_output = transformer_output.permute(1, 2, 0)  # [bs, latent_dim, max_len]

        # Pooling to get [bs, latent_dim]
        # Here, we use mean pooling over the temporal dimension
        motion_emb = transformer_output.mean(dim=2)  # [bs, latent_dim]

        return motion_emb


class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, modeltype, njoints, nfeats, num_frames, num_classes, translation, pose_rep, glob, glob_rot,
                 latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.1, activation="gelu",
                 ablation=None, **kargs):
        super().__init__()

        self.modeltype = modeltype
        self.njoints = njoints
        self.nfeats = nfeats
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        self.pose_rep = pose_rep
        self.glob = glob
        self.glob_rot = glob_rot
        self.translation = translation
        
        self.latent_dim = latent_dim
        
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.ablation = ablation

        self.activation = activation
                
        self.input_feats = self.njoints*self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            self.ztimelinear = nn.Linear(self.latent_dim + self.num_classes, self.latent_dim)

        self.actionBiases = nn.Parameter(torch.randn(1, self.latent_dim))

        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            self.sequence_pos_encoder = TimeEncoding(self.dropout)
        else:
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats)
        
    def forward(self, batch, use_text_emb=False):
        z, y, mask, lengths = batch["z"], batch["y"], batch["mask"], batch["lengths"]
        if use_text_emb:
            z = batch["clip_text_emb"]
        latent_dim = z.shape[1]
        bs, nframes = mask.shape
        njoints, nfeats = self.njoints, self.nfeats

        # only for ablation / not used in the final model
        if self.ablation == "zandtime":
            yoh = F.one_hot(y, self.num_classes)
            z = torch.cat((z, yoh), axis=1)
            z = self.ztimelinear(z)
            z = z[None]  # sequence of size 1
        else:
            # only for ablation / not used in the final model
            if self.ablation == "concat_bias":
                # sequence of size 2
                z = torch.stack((z, self.actionBiases[y]), axis=0)
            else:
                z = z[None]  # sequence of size 1  #

        timequeries = torch.zeros(nframes, bs, latent_dim, device=z.device)
        
        # only for ablation / not used in the final model
        if self.ablation == "time_encoding":
            timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        else:
            timequeries = self.sequence_pos_encoder(timequeries)
        
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        
        output = self.finallayer(output).reshape(nframes, bs, njoints, nfeats)
        
        # zero for padded area
        output[~mask.T] = 0
        output = output.permute(1, 2, 3, 0)

        if use_text_emb:
            batch["txt_output"] = output
        else:
            batch["output"] = output
        return batch
