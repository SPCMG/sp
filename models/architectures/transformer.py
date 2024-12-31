import torch
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, num_heads=8, ff_size=1024, dropout=0.1):
        """
        Transformer-based Motion Encoder
        Args:
            input_size (int): Dimensionality of input features (e.g., 263).
            hidden_size (int): Dimensionality of the transformer model (e.g., 512).
            output_size (int): Dimensionality of the final output embedding (e.g., 512).
            num_layers (int): Number of transformer encoder layers.
            num_heads (int): Number of attention heads in each layer.
            ff_size (int): Dimensionality of feedforward layers.
            dropout (float): Dropout rate for regularization.
        """
        super(TransformerEncoder, self).__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)  # Project input features to transformer hidden size
        self.positional_encoding = PositionalEncoding(hidden_size, max_len=200)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation="gelu"
            ),
            num_layers=num_layers
        )
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, inputs):
        """
        Forward pass for motion encoder.
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
                                   seq_len = 200, input_size = 263.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # inputs: (batch_size, seq_len, input_size)
        batch_size, seq_len, _ = inputs.shape

        # Project input to hidden size
        x = self.input_proj(inputs)  # (batch_size, seq_len, hidden_size)

        # Add positional encoding
        x = x + self.positional_encoding(torch.arange(seq_len).to(inputs.device))

        # Transpose for transformer: (seq_len, batch_size, hidden_size)
        x = x.permute(1, 0, 2)

        # Pass through transformer encoder
        encoded = self.encoder(x)  # (seq_len, batch_size, hidden_size)

        # Aggregate the sequence to a single embedding (mean pooling)
        pooled = torch.mean(encoded, dim=0)  # (batch_size, hidden_size)

        # Project to final output size
        output = self.output_proj(pooled)  # (batch_size, output_size)

        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200):
        """
        Positional encoding for adding positional information to the transformer input.
        Args:
            d_model (int): Dimensionality of the model (e.g., 512).
            max_len (int): Maximum sequence length (e.g., 200).
        """
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0)  # Shape (1, max_len, d_model)

    def forward(self, pos):
        return self.pe[:, pos, :]


# Example usage
if __name__ == "__main__":
    batch_size = 16
    seq_len = 200
    input_size = 263
    hidden_size = 512
    output_size = 512

    model = TransformerEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=4,
        num_heads=8,
        ff_size=1024,
        dropout=0.1
    )
    inputs = torch.rand(batch_size, seq_len, input_size)
    outputs = model(inputs)
    print("Output shape:", outputs.shape)  # Should print (batch_size, output_size)
