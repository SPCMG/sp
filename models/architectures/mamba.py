import torch
import torch.nn as nn

class MambaEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        """
        Mamba-based Motion Encoder
        Args:
            input_size (int): Dimensionality of input features (e.g., 263).
            hidden_size (int): Dimensionality of hidden layers (e.g., 512).
            output_size (int): Dimensionality of the final output embedding (e.g., 512).
            num_layers (int): Number of layers in the GRU.
            dropout (float): Dropout rate for regularization.
        """
        super(MambaEncoder, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)  # Input projection layer

        # GRU for sequential processing
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Output projection layers
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),  # Combine bidirectional outputs
            nn.LayerNorm(hidden_size),               # Normalize features
            nn.LeakyReLU(0.2, inplace=True),         # Activation function
            nn.Linear(hidden_size, output_size)      # Project to final output size
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, inputs):
        """
        Forward pass for motion encoder.
        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).
                                   seq_len = 200, input_size = 263.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_size).
        """
        # Project input features to hidden size
        x = self.input_layer(inputs)  # (batch_size, seq_len, hidden_size)

        # Pass through GRU
        gru_out, _ = self.gru(x)  # (batch_size, seq_len, hidden_size * 2)

        # Mean pooling over the sequence dimension
        pooled_out = torch.mean(gru_out, dim=1)  # (batch_size, hidden_size * 2)

        # Project to the final output embedding
        output = self.output_layer(pooled_out)  # (batch_size, output_size)

        return output


# Example usage
if __name__ == "__main__":
    batch_size = 16
    seq_len = 200
    input_size = 263
    hidden_size = 512
    output_size = 512

    model = MambaEncoder(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=2,
        dropout=0.1
    )
    inputs = torch.rand(batch_size, seq_len, input_size)
    outputs = model(inputs)
    print("Output shape:", outputs.shape)  # Should print (batch_size, output_size)
