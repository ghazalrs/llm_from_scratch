import math
import torch
import torch.nn as nn
from multi_head_attention import MultiHeadAttention


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Token and positional embeddings
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # Stack of Transformer blocks
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        # Final normalization and output projection
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"],
            cfg["vocab_size"],
            bias=False
        )

    def forward(self, in_idx):
        # in_idx shape: [batch_size, seq_len]
        _, seq_len = in_idx.shape

        # Convert token IDs into embeddings
        tok_embeds = self.tok_emb(in_idx)

        # Create positional embeddings
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )

        # Combine token and positional information
        x = tok_embeds + pos_embeds

        # Apply embedding dropout
        x = self.drop_emb(x)

        # Pass through Transformer blocks
        x = self.trf_blocks(x)

        # Final normalization
        x = self.final_norm(x)

        # Convert embeddings into vocabulary logits
        logits = self.out_head(x)

        return logits


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Multi-head self-attention
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"]
        )

        # Feed-forward network
        self.ff = FeedForward(cfg)

        # Layer normalization before each sublayer
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])

        # Dropout before shortcut addition
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        # Attention block with shortcut connection
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        # Feed-forward block with shortcut connection
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # GELU activation approximation
        return 0.5 * x * (
            1 + torch.tanh(
                math.sqrt(2.0 / math.pi)
                * (x + 0.044715 * torch.pow(x, 3))
            )
        )


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Expand features, apply GELU, then project back
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )

    def forward(self, x):
        return self.layers(x)


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        # Prevent division by zero
        self.eps = 1e-5

        # Learnable scale and shift
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # Calculate mean and variance across features
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False
        )

        # Normalize features
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Apply learnable scale and shift
        return self.scale * norm_x + self.shift
