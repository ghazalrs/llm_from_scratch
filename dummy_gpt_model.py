import torch
import torch.nn as nn


class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        # Use a placeholder for TransformerBlock
        self.trf_blocks = nn.Sequential(
            *[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        
        # Use a placeholder for LayerNorm
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # A simple placeholder

    def forward(self, x):
        # This block does nothing and just returns its input.
        return x


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        # Small number to prevent division by zero
        self.eps = 1e-5

        # Learnable multiplier for each feature
        self.scale = nn.Parameter(torch.ones(emb_dim))

        # Learnable offset for each feature
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):

        # Mean across features
        mean = x.mean(dim=-1, keepdim=True)

        # Variance across features
        var = x.var(
            dim=-1,
            keepdim=True,
            unbiased=False
        )

        # Normalize to approximately mean=0, variance=1
        norm_x = (x - mean) / torch.sqrt(var + self.eps)

        # Allow the model to learn how to rescale/reposition features
        return self.scale * norm_x + self.shift