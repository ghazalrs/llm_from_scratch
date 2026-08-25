import torch.nn as nn
import torch


# Each row is a token's embedding vector (already embedded, 3 dims here for simplicity)
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2


# Implementing a compact SelfAttention class
class SelfAttention(nn.Module):

    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()

        # Initialize the three weight matrices
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        
        # Compute the query, key, and value vectors
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Compute the unnormalized attention scores by computing the dot product between the query and each key vector
        attn_scores = queries @ keys.T

        # Compute the attention weights 
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)

        # Compute the context vectors
        context_vec = attn_weights @ values
        return context_vec

torch.manual_seed(789)
sa = SelfAttention(d_in, d_out)
print(sa(inputs))