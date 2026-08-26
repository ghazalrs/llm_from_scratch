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

# Compute the unnormalized attention scores by computing the dot product between the query  and all other input tokens
attn_scores = torch.empty(6, 6)

for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)

print(attn_scores)

# We can achieve the same as above more efficiently via matrix multiplication
attn_scores = inputs @ inputs.T
print(attn_scores)

# Normalize each row so that the values in each row sum to 1
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# Compute all context vectors
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)