import torch
import torch.nn as nn
from einops import rearrange


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super(Linear, self).__init__()
        self.d_in = in_features
        self.d_out = out_features
        self.w = nn.Parameter(torch.empty((out_features, in_features), device=device, dtype=dtype))

        self.initialize_weights()
    
    def initialize_weights(self):
        var = 2 / (self.d_in + self.d_out)
        std = var ** 0.5
        nn.init.trunc_normal_(self.w, mean=0.0, std=std, a=-3 * std, b=3 * std)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.w.t()
    

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings        # Size of the vocabulary
        self.embedding_dim = embedding_dim          # Dimension of the embedding vectors, i.e., d_model
        self.embed_matrix = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        self.initialize_weights()
    
    def initialize_weights(self):
        var = 1
        std = 1
        nn.init.trunc_normal_(self.embed_matrix, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_matrix[token_ids]
    

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super(RMSNorm, self).__init__()
        self.d_model = d_model
        self.eps = eps
        self.gains = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        in_dype = x.dtype
        x = x.to(torch.float32)  # Convert to float32 for numerical stability (prevent overflow)

        norm = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        result = self.gains * (x / norm)

        return result.to(in_dype)
    

class SiLU(nn.Module):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)
    

# def SiLU(x: torch.Tensor) -> torch.Tensor:
#     return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()

        # d_ff = 8/3 * d_model is a common choice, also d_ff should be a multiple of 64 for efficient computation
        assert d_ff % 64 == 0, "d_ff must be a multiple of 64 for SwiGLU."

        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))

        self.silu = SiLU()

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x1 = x @ self.w1.t()
        x3 = x @ self.w3.t()
        x2 = self.silu(x1) * x3
        out = x2 @ self.w2.t()
        return out


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super(RotaryPositionalEmbedding, self).__init__()
        assert d_k % 2 == 0, "d_k must be even for Rotary Positional Embedding."

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, device=device).float() / d_k))
        position = torch.arange(0, max_seq_len, device=device).float()

        sinusoid_inp = torch.outer(position, inv_freq)                  # shape: [max_seq_len, d_k//2]

        self.register_buffer('sin', torch.sin(sinusoid_inp), persistent=False)  # shape: [max_seq_len, d_k//2]
        self.register_buffer('cos', torch.cos(sinusoid_inp), persistent=False)  # shape: [max_seq_len, d_k//2]

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # Get sin and cos values via advanced indexing
        # x: (..., seq_len, d_k)
        # token_positions: (..., seq_len)
        sin = self.sin[token_positions]  # shape: [seq_len, d_k//2]
        cos = self.cos[token_positions]  # shape: [seq_len, d_k//2]

        # Split x into even/odd parts
        x_even = x[..., 0::2]  # [..., seq_len, d_k//2]
        x_odd  = x[..., 1::2]

        # Apply rotation
        x_rotated_even = x_even * cos - x_odd * sin
        x_rotated_odd  = x_even * sin + x_odd * cos

        # Interleave even and odd
        x_out = torch.empty_like(x)
        x_out[..., 0::2] = x_rotated_even
        x_out[..., 1::2] = x_rotated_odd

        return x_out
    

class Softmax(nn.Module):
    def __init__(self, dim: int = -1):
        super(Softmax, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # subtract max for numerical stability
        x_max = x.max(dim=self.dim, keepdim=True).values
        x_exp = torch.exp(x - x_max)
        x_sum = x_exp.sum(dim=self.dim, keepdim=True)
        return x_exp / x_sum


if __name__ == "__main__":
    device = 'cuda'
    m = RotaryPositionalEmbedding(theta=10000, d_k=64, max_seq_len=512, device='cuda')
    x = torch.randn(2, 10, 64, device=device)  # [B, S, d_k]
    token_positions = torch.arange(10, device=device).unsqueeze(0).repeat(2, 1)  # [B, S]
    out = m(x, token_positions)
    print(out.shape)  # Should be [2, 10, 64]
    print(out.dtype, out.device)