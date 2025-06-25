import torch
import torch.nn as nn



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
    

# class SiLU(nn.Module):
#     def __init__(self):
#         super(SiLU, self).__init__()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x * torch.sigmoid(x)
    

def SiLU(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super(SwiGLU, self).__init__()
        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), device=device, dtype=dtype))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), device=device, dtype=dtype))

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.w1)
        nn.init.xavier_uniform_(self.w2)
        nn.init.xavier_uniform_(self.w3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, d_model)
        x1 = x @ self.w1.t()
        x3 = x @ self.w3.t()
        x2 = SiLU(x1) * x3
        out = x2 @ self.w2.t()
        return out

