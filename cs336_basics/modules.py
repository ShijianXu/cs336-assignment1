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