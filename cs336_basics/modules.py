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