import torch
import torch.nn as nn

class Largest_downscale(nn.Module): 
    def __init__(self, dim = 5000): 
        super().__init__()
        self.dim = dim 

    def forward(self, x): 
        top_5000_values, top_5000_indices = torch.topk(x, dim, sorted=False)
        
        return top_5000_values[torch.argsort(top_5000_indices)]
        