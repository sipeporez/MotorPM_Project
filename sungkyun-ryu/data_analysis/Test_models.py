import torch 
import torch.nn as nn
import transform_fns as trans

class Fully_connected(nn.Module):
    def __init__(self, maxval, rate):
        super().__init__()
        self.maxval = nn.Parameter(torch.tensor(maxval, dtype=torch.float32))
        self.rate = nn.Parameter(torch.tensor(rate, dtype=torch.float32))

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            raise ValueError("Input x must be a PyTorch tensor.")
            
        standardized_freq = trans.standardized(y, self.maxval)
        weighted_arr = trans.weighted_function(standardized_freq, self.rate)
        expanded_arr = weighted_arr.repeat(36)

        x = x * expanded_arr
        
        return x