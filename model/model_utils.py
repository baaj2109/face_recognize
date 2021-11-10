
import torch
from torch.nn import Module

class Flatten(Module):
    def forward(self, input: torch.tensor):
        return input.view(input.size(0), -1)

def l2_norm(input: torch.tensor ,axis: int = 1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output