
import math
import torch
from torch import nn
from torch.nn import Parameter, Module
from .model_utils import l2_norm


class ArcFace(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599    
    def __init__(self, 
                 embedding_size: int = 512,
                 classnum: int = 51332,
                 s: float = 64.,
                 m: float = 0.5):
        super(ArcFace, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size,classnum))
        nn.init.xavier_uniform_(self.kernel)
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        # the margin value, default is 0.5
        self.m = m 
        # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.s = s 
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
        
    def forward(self, 
                embbedings: torch.tensor, 
                label: int):
        # weights norm
        
        nB = len(embbedings)
        kernel_norm = l2_norm(self.kernel, axis = 0) # normalize for each column
        
        # cos(theta+m)        
        cos_theta = torch.mm(embbedings, kernel_norm)
#         output = torch.mm(embbedings,kernel_norm)
        
        cos_theta = cos_theta.clamp(-1, 1) # for numerical stability
        cos_theta_2 = torch.pow(cos_theta, 2)
        sin_theta_2 = 1 - cos_theta_2
        sin_theta = torch.sqrt(sin_theta_2)
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        cond_v = cos_theta - self.threshold
        cond_mask = cond_v <= 0

        # when theta not in [0,pi], use cosface instead
        keep_val = (cos_theta - self.mm) 
        cos_theta_m[cond_mask] = keep_val[cond_mask]

        # a little bit hacky way to prevent in_place operation on cos_theta
        output = cos_theta * 1.0
        idx_ = torch.arange(0, nB, dtype = torch.long)
        output[idx_, label] = cos_theta_m[idx_, label]

        # scale up in order to make softmax work, first introduced in normface
        output *= self.s 
        return output


if __name__ == "__main__":

    arcface = ArcFace()
    tensor_input = torch.Tensor((1, 512))
    label = torch.Tensor(1)
    output = arcface(tensor_input, label)
    print(output.shape)

