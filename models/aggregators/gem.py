import torch
import torch.nn.functional as F
import torch.nn as nn

class GeMPool(nn.Module):
    '''
    Implementation of GemPool.
    Args:
        p (int):  parameter that controls the exponent of the generalized mean. Default value=3.
        eps (float): parameter for avoiding the division by zero. Default value=1e-6.
    '''
    def __init__(self, p=3, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1)*p) # tensor([p.])
        self.eps = eps

    def forward(self, x):
        x = F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1./self.p)
        x = F.normalize(x.flatten(1), p=2, dim=1)
        return x     

if __name__ == '__main__':
    x = torch.randn(4, 2048, 10, 10) 
    m = GeMPool(p=3, eps=1e-6)
    r = m(x) 
    print(r.shape)              