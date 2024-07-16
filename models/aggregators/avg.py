import torch
import torch.nn.functional as F
import torch.nn as nn

class AVG(nn.Module):
    def __init__(self):
        super(AVG, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # from resent18 source code

    def forward(self, x):
        x = self.avgpool(x)
        x = F.normalize(x.flatten(1), p=2, dim=1)
        return x

if __name__ == '__main__':
    x = torch.randn(4, 2048, 10, 10)
    m = AVG()
    r = m(x)
    print(r.shape)