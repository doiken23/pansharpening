import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss2d(nn.Module):
    def __init__(self,
            size_average=False,
            reduce=True,
            reduction='elementwise_mean'
            ):

        self.mse = nn.MSELoss(
                size_average=size_average,
                reduce=reduce,
                reduction=reduction)

    def forward(self, x, target):
        n = x.size(0)
        x = x.view(n, -1)
        target = target(n, -1)

        return self.mse(x, target)

class Criterion(nn.Module):
    def __init__(self, model, out_channels, alpha=1,
            size_average=True,
            reduce=True,
            reduction='elementwise_mean'):

        super(Criterion, self).__init__()
        self.model = model
        self.downsamlle = nn.AvgPool2d(2, 2)
        self.conv = nn.Conv2d(out_channels, 1, 1)
        self.mse = MSELoss2D(
                size_average=size_average,
                reduce=reduce,
                reduction=reduction)

    def forward(self, x1, x2):
        outputs = self.model(x1, x2)
        spectral_loss = self.mse(self.downsamlle(outpus), x1)
        spatial_loss = self.mse(self.conv(outpus), x2)
        loss = spectral_loss + spatial_loss

        return loss
