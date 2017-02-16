import torch.nn as nn
import torch.utils.model_zoo as model_zoo


__all__ = ['Tmp', 'tmp']


class Tmp(nn.Module):
    def __init__(self, num_classes=1000):
        super(Tmp, self).__init__()

        self.layer = nn.Linear(224 * 224 * 3, num_classes)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.layer(x)


def tmp(pretrained=False):
    return Tmp()
