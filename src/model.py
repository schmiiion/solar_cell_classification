import torch
import torch.nn as nn
import torchvision.models as models

class DenseNetMultiLabel(torch.nn.Module):
    def __init__(self, num_labels=2):
        super().__init__()
        self.base = models.densenet121()
        self.base.classifier = nn.Linear(self.base.classifier.in_features, num_labels)

    def forward(self, x):
        return torch.sigmoid(self.base(x))

    def train(self, mode=True):
        super().train(mode)        # ensures proper mode setting
        self.base.train(mode)

    def eval(self):
        super().eval()
        self.base.eval()
