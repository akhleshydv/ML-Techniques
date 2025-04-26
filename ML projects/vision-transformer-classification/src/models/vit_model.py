import torch
import torch.nn as nn
from torchvision.models import vit_b_16

class ViTBinaryClassifier(nn.Module):
    def __init__(self):
        super(ViTBinaryClassifier, self).__init__()
        # Load the pre-trained Vision Transformer model
        self.model = vit_b_16(pretrained=True)
        # Modify the classifier head for binary classification
        self.model.heads = nn.Sequential(
            nn.Linear(self.model.heads.head.in_features, 1),  # Binary classification
            nn.Sigmoid()  # Output between 0 and 1
        )

    def forward(self, x):
        return self.model(x)