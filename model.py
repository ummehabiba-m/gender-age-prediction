
import torch
import torch.nn as nn
from torchvision import models

class GenderAgeModel(nn.Module):
    def __init__(self):
        super().__init__()

        vgg = models.vgg16(weights=None)
        vgg.load_state_dict(torch.load(
            '/content/vgg16-397923af.pth',
            map_location='cpu',
            weights_only=False
        ))
        print("✅ VGG16 weights loaded!")

        for param in vgg.features.parameters():
            param.requires_grad = False

        self.features = vgg.features
        self.avgpool  = nn.AdaptiveAvgPool2d((7, 7))

        self.shared = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(4096, 1024), nn.ReLU(), nn.Dropout(0.5)
        )

        self.gender_head = nn.Linear(1024, 1)
        self.age_head    = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.shared(x)
        gender = torch.sigmoid(self.gender_head(x))
        age    = self.age_head(x)
        return gender, age
