import torch.nn as nn
from torchvision import models

NAMES = {'Custom CNN': 'cnn', 'ResNet50': 'resnet50', 'EfficientNet-B0': 'efficientnet_b0'}

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b in [(3, 32), (32, 64), (64, 128)]:
            layers.extend([nn.Conv2d(a, b, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)])
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(.4), nn.Linear(128, 2))
    def forward(self, x):
        return self.head(self.features(x))

def build(name, pretrained=False):
    name = NAMES.get(name, name)
    if name == 'cnn':
        model = CNN()
        target = model.features[-3]
    elif name == 'resnet50':
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        target = model.layer4[-1]
    elif name == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
        target = model.features[-1]
    else:
        raise ValueError('Unknown architecture')
    return model, target
