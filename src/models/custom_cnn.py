import torch
import torch.nn as nn

class CustomCNN(nn.Module):
    """
    Baseline Custom Convolutional Neural Network (CNN) for Plant Disease Classification.
    Architecture:
      Input (3 x 224 x 224)
        │
      ConvBlock 1: Conv2D(3->32) -> BatchNorm -> ReLU -> MaxPool2D(2)
        │
      ConvBlock 2: Conv2D(32->64) -> BatchNorm -> ReLU -> MaxPool2D(2)
        │
      ConvBlock 3: Conv2D(64->128) -> BatchNorm -> ReLU -> MaxPool2D(2)
        │
      AdaptiveAvgPool2d (Global Average Pooling)
        │
      Linear(128 -> 256) -> ReLU -> Dropout(p=0.5)
        │
      Linear(256 -> num_classes)
    """
    def __init__(self, num_classes: int = 15, dropout_rate: float = 0.5):
        super(CustomCNN, self).__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.global_pool(x)
        logits = self.classifier(x)
        return logits
