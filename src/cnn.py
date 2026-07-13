from typing import Iterator

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import Inception_V3_Weights


class InceptionV3(nn.Module):
    """Inception v3 fine-tuned for GTZAN genre classification."""

    NUM_GENRES = 10

    def __init__(
        self,
        num_classes: int = NUM_GENRES,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()

        # Load pretrained backbone; aux_logits=True is required to load the weights.
        self.backbone = models.inception_v3(weights=Inception_V3_Weights.DEFAULT, aux_logits=True)

        # Replace the classifier head with a small MLP for genre prediction.
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.55),
            nn.Linear(512, num_classes),
        )

        if freeze_backbone:
            # Freeze everything, then unfreeze the last two mixed blocks + the new head.
            for param in self.backbone.parameters():
                param.requires_grad = False
            for name, param in self.backbone.named_parameters():
                if "Mixed_6" in name or "Mixed_7" in name or "fc" in name:
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In train() the backbone returns (logits, aux_logits); keep only the main branch.
        if self.training:
            return self.backbone(x).logits
        return self.backbone(x)

    def trainable_parameters(self) -> Iterator[torch.nn.Parameter]:
        # Iterator over parameters that will receive gradients.
        return filter(lambda p: p.requires_grad, self.parameters())

    def parameter_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total:     {total:>12,}")
        print(f"Trainable: {trainable:>12,}")
        print(f"Frozen:    {total - trainable:>12,}")


class ConvNet(nn.Module):
    """Classic CNN for spectrogram classification (trained from scratch)."""

    NUM_GENRES = 10

    def __init__(self, num_classes: int = NUM_GENRES, dropout_rate: float = 0.3) -> None:
        super().__init__()

        # Feature extractor: four conv+ReLU+maxpool blocks doubling channels each time.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier head; 256*8*8 assumes a 128x130 input after four stride-2 pools.
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 128)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # CrossEntropyLoss applies softmax internally, so we return raw logits.
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool3(self.relu3(self.conv3(x)))
        x = self.pool4(self.relu4(self.conv4(x)))
        x = self.dropout(self.relu_fc(self.fc1(self.flatten(x))))
        return self.fc2(x)

    def parameter_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total:     {total:>12,}")
        print(f"Trainable: {trainable:>12,}")
