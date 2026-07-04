import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import Inception_V3_Weights

class InceptionV3(nn.Module):
  """Inception v3 fine-tuned for GTZAN genre classification.

  Backbone: torchvision's Inception v3 pretrained on ImageNet. The
  auxiliary classifier is kept in the loaded architecture (torchvision
  requires it when loading pretrained weights) but disabled at forward
  time and frozen during training.
  Head: replaced with a ``num_classes``-way Linear layer.
  All original parameters are frozen; only the new head is trainable.
  """

  NUM_GENRES = 10

  def __init__(
    self,
    num_classes: int = NUM_GENRES,
    freeze_backbone: bool = True,
  ) -> None:
    super().__init__()

    weights = Inception_V3_Weights.DEFAULT
    self.backbone = models.inception_v3(weights=weights, aux_logits=True)
    # `fc` is the final "fully connected" layer; we swap it for a 10-class head.
    self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
    # Disable the aux branch since fine-tuning doesn't need it.
    self.backbone.aux_logits = False

    if freeze_backbone:
      # Freeze the pretrained backbone so its weights aren't updated during training.
      for param in self.backbone.parameters():
        param.requires_grad = False
      # Only the new classification head will be trained.
      for param in self.backbone.fc.parameters():
        param.requires_grad = True

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.backbone(x)

  def trainable_parameters(self):
    return filter(lambda p: p.requires_grad, self.parameters())

  def parameter_summary(self) -> None:
    total = sum(p.numel() for p in self.parameters())
    trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print(f"Total:     {total:>12,}")
    print(f"Trainable: {trainable:>12,}")
    print(f"Frozen:    {total - trainable:>12,}")
