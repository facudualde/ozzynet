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

    weights = Inception_V3_Weights.DEFAULT
    
    # Load the base model. Keep aux_logits=True temporarily so torchvision accepts the pretrained weights.
    self.backbone = models.inception_v3(weights=weights, aux_logits=True)

    # Replace the final classification layer with our 10-genre head.
    self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    if freeze_backbone:
      # First: freeze every parameter of the original backbone.
      for param in self.backbone.parameters():
        param.requires_grad = False

      # Second: explicitly disable gradients on the internal auxiliary block.
      if self.backbone.AuxLogits is not None:
        for param in self.backbone.AuxLogits.parameters():
          param.requires_grad = False

      # Third: force the new FC layer to be trainable.
      for param in self.backbone.fc.parameters():
        param.requires_grad = True

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    # When the model is in training mode (model.train()), torchvision's InceptionV3
    # returns an InceptionOutputs named tuple with .logits and .aux_logits.
    # To avoid issues with CrossEntropyLoss, we extract only the main logits.
    if self.training:
      output = self.backbone(x)
      return output.logits  # Extracts the output of the 10-class main classifier.
    else:
      return self.backbone(x)  # In model.eval() it returns a plain Tensor directly.

  def trainable_parameters(self):
    return filter(lambda p: p.requires_grad, self.parameters())

  def parameter_summary(self) -> None:
    total = sum(p.numel() for p in self.parameters())
    trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print(f"Total:     {total:>12,}")
    print(f"Trainable: {trainable:>12,}")
    print(f"Frozen:    {total - trainable:>12,}")
