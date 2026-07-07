import torch
import torch.nn as nn
import torch.nn.functional as F
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
        
        # Cargamos el modelo base. Mantenemos aux_logits=True temporalmente para que acepte los pesos preentrenados
        self.backbone = models.inception_v3(weights=weights, aux_logits=True)
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            
            # === AQUÍ AGREGAS LA CAPA DE DROPOUT ===
            nn.Dropout(0.55), # Sube de 0.40 a 0.55 para frenar el sobreajuste
            
            nn.Linear(512, num_classes)
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # 2. Descongelamos selectivamente los bloques 6 y 7, más la capa fc
            # Usamos named_parameters() para poder filtrar por el nombre de las capas
            for name, param in self.backbone.named_parameters():
                if "Mixed_6" in name or "Mixed_7" in name or "fc" in name:
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Si el modelo está en modo entrenamiento (model.train()), InceptionV3 de PyTorch 
        # devuelve un objeto (InceptionOutputs) con .logits y .aux_logits.
        # Para evitar problemas con la función de pérdida CrossEntropyLoss, extraemos solo los logits principales.
        if self.training:
            output = self.backbone(x)
            return output.logits  # Extrae la salida del clasificador principal de 10 clases
        else:
            return self.backbone(x) # En model.eval() devuelve directamente un Tensor normal

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def parameter_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total:     {total:>12,}")
        print(f"Trainable: {trainable:>12,}")
        print(f"Frozen:    {total - trainable:>12,}")

class ConvNet(nn.Module):
  """VGG-13 style plain CNN for mel-spectrogram genre classification.

  Sequential Conv-BN-ReLU blocks.
  """

  NUM_GENRES = 10

  def __init__(
    self,
    num_classes: int = NUM_GENRES,
    dropout: float = 0.5,
    spatial_dropout: float = 0.1,
  ) -> None:
    super().__init__()

    self.block1 = self._make_block(1,   32,  spatial_dropout)
    self.block2 = self._make_block(32,  64,  spatial_dropout)
    self.block3 = self._make_block(64,  128, spatial_dropout)
    self.block4 = self._make_block(128, 256, spatial_dropout)
    self.block5 = self._make_block(256, 256, spatial_dropout)

    self.gap = nn.AdaptiveAvgPool2d(1)
    self.dropout = nn.Dropout(dropout)
    self.fc = nn.Linear(256, num_classes)

    self._init_weights()

  def _make_block(self, in_c: int, out_c: int, spatial_dropout: float) -> nn.Sequential:
    return nn.Sequential(
      nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_c),
      nn.ReLU(inplace=True),
      nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
      nn.BatchNorm2d(out_c),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2),
      nn.Dropout2d(spatial_dropout) if spatial_dropout > 0 else nn.Identity(),
    )

  def _init_weights(self) -> None:
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
      elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)
      elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.01)
        nn.init.constant_(m.bias, 0.0)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    x = self.block5(x)
    x = self.gap(x).flatten(1)
    x = self.dropout(x)
    return self.fc(x)

  def parameter_summary(self) -> None:
    total = sum(p.numel() for p in self.parameters())
    trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
    print(f"Total:     {total:>12,}")
    print(f"Trainable: {trainable:>12,}")
