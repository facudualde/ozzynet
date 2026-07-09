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