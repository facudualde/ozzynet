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
        
        # Reemplazamos la capa de clasificación final por nuestra capa de 10 géneros
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

        if freeze_backbone:
            # Primero: Congelamos absolutamente todos los parámetros del backbone original
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Segundo: Desactivamos explícitamente los gradientes del bloque auxiliar interno
            if self.backbone.AuxLogits is not None:
                for param in self.backbone.AuxLogits.parameters():
                    param.requires_grad = False

            # Tercero: Forzamos que la nueva capa FC SÍ sea entrenable
            for param in self.backbone.fc.parameters():
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