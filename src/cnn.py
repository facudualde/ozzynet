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

class ConvNet(nn.Module):
    """
    CNN clásica para clasificación de espectrogramas.
    Estructura simplificada alineada con los conceptos del curso.
    """
    NUM_GENRES = 10

    def __init__(self, num_classes: int = NUM_GENRES, dropout_rate: float = 0.3) -> None:
        super().__init__()

        # --- EXTRACTOR DE CARACTERÍSTICAS (Capas Convolucionales) ---
        # Bloque 1: Entrada (1 canal, ej. gris) -> 32 filtros
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque 2: 32 filtros -> 64 filtros
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque 3: 64 filtros -> 128 filtros
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bloque 4: 128 filtros -> 256 filtros
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- CLASIFICADOR (Capas Densas / Totalmente Conectadas) ---
        # El Flatten tradicional de PyTorch se define aquí o directamente en el forward
        self.flatten = nn.Flatten()

        # Nota: El tamaño de entrada '256 * 8 * 8' asume que tu espectrograma 
        # se reduce a ese tamaño tras los 4 MaxPool. Si cambias el tamaño de la imagen,
        # este número cambia (tal como pasa con el Flatten de Keras).
        self.fc1 = nn.Linear(256 * 8 * 8, 128)
        self.relu_fc = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate) # Dropout estándar del curso
        
        # Capa de salida: 128 neuronas -> 10 clases (géneros)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Bloque 1
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # Bloque 2
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # Bloque 3
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # Bloque 4
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        # Clasificador
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu_fc(x)
        x = self.dropout(x)
        
        # En PyTorch devolvemos los "logits" directos. 
        # La función de pérdida (CrossEntropyLoss) ya aplica el Softmax internamente.
        return self.fc2(x)
    def parameter_summary(self) -> None:
      total = sum(p.numel() for p in self.parameters())
      trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
      print(f"Total:     {total:>12,}")
      print(f"Trainable: {trainable:>12,}")
