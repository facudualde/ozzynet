import torch
import torch.nn as nn

class AudioCNNScratch(nn.Module):
    """CNN personalizada para clasificación de géneros musicales basada en espectrogramas.
    
    Diseñada desde cero para procesar entradas de 3x299x299, reduciendo progresivamente
    las dimensiones espaciales mientras aumenta los canales de características.
    """
    
    NUM_GENRES = 10

    def __init__(self, num_classes: int = NUM_GENRES) -> None:
        super().__init__()
        
        # Bloque Convolucional 1: Entrada 299x299 -> Salida 149x149
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        
        # Bloque Convolucional 2: Entrada 149x149 -> Salida 74x74
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloque Convolucional 3: Entrada 74x74 -> Salida 37x37
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloque Convolucional 4: Entrada 37x37 -> Salida 18x18
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Bloque Convolucional 5: Entrada 18x18 -> Salida 9x9
        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Capa de Reducción Global Adaptativa (Fuerza a que la salida espacial sea de 1x1)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Clasificador Lineal Final (Totalmente conectado)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5), # Regularización estricta para evitar sobreajuste desde cero
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.global_pool(x)
        return self.fc(x)

    def trainable_parameters(self):
        return filter(lambda p: p.requires_grad, self.parameters())

    def parameter_summary(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        print(f"--- CNN Scratch Summary ---")
        print(f"Total Parameters: {total:>12,}")
        print(f"---------------------------")