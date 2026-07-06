import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import GTZANDataset
from cnn_scratch import AudioCNNScratch

from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Métricas
        total_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total += y.size(0)
        
    avg_loss = total_loss / len(dataloader)
    accuracy = (correct / total) * 100
    print(f"Train Error: \n Accuracy: {accuracy:.1f}%, Avg loss: {avg_loss:.4f}")

def validate(dataloader, model, loss_fn):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            
            total_loss += loss.item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            total += y.size(0)
            
    avg_loss = total_loss / len(dataloader)
    accuracy = (correct / total) * 100
    print(f"Validation Error: \n Accuracy: {accuracy:.1f}%, Avg loss: {avg_loss:.4f}")
    return avg_loss

def final_evaluation(dataloader, model, genres_list):
    """Ejecuta una evaluación exhaustiva generando la matriz de confusión 
    
    y las métricas de precisión, recall y f1-score por cada género.
    """
    print("\n" + "="*60)
    print("INICIANDO EVALUACIÓN FINAL DE MÉTRICAS DETALLADAS")
    print("="*60)
    
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            pred = model(X)
            
            # Guardamos las predicciones y las etiquetas reales
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
            
    # 1. Reporte de Clasificación (Precisión, Recall, F1-Score)
    print("\n--- REPORTE DE CLASIFICACIÓN POR GÉNERO ---")
    print(classification_report(all_labels, all_preds, target_names=genres_list, zero_division=0))
    
    # 2. Matriz de Confusión Pura
    print("--- MATRIZ DE CONFUSIÓN ANALÍTICA ---")
    print(confusion_matrix(all_labels, all_preds))
    print("="*60 + "\n")
def loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs):
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        print("-" * 60)
        
        # Mantenemos la aleatoriedad dinámica por época en el entrenamiento
        train_dataloader.dataset.reset_epoch_samples()
        
        epoch_start = time.perf_counter()
        train(train_dataloader, model, loss_fn, optimizer)
        validate(val_dataloader, model, loss_fn)
        epoch_end = time.perf_counter()
        
        print(f"Epoch time: {epoch_end - epoch_start:.2f}s")
    print("\n¡Entrenamiento desde cero completado con éxito!")
    genres_list = train_dataloader.dataset.GENRES 
    final_evaluation(val_dataloader, model, genres_list)

def main():
    if len(sys.argv) < 3:
        print("Uso: docker compose exec pytorch python3 src/train_scratch.py <batch_size> <epochs>")
        sys.exit(1)
        
    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])
    
    print(f"Configurando Entrenamiento desde cero (Scratch) en dispositivo: {device}")
    
    # Instanciamos los dos conjuntos de forma nativa e independiente usando tu dataset por canciones
    train_ds = GTZANDataset(split="train")
    val_ds = GTZANDataset(split="val")
    
    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    # Inicializamos la red desde cero y mostramos su resumen
    model = AudioCNNScratch().to(device)
    model.parameter_summary()
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Al entrenar desde cero, empezamos con el LR estándar recomendado para Adam (1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs)

if __name__ == "__main__":
    main()