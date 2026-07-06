import os
import sys
import time
from datetime import datetime, timedelta
import torch
from torch import nn
from torch.utils.data import DataLoader
from cnn import InceptionV3
from dataset import GTZANDataset
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_duration(seconds: float) -> str:
  return str(timedelta(seconds=int(seconds)))

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    correct = 0  # <--- Agregamos contador de aciertos
    total = 0    # <--- Agregamos contador de muestras totales
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Métricas del batch
        total_loss += loss.item()
        
        # --- NUEVA LÓGICA DE MÉTRICAS EN TRAIN ---
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        total += y.size(0)
        # ----------------------------------------
        
    # Calculamos los promedios finales de la época de entrenamiento
    avg_loss = total_loss / len(dataloader)
    train_accuracy = (correct / total) * 100  # <--- Calculamos el porcentaje
    
    # Modificamos el print para que te muestre ambas métricas alineadas
    print(f"Train Error: \n Accuracy: {train_accuracy:.1f}%, Avg loss: {avg_loss:.4f}")

def validate(val_dataloader, model, loss_fn):
  size = len(val_dataloader.dataset)
  num_batches = len(val_dataloader)
  model.eval()
  test_loss, correct = 0, 0
  with torch.no_grad():
    for X, y in val_dataloader:
      X, y = X.to(device), y.to(device)
      pred = model(X)
      test_loss += loss_fn(pred, y).item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  test_loss /= num_batches
  correct /= size
  print(
    f"Validation Error: \n" 
    f"Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
  )

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
  print("=" * 60)
  print("  Training InceptionV3 on GTZAN")
  print(f"  Device:  {device}")
  print(f"  Epochs:  {epochs}")
  print(f"  Batch:   {train_dataloader.batch_size}")
  print(f"  Train:   {len(train_dataloader.dataset)} samples / {len(train_dataloader)} batches")
  print(f"  Val:     {len(val_dataloader.dataset)} samples / {len(val_dataloader)} batches")
  model.parameter_summary()
  print("=" * 60)

  total_start = time.perf_counter()
  for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    print("-" * 60)

    train_dataloader.dataset.reset_epoch_samples()

    epoch_start = time.perf_counter()
    train(train_dataloader, model, loss_fn, optimizer)

    validate(val_dataloader, model, loss_fn)
    epoch_time = time.perf_counter() - epoch_start
    print(f"  epoch {epoch} time: {format_duration(epoch_time)}")
  
  genres_list = train_dataloader.dataset.GENRES 
  final_evaluation(val_dataloader, model, genres_list)
  
  total_time = time.perf_counter() - total_start
  print("\n" + "=" * 60)
  print(f"  Done!  Total time: {format_duration(total_time)}")
  print("=" * 60)

  print()
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
  save_dir = os.path.join("checkpoints", timestamp)
  os.makedirs(save_dir, exist_ok=True)
  save_path = os.path.join(save_dir, f"{timestamp}.pth")
  torch.save(model.state_dict(), save_path)
  print(f"Model saved to: {save_path}")

def setup():
    batch_size = int(sys.argv[1])
    epochs = int(sys.argv[2])

    # Instanciación nativa e independiente de cada conjunto
    train_ds = GTZANDataset(split="train")
    val_ds   = GTZANDataset(split="val")

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) # IMPORTANTE: pon shuffle=True aquí
    val_dataloader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = InceptionV3().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-5,weight_decay=0.01)

    loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs)

    
if __name__ == "__main__":
  setup()
