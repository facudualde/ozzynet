import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from cnn import InceptionV3
from dataset_fine_tuning import GTZANDataset as DatasetFT
from sklearn.metrics import classification_report, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def format_duration(seconds: float) -> str:
  return str(timedelta(seconds=int(seconds)))

def train(dataloader, model, loss_fn, optimizer):
    model.train()
    total_loss = 0
    correct = 0  # <--- Agregamos contador de aciertos
    total = 0    # <--- Agregamos contador de muestras totales
    
    for batch, batch_data in enumerate(dataloader):
        X, y = batch_data[0].to(device), batch_data[1].to(device)
        
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
  #size = len(val_dataloader.dataset)
  
  num_batches = len(val_dataloader)

  model.eval()
  
  #test_loss, correct = 0, 0
  test_loss = 0
  song_probs = defaultdict(list)
  song_targets = {}
  with torch.no_grad():
    for batch_data in val_dataloader:
            X, y = batch_data[0].to(device), batch_data[1].to(device)
            paths = batch_data[2] # Tercer parámetro devuelto por el DatasetFT

            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            probs = torch.softmax(pred, dim=1).cpu().numpy()
            y_cpu = y.cpu().numpy()
            #correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            for i in range(len(paths)):
                # Extraemos el identificador único (ej: 'blues.00012') asumiendo 
                # que están estructurados dentro de una carpeta por canción
                song_id = Path(paths[i]).parent.name
                
                song_probs[song_id].append(probs[i])
                song_targets[song_id] = y_cpu[i]
  
  test_loss /= num_batches
 # correct /= size
 
  correct_songs = 0
  total_songs = len(song_probs)
  
  for song_id, probs_list in song_probs.items():
      # Soft voting: promediamos las probabilidades de los 10 espectrogramas
      mean_probs = np.mean(probs_list, axis=0)
      final_pred = np.argmax(mean_probs)
      
      if final_pred == song_targets[song_id]:
          correct_songs += 1
          
  voting_accuracy = (correct_songs / total_songs) * 100
  
  print(
      f"Validation Error (VOTING): \n" 
      f" Accuracy: {voting_accuracy:.1f}% ({correct_songs}/{total_songs} canciones)\n"
      f" Avg loss (per segment): {test_loss:.6f}\n"
  )

def final_evaluation(dataloader, model, genres_list):
    """Ejecuta una evaluación exhaustiva generando la matriz de confusión 
    
    y las métricas de precisión, recall y f1-score por cada género.
    """
    print("\n" + "="*60)
    print("INICIANDO EVALUACIÓN FINAL DE MÉTRICAS DETALLADAS")
    print("="*60)
    
    model.eval()
    song_probs = defaultdict(list)
    song_targets = {}
    with torch.no_grad():
        for batch_data in dataloader:
            X, y = batch_data[0].to(device), batch_data[1].to(device)
            paths = batch_data[2]
            
            pred = model(X)
            probs = torch.softmax(pred, dim=1).cpu().numpy()
            y_cpu = y.cpu().numpy()
            
            for i in range(len(paths)):
                song_id = Path(paths[i]).parent.name
                song_probs[song_id].append(probs[i])
                song_targets[song_id] = y_cpu[i]
    '''
    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            pred = model(X)
            
            # Guardamos las predicciones y las etiquetas reales
            all_preds.extend(pred.argmax(1).cpu().numpy())
            all_labels.extend(y.numpy())
    '''


    all_preds = []
    all_labels = []
    for song_id, probs_list in song_probs.items():
        mean_probs = np.mean(probs_list, axis=0)
        all_preds.append(np.argmax(mean_probs))
        all_labels.append(song_targets[song_id])

            
    # 1. Reporte de Clasificación (Precisión, Recall, F1-Score)
    print("\n--- REPORTE DE CLASIFICACIÓN POR GÉNERO ---")
    print(classification_report(all_labels, all_preds, target_names=genres_list, zero_division=0))
    
    # 2. Matriz de Confusión Pura
    print("--- MATRIZ DE CONFUSIÓN ANALÍTICA ---")
    print(confusion_matrix(all_labels, all_preds))
    print("="*60 + "\n")

def loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs):
  print("=" * 60)
  print("  Training InceptionV3 on GTZAN with Voting Validation")
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
    train_ds = DatasetFT(split="train")
    val_ds   = DatasetFT(split="val")

    train_dataloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True) # IMPORTANTE: pon shuffle=True aquí
    val_dataloader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = InceptionV3().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=1e-5,weight_decay=0.01)

    loop(train_dataloader, val_dataloader, model, loss_fn, optimizer, epochs)

    
if __name__ == "__main__":
  setup()
