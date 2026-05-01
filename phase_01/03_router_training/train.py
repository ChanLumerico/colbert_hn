import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from typing import Dict
import os

def train_router(model: nn.Module, dataset, epochs: int = 20, batch_size: int = 32, 
                 lr: float = 1e-3, device: str = "cpu", pbar_desc: str = "    Training Epochs") -> Dict:
    """
    Standard training loop for the LayerRouter with tqdm.notebook bars.
    """
    from tqdm.notebook import tqdm
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_acc": []}

    epoch_pbar = tqdm(range(epochs), desc=pbar_desc, leave=False)
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        batch_pbar = tqdm(train_loader, desc=f"      Epoch {epoch+1} Batches", leave=False)
        for q_batch, d_batch, label_batch in batch_pbar:
            q_batch, d_batch, label_batch = q_batch.to(device), d_batch.to(device), label_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(q_batch, d_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for q_batch, d_batch, label_batch in val_loader:
                q_batch, d_batch, label_batch = q_batch.to(device), d_batch.to(device), label_batch.to(device)
                logits = model(q_batch, d_batch)
                preds = (torch.sigmoid(logits) > 0.5).float()
                correct += (preds == label_batch).sum().item()
                total += label_batch.size(0)

        val_acc = correct / total if total > 0 else 0
        history["train_loss"].append(total_loss / len(train_loader))
        history["val_acc"].append(val_acc)
        
        epoch_pbar.set_postfix({"loss": f"{total_loss/len(train_loader):.4f}", "val_acc": f"{val_acc:.4f}"})

    return history
