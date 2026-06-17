"""
model_training.py
End-to-end training loop for the Mamba Neural Operator and baseline models.
Strictly adheres to the hyperparameter configurations, regularization terms, 
and training protocols described in Section 4.4 and Table 3 of the manuscript.

Author: [Your Name/Team]
License: MIT
"""

import os
import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from typing import Tuple, Optional, Dict

# ==============================================================================
# 1. Reproducibility Setup (Section 5.3: Fixed random seeds)
# ==============================================================================
def set_seed(seed: int):
    """Sets all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensures deterministic behavior on CUDA (may slightly impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random seed set to: {seed}")

# ==============================================================================
# 2. Custom Loss Functions (Section 4.4: Eq. 21, 22, 23)
# ==============================================================================
class PlantGrowthMetabolismLoss(nn.Module):
    def __init__(self, lambda_smooth: float = 0.1, lambda_sparse: float = 0.01):
        super().__init__()
        self.lambda_smooth = lambda_smooth
        self.lambda_sparse = lambda_sparse
        self.mse_loss = nn.MSELoss()

    def forward(self, predictions: torch.Tensor, targets: torch.Tensor, 
                hidden_states: torch.Tensor, gating_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            predictions: Model output (batch, seq_len, features)
            targets: Ground truth (batch, seq_len, features)
            hidden_states: Latent metabolic-growth state (batch, seq_len, hidden_dim)
            gating_weights: Metabolite-gated values (batch, seq_len, 1) in [0, 1]
        """
        # 1. Primary Prediction Loss (L_pred)
        loss_pred = self.mse_loss(predictions, targets)
        
        # 2. State Transition Smoothness (L_smooth, Eq. 22)
        # Penalizes abrupt changes in hidden dynamics along the time dimension
        state_diff = torch.diff(hidden_states, dim=1)
        loss_smooth = torch.mean(state_diff ** 2)
        
        # 3. Gating Sparsity (L_sparse, Eq. 23)
        # Encourages gating values toward 0 or 1 (binary-like) to avoid ambiguous coupling
        # Maximized when gating is 0.5, minimized at 0 or 1
        loss_sparse = torch.mean(gating_weights * (1.0 - gating_weights))
        
        total_loss = loss_pred + self.lambda_smooth * loss_smooth + self.lambda_sparse * loss_sparse
        return total_loss, loss_pred, loss_smooth, loss_sparse

# ==============================================================================
# 3. Optimizer and Scheduler (Table 3: AdamW + Cosine Decay + Warmup)
# ==============================================================================
def get_optimizer_and_scheduler(model: nn.Module, total_steps: int, warmup_ratio: float = 0.1):
    """
    Configures AdamW optimizer and Cosine Annealing scheduler with linear warmup.
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,          # eta_max
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    warmup_steps = int(total_steps * warmup_ratio)
    
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1e-5 / 3e-4) # Scale to eta_min

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return optimizer, scheduler

# ==============================================================================
# 4. Training Loop with Curriculum Learning & Scheduled Sampling
# ==============================================================================
def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LambdaLR,
    criterion: PlantGrowthMetabolismLoss,
    epoch: int,
    device: torch.device,
    scaler: GradScaler,
    max_seq_len: int,
    teacher_forcing_ratio: float,
    grad_clip_history: list
) -> Dict[str, float]:
    
    model.train()
    epoch_loss = 0.0
    epoch_pred_loss = 0.0
    
    # Adaptive gradient clipping threshold (Table 3: 0.5 * recent_norm)
    recent_grad_norm = np.mean(grad_clip_history[-10:]) if len(grad_clip_history) >= 10 else 1.0
    clip_threshold = 0.5 * recent_grad_norm

    for batch_idx, batch in enumerate(dataloader):
        # Truncate or pad sequence to current curriculum length
        inputs = batch['x'][:, :max_seq_len, :].to(device)
        targets = batch['y'][:, :max_seq_len, :].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(enabled=True): # Mixed precision training (AMP)
            # Forward pass (Model must return predictions, hidden_states, and gating_weights)
            predictions, hidden_states, gating_weights = model(inputs, teacher_forcing_ratio=teacher_forcing_ratio)
            
            # Compute custom loss
            total_loss, pred_loss, smooth_loss, sparse_loss = criterion(
                predictions, targets, hidden_states, gating_weights
            )
        
        # Backward pass with gradient scaling
        scaler.scale(total_loss).backward()
        
        # Adaptive gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_threshold)
        
        # Optimizer step and scaler update
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        # Track gradient norm for adaptive clipping in next epoch
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        grad_clip_history.append(total_norm ** 0.5)
        
        epoch_loss += total_loss.item()
        epoch_pred_loss += pred_loss.item()
        
    avg_loss = epoch_loss / len(dataloader)
    avg_pred_loss = epoch_pred_loss / len(dataloader)
    
    return {
        'total_loss': avg_loss,
        'pred_loss': avg_pred_loss,
        'current_lr': optimizer.param_groups[0]['lr'],
        'seq_len': max_seq_len,
        'tf_ratio': teacher_forcing_ratio
    }

# ==============================================================================
# 5. Main Training Protocol (Section 4.4 & Table 3)
# ==============================================================================
def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    model_name: str,
    seed: int,
    num_epochs: int = 100,
    patience: int = 20,
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
):
    print(f"\n{'='*60}")
    print(f"Starting Training: {model_name} | Seed: {seed}")
    print(f"{'='*60}")
    
    set_seed(seed)
    model = model.to(device)
    
    # Initialize components per Table 3
    criterion = PlantGrowthMetabolismLoss(lambda_smooth=0.1, lambda_sparse=0.01)
    
    # Calculate total steps for scheduler
    total_steps = num_epochs * len(train_loader)
    optimizer, scheduler = get_optimizer_and_scheduler(model, total_steps, warmup_ratio=0.1)
    scaler = GradScaler()
    
    # Curriculum Learning Schedule (Table 3)
    curriculum_lengths = [100, 500, 2000, 16384]
    
    # Scheduled Sampling: starts at 1.0, decays to 0.7
    def get_tf_ratio(epoch):
        return max(0.7, 1.0 - (epoch / num_epochs) * 0.3)
    
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    grad_clip_history = []
    
    for epoch in range(1, num_epochs + 1):
        # 1. Determine curriculum sequence length
        if epoch <= num_epochs * 0.25:
            current_seq_len = curriculum_lengths[0]
        elif epoch <= num_epochs * 0.50:
            current_seq_len = curriculum_lengths[1]
        elif epoch <= num_epochs * 0.75:
            current_seq_len = curriculum_lengths[2]
        else:
            current_seq_len = curriculum_lengths[3]
            
        tf_ratio = get_tf_ratio(epoch)
        
        # 2. Train
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, criterion, epoch, device, 
            scaler, current_seq_len, tf_ratio, grad_clip_history
        )
        
        # 3. Validate (Simplified for brevity)
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch['x'][:, :current_seq_len, :].to(device)
                targets = batch['y'][:, :current_seq_len, :].to(device)
                with autocast(enabled=True):
                    preds, _, _ = model(inputs, teacher_forcing_ratio=1.0) # No scheduled sampling in val
                    val_loss += nn.functional.mse_loss(preds, targets).item()
        val_loss /= len(val_loader)
        
        # 4. Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best model for this seed
            torch.save(model.state_dict(), f"checkpoints/{model_name}_seed{seed}_best.pth")
            print(f"  [Epoch {epoch:03d}] Saved Best Model | Val Loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            
        print(f"  [Epoch {epoch:03d}] Train Loss: {train_metrics['total_loss']:.4f} | "
              f"Val Loss: {val_loss:.4f} | LR: {train_metrics['current_lr']:.2e} | "
              f"Seq Len: {train_metrics['seq_len']} | TF Ratio: {train_metrics['tf_ratio']:.2f}")
        
        if epochs_without_improvement >= patience:
            print(f"  [INFO] Early stopping triggered at epoch {epoch}.")
            break
            
    print(f"[SUCCESS] Training completed for {model_name} (Seed {seed}). Best Val Loss: {best_val_loss:.4f}")
    return best_val_loss

# ==============================================================================
# 6. Execution Example (Demonstrates Reproducibility across 5 seeds)
# ==============================================================================
if __name__ == "__main__":
    # Ensure checkpoint directory exists
    os.makedirs("checkpoints", exist_ok=True)
    
    # The 5 fixed random seeds reported in Section 5.3 and Table 2/5 notes
    SEEDS = [42, 123, 456, 789, 1024]
    
    # Mock dataloaders (Replace with actual dataset loading from preprocessing_pipeline.py)
    # batch_size = 16 (Table 3)
    # train_loader = ... 
    # val_loader = ...
    
    # List of models to train (Section 5.2)
    models_to_train = [
        "MambaNeuralOperator", # Our proposed method
        "TFT",                 # Temporal Fusion Transformer
        "PhytoLSTM",           # Bidirectional LSTM with gated metabolite attention
        "NODE_GM",             # Neural ODE Growth Model
        "S4",                  # Structured State Space Model
        "Mamba_Weather",       # Adapted Mamba-Weather
        "MambaDNA"             # Fine-tuned MambaDNA
    ]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # NOTE: In the actual repository, you would import these from a `models/` directory.
    # For this script to be executable as a demo, we assume a factory function exists:
    # from models.factory import get_model
    
    results_summary = {}
    
    for model_name in models_to_train:
        seed_results = []
        for seed in SEEDS:
            # model = get_model(model_name, hidden_dim=256, num_layers=4) # Per Table 3
            # Mocking the model for demonstration:
            class MockModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.linear = nn.Linear(68, 68) # 12 growth + 56 metabolites
                def forward(self, x, teacher_forcing_ratio=1.0):
                    out = self.linear(x)
                    return out, x, torch.sigmoid(x.mean(dim=-1, keepdim=True))
            
            model = MockModel()
            
            # Run training
            # best_loss = train_model(model, train_loader, val_loader, model_name, seed, device=device)
            # seed_results.append(best_loss)
            print(f"[DRY RUN] Trained {model_name} with seed {seed}. (Replace MockModel with actual architecture)")
            
        # results_summary[model_name] = {
        #     'mean_best_loss': np.mean(seed_results),
        #     'std_best_loss': np.std(seed_results)
        # }
        
    # print("\n" + "="*60)
    # print("FINAL REPRODUCIBILITY SUMMARY (Mean ± Std across 5 seeds)")
    # print("="*60)
    # for name, metrics in results_summary.items():
    #     print(f"{name:20s}: {metrics['mean_best_loss']:.4f} ± {metrics['std_best_loss']:.4f}")