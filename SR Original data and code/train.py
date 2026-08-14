"""
train.py
End-to-end training pipeline for ET-NODE-LNN and all baselines.
Section 5.2: Adam optimizer, lr=1e-3, decay 0.5 every 50 epochs,
             gradient clipping max_norm=1.0, 200 epochs.
Section 5.4: 5 independent runs with different seeds.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import json
import time
from config import CFG
from data_generator import SyntheticTimberHallDataset
from model.et_node_lnn import ETNODELNN
from baselines import get_model


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(batch_size=None):
    """Create train/val/test dataloaders (Section 5.1: 70/15/15 split)."""
    batch_size = batch_size or CFG.BATCH_SIZE  # 256
    
    full_dataset = SyntheticTimberHallDataset(
        duration_hours=24,
        seed=CFG.RANDOM_SEED
    )
    
    n = len(full_dataset)
    train_size = int(n * CFG.TRAIN_SPLIT)    # 70%
    val_size = int(n * CFG.VAL_SPLIT)        # 15%
    test_size = n - train_size - val_size     # 15%
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(CFG.RANDOM_SEED)
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    test_loader = DataLoader(test_ds, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


def train_one_epoch(model, train_loader, optimizer, criterion, device, 
                    is_full_model=False):
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch_x, batch_y, batch_event in train_loader:
        batch_x = batch_x.unsqueeze(0).to(device)      # (1, features)
        batch_y = batch_y.unsqueeze(0).unsqueeze(-1).to(device)
        batch_event = batch_event.unsqueeze(0).bool().to(device)
        
        optimizer.zero_grad()
        
        if is_full_model:
            vul_pred, aux = model(batch_x, event_flags=batch_event)
            loss, _ = model.compute_loss(vul_pred, batch_y, aux)
        else:
            vul_pred, _ = model(batch_x)
            loss = criterion(vul_pred, batch_y)
        
        loss.backward()
        
        # Gradient clipping (Section 5.2: max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.MAX_GRAD_NORM
        )
        
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / max(n_batches, 1)


def validate(model, val_loader, criterion, device, is_full_model=False):
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for batch_x, batch_y, batch_event in val_loader:
            batch_x = batch_x.unsqueeze(0).to(device)
            batch_y = batch_y.unsqueeze(0).unsqueeze(-1).to(device)
            batch_event = batch_event.unsqueeze(0).bool().to(device)
            
            if is_full_model:
                vul_pred, aux = model(batch_x, event_flags=batch_event)
                loss, _ = model.compute_loss(vul_pred, batch_y, aux)
            else:
                vul_pred, _ = model(batch_x)
                loss = criterion(vul_pred, batch_y)
            
            total_loss += loss.item()
            n_batches += 1
    
    return total_loss / max(n_batches, 1)


def train_model(model_name='et_node_lnn', seed=None, save_dir='checkpoints'):
    """
    Train a single model with full configuration from Section 5.2.
    """
    seed = seed or CFG.RANDOM_SEED
    set_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training {model_name} | Seed={seed} | Device={device}")
    
    # Create model
    is_full_model = (model_name == 'et_node_lnn')
    if is_full_model:
        model = ETNODELNN(input_dim=5, num_channels=5)
    else:
        model = get_model(model_name)
    
    model = model.to(device)
    
    # Count parameters (Table 4)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Parameters: {n_params:.2f}M")
    
    # Data
    train_loader, val_loader, test_loader = create_dataloaders()
    
    # Optimizer: Adam, lr=1e-3 (Section 5.2)
    optimizer = optim.Adam(model.parameters(), lr=CFG.INITIAL_LR)
    
    # LR Scheduler: decay by 0.5 every 50 epochs (Section 5.2)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=CFG.LR_DECAY_EVERY_EPOCHS,  # 50
        gamma=CFG.LR_DECAY_FACTOR              # 0.5
    )
    
    criterion = nn.MSELoss()
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    start_time = time.time()
    
    for epoch in range(CFG.NUM_EPOCHS):  # 200 epochs
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, is_full_model
        )
        val_loss = validate(model, val_loader, criterion, device, is_full_model)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(save_dir, exist_ok=True)
            torch.save(model.state_dict(), 
                      f'{save_dir}/{model_name}_seed{seed}_best.pth')
        
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{CFG.NUM_EPOCHS} | "
                  f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    training_time = time.time() - start_time
    print(f"  Training time: {training_time:.1f}s")
    
    return model, history, training_time, n_params


def run_all_experiments():
    """
    Run all models × 5 seeds (Section 5.4).
    Models: ET-NODE-LNN + 5 baselines
    Seeds: 5 independent runs
    """
    model_names = [
        'et_node_lnn',
        'standard_lstm',
        'bilstm',
        'neural_ode',
        'liquid_nn',
        'pi_lstm'
    ]
    
    seeds = [42, 43, 44, 45, 46]  # 5 independent runs
    results = {}
    
    for model_name in model_names:
        results[model_name] = []
        for seed in seeds:
            print(f"\n{'='*60}")
            print(f"Model: {model_name} | Seed: {seed}")
            print(f"{'='*60}")
            
            model, history, train_time, n_params = train_model(
                model_name, seed
            )
            
            results[model_name].append({
                'seed': seed,
                'train_time': train_time,
                'n_params_M': n_params,
                'final_train_loss': history['train_loss'][-1],
                'final_val_loss': history['val_loss'][-1]
            })
    
    # Save results
    with open('experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\nAll experiments completed. Results saved to experiment_results.json")
    return results


if __name__ == '__main__':
    run_all_experiments()