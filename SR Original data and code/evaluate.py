"""
evaluate.py
Section 5.4: Evaluation Metrics
- RMSE (Eq. 27)
- MAE (Eq. 28)
- F1 Score (threshold at 0.5)
- Drift Error (RMSE on post-event ambient segments)
- Modal Frequency Tracking Error
"""

import torch
import numpy as np
from sklearn.metrics import f1_score
from config import CFG
from model.physics_loss import PhysicsInformedLoss


def compute_rmse(y_pred, y_true):
    """Eq. 27: RMSE = sqrt(1/N Σ (ŷ(t) - y(t))²)"""
    return np.sqrt(np.mean((y_pred - y_true) ** 2))


def compute_mae(y_pred, y_true):
    """Eq. 28: MAE = 1/N Σ |ŷ(t) - y(t)|"""
    return np.mean(np.abs(y_pred - y_true))


def compute_f1(y_pred, y_true, threshold=None):
    """F1 score for binary classification at threshold 0.5."""
    threshold = threshold or CFG.HIGH_VUL_THRESHOLD  # 0.5
    y_pred_binary = (y_pred > threshold).astype(int)
    y_true_binary = (y_true > threshold).astype(int)
    return f1_score(y_true_binary, y_pred_binary, zero_division=0)


def compute_drift_error(y_pred, y_true, event_flags, post_event_window=1000):
    """
    Drift Error: RMSE computed only on ambient vibration segments
    that follow a seismic event (Section 5.4).
    
    Args:
        y_pred: (N,) predicted vulnerability
        y_true: (N,) ground truth
        event_flags: (N,) boolean event markers
        post_event_window: number of samples after event to evaluate
    """
    # Find event end indices
    changes = np.diff(event_flags.astype(int))
    event_ends = np.where(changes == -1)[0] + 1
    
    drift_errors = []
    for end_idx in event_ends:
        start = end_idx
        end = min(end_idx + post_event_window, len(y_pred))
        if start < len(y_pred):
            segment_pred = y_pred[start:end]
            segment_true = y_true[start:end]
            rmse = np.sqrt(np.mean((segment_pred - segment_true) ** 2))
            drift_errors.append(rmse)
    
    return np.mean(drift_errors) if drift_errors else 0.0


def compute_modal_tracking_error(z_latent, true_freqs=None, sampling_rate=None):
    """
    Modal Frequency Tracking Error (Section 5.4):
    Mean absolute difference between extracted and true modal frequencies.
    """
    true_freqs = true_freqs or CFG.TRUE_MODAL_FREQS_HZ
    sampling_rate = sampling_rate or CFG.SAMPLING_RATE_HZ
    
    physics_loss_fn = PhysicsInformedLoss(
        true_freqs=true_freqs,
        sampling_rate=sampling_rate
    )
    
    if isinstance(z_latent, np.ndarray):
        z_latent = torch.tensor(z_latent, dtype=torch.float32)
    
    if z_latent.dim() == 2:
        z_latent = z_latent.unsqueeze(0)
    
    inst_freqs = physics_loss_fn.extract_instantaneous_frequencies(z_latent)
    
    true_freqs_tensor = torch.tensor(true_freqs, dtype=torch.float32)
    inst_sorted, _ = torch.sort(inst_freqs.squeeze(0))
    true_sorted, _ = torch.sort(true_freqs_tensor)
    
    tracking_error = torch.mean(torch.abs(inst_sorted - true_sorted)).item()
    
    return tracking_error


def evaluate_model(model, test_loader, device, is_full_model=False):
    """
    Complete evaluation pipeline computing all metrics.
    
    Returns dict with: RMSE, MAE, F1, Drift Error, Modal Tracking Error
    """
    model.eval()
    all_preds = []
    all_trues = []
    all_events = []
    all_latents = []
    
    with torch.no_grad():
        for batch_x, batch_y, batch_event in test_loader:
            batch_x = batch_x.unsqueeze(0).to(device)
            batch_event = batch_event.unsqueeze(0).bool().to(device)
            
            if is_full_model:
                vul_pred, aux = model(batch_x, event_flags=batch_event)
                all_latents.append(aux.get('z_ode', None))
            else:
                vul_pred, _ = model(batch_x)
            
            all_preds.append(vul_pred.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
            all_events.append(batch_event.cpu().numpy())
    
    y_pred = np.concatenate(all_preds, axis=1).squeeze()
    y_true = np.concatenate(all_trues, axis=1).squeeze()
    event_flags = np.concatenate(all_events, axis=1).squeeze()
    
    metrics = {
        'RMSE': compute_rmse(y_pred, y_true),
        'MAE': compute_mae(y_pred, y_true),
        'F1_Score': compute_f1(y_pred, y_true),
        'Drift_Error': compute_drift_error(y_pred, y_true, event_flags),
    }
    
    # Modal tracking error (only for models with physics constraints)
    if all_latents and all_latents[0] is not None:
        z_cat = torch.cat(all_latents, dim=1)
        metrics['Modal_Tracking_Error_Hz'] = compute_modal_tracking_error(z_cat)
    else:
        metrics['Modal_Tracking_Error_Hz'] = None
    
    return metrics


def run_evaluation_5_seeds(model_class, model_name, seeds=None):
    """
    Evaluate over 5 independent runs (Section 5.4).
    Report mean ± std for each metric.
    """
    seeds = seeds or [42, 43, 44, 45, 46]
    all_metrics = []
    
    for seed in seeds:
        # Load trained model
        model = model_class()
        ckpt_path = f'checkpoints/{model_name}_seed{seed}_best.pth'
        
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
        except FileNotFoundError:
            print(f"Checkpoint not found: {ckpt_path}")
            continue
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Create test loader
        from train import create_dataloaders
        _, _, test_loader = create_dataloaders()
        
        is_full = (model_name == 'et_node_lnn')
        metrics = evaluate_model(model, test_loader, device, is_full)
        all_metrics.append(metrics)
    
    # Aggregate: mean ± std
    summary = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics if m[key] is not None]
        if values:
            summary[key] = f"{np.mean(values):.4f} ± {np.std(values):.4f}"
        else:
            summary[key] = "N/A"
    
    return summary