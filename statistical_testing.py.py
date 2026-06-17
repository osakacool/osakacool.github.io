"""
statistical_testing.py
Statistical testing and robustness evaluation suite for Plant Growth-Metabolism Modeling.
Implements the exact protocols described in Sections 5.4, 5.5, and 6.7 of the manuscript:
1. Paired t-tests with Bonferroni correction for multiple comparisons.
2. Noise Robustness Test (Gaussian noise injection at 20, 10, 5 dB SNR).
3. Abrupt Environmental Shock Test (RMSE evaluation before/after shocks).
4. Cross-Species Generalization Statistical Analysis.

Author: [Your Name/Team]
License: MIT
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Dict, Tuple, Optional

# ==============================================================================
# 1. Statistical Significance Testing (Section 5.3, 5.5, 6.1)
# ==============================================================================
class StatisticalTester:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def paired_ttest_bonferroni(self, metrics_ours: np.ndarray, metrics_baseline: np.ndarray, 
                                metric_name: str, num_comparisons: int = 1) -> Dict:
        """
        Performs a paired t-test between our model and a baseline across 5 random seeds,
        and applies the Bonferroni correction for multiple comparisons.
        
        Args:
            metrics_ours: Array of metric values for our model (shape: [num_seeds], e.g., 5).
            metrics_baseline: Array of metric values for the baseline model.
            metric_name: Name of the metric (e.g., 'TPA', 'LDS', 'MGCI').
            num_comparisons: Total number of pairwise comparisons being made (for Bonferroni).
        """
        if len(metrics_ours) != len(metrics_baseline):
            raise ValueError("Both models must have the same number of seeds (e.g., 5).")
            
        # 1. Perform Paired t-test (two-tailed)
        t_stat, p_value = stats.ttest_rel(metrics_ours, metrics_baseline)
        
        # 2. Apply Bonferroni Correction
        p_value_corrected = min(p_value * num_comparisons, 1.0)
        
        # 3. Determine significance stars based on corrected p-value
        if p_value_corrected < 0.001:
            stars = "***"
        elif p_value_corrected < 0.01:
            stars = "**"
        elif p_value_corrected < 0.05:
            stars = "*"
        else:
            stars = "ns" # not significant
            
        return {
            'Metric': metric_name,
            'Ours (Mean±Std)': f"{np.mean(metrics_ours):.2f}±{np.std(metrics_ours):.2f}",
            'Baseline (Mean±Std)': f"{np.mean(metrics_baseline):.2f}±{np.std(metrics_baseline):.2f}",
            't-statistic': t_stat,
            'p-value (raw)': p_value,
            'p-value (Bonferroni)': p_value_corrected,
            'Significance': stars
        }

# ==============================================================================
# 2. Robustness Evaluations (Section 5.4, 6.7.1)
# ==============================================================================
class RobustnessEvaluator:
    @staticmethod
    def calculate_tpa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Helper to calculate TPA (100 * (1 - NRMSE))."""
        mse = np.mean((y_true - y_pred) ** 2)
        var = np.var(y_true) + 1e-8
        nrmse = np.sqrt(mse / var)
        return max(0.0, min(100.0, 100.0 * (1.0 - nrmse)))

    def noise_injection_test(self, y_true: np.ndarray, y_pred_clean: np.ndarray, 
                             snr_levels: List[int] = [20, 10, 5]) -> pd.DataFrame:
        """
        Simulates sensor measurement errors by adding Gaussian noise to predictions.
        Calculates TPA under different Signal-to-Noise Ratios (SNR).
        Corresponds to Section 6.7.1 and Table 8.
        
        Math: SNR_dB = 10 * log10(P_signal / P_noise)
              P_noise = P_signal / 10^(SNR_dB / 10)
              sigma_noise = sqrt(P_noise)
        """
        results = []
        signal_power = np.var(y_pred_clean) + 1e-8
        
        # Clean prediction
        tpa_clean = self.calculate_tpa(y_true, y_pred_clean)
        results.append({'Condition': 'Clean', 'TPA': tpa_clean})
        
        for snr_db in snr_levels:
            # Calculate noise standard deviation based on SNR formula
            noise_power = signal_power / (10 ** (snr_db / 10.0))
            noise_std = np.sqrt(noise_power)
            
            # Inject Gaussian noise (simulating different random sensor errors)
            np.random.seed(42) # For reproducible noise injection
            noise = np.random.normal(0, noise_std, y_pred_clean.shape)
            y_pred_noisy = y_pred_clean + noise
            
            tpa_noisy = self.calculate_tpa(y_true, y_pred_noisy)
            results.append({'Condition': f'{snr_db} dB', 'TPA': tpa_noisy})
            
        return pd.DataFrame(results)

    def abrupt_shock_test(self, rmse_before_shock: float, rmse_after_shock: float) -> Dict:
        """
        Evaluates prediction stability under sudden environmental shocks 
        (e.g., temperature drop 28°C -> 16°C, or sudden rewatering).
        Corresponds to Section 6.7.2.
        """
        degradation_ratio = (rmse_after_shock - rmse_before_shock) / rmse_before_shock
        return {
            'RMSE_Before': rmse_before_shock,
            'RMSE_After': rmse_after_shock,
            'Degradation_Ratio': degradation_ratio,
            'Stability_Assessment': 'High' if degradation_ratio < 0.5 else 'Low'
        }

# ==============================================================================
# 3. Cross-Species Generalization Analysis (Section 5.4, 6.7.3)
# ==============================================================================
class GeneralizationEvaluator:
    def summarize_cross_species(self, zero_shot_metrics: Dict[str, np.ndarray], 
                                fine_tuned_metrics: Dict[str, np.ndarray]) -> pd.DataFrame:
        """
        Summarizes zero-shot and 10% fine-tuned performance on target species.
        Corresponds to Table 9.
        """
        rows = []
        for species in zero_shot_metrics.keys():
            z_mean, z_std = np.mean(zero_shot_metrics[species]), np.std(zero_shot_metrics[species])
            f_mean, f_std = np.mean(fine_tuned_metrics[species]), np.std(fine_tuned_metrics[species])
            
            rows.append({
                'Target Species': species,
                'TPA (zero-shot)': f"{z_mean:.1f}±{z_std:.1f}",
                'TPA (fine-tuned 10%)': f"{f_mean:.1f}±{f_std:.1f}",
                'LDS (fine-tuned)': f"{np.mean(fine_tuned_metrics[species+'_LDS']):.2f}±{np.std(fine_tuned_metrics[species+'_LDS']):.2f}"
            })
        return pd.DataFrame(rows)

# ==============================================================================
# 4. Execution Example (Demonstrates Reproduction of Manuscript Tables)
# ==============================================================================
if __name__ == "__main__":
    np.random.seed(42)
    SEEDS = 5
    
    print("="*70)
    print("1. STATISTICAL SIGNIFICANCE TESTING (Reproducing Section 5.5 & 6.1)")
    print("="*70)
    
    # Simulate 5-seed results for TPA (Mean ~92.8, Std ~0.6 for Ours; Mean ~83.5, Std ~0.6 for S4)
    tpa_ours = np.random.normal(92.8, 0.6, SEEDS)
    tpa_s4 = np.random.normal(83.5, 0.6, SEEDS)
    tpa_tft = np.random.normal(82.3, 0.7, SEEDS)
    
    tester = StatisticalTester()
    # We are comparing Ours vs 3 baselines, so num_comparisons = 3
    num_comparisons = 3 
    
    for name, baseline_tpa in [("S4", tpa_s4), ("TFT", tpa_tft)]:
        res = tester.paired_ttest_bonferroni(tpa_ours, baseline_tpa, "TPA", num_comparisons)
        print(f"Ours vs {name}: {res['Ours (Mean±Std)']} vs {res['Baseline (Mean±Std)']}")
        print(f"  -> Raw p-value: {res['p-value (raw)']:.2e} | Bonferroni p-value: {res['p-value (Bonferroni)']:.2e} | Significance: {res['Significance']}")

    print("\n" + "="*70)
    print("2. NOISE ROBUSTNESS TEST (Reproducing Table 8)")
    print("="*70)
    
    # Simulate a sequence of true and clean predicted values
    seq_len = 1000
    y_true = np.sin(np.linspace(0, 10*np.pi, seq_len)) + np.random.normal(0, 0.1, seq_len)
    y_pred_clean = y_true + np.random.normal(0, 0.05, seq_len) # High accuracy clean prediction
    
    evaluator = RobustnessEvaluator()
    noise_df = evaluator.noise_injection_test(y_true, y_pred_clean, snr_levels=[20, 10, 5])
    print("Model: Ours (Simulated)")
    print(noise_df.to_string(index=False))
    
    # Simulate TFT (lower clean TPA, drops faster under noise)
    y_pred_clean_tft = y_true + np.random.normal(0, 0.15, seq_len)
    noise_df_tft = evaluator.noise_injection_test(y_true, y_pred_clean_tft, snr_levels=[20, 10, 5])
    print("\nModel: TFT (Simulated)")
    print(noise_df_tft.to_string(index=False))

    print("\n" + "="*70)
    print("3. ABRUPT ENVIRONMENTAL SHOCK TEST (Section 6.7.2)")
    print("="*70)
    
    # Ours: RMSE 0.32 -> 0.57
    shock_ours = evaluator.abrupt_shock_test(rmse_before_shock=0.32, rmse_after_shock=0.57)
    print(f"Ours: RMSE {shock_ours['RMSE_Before']} -> {shock_ours['RMSE_After']} (Degradation: {shock_ours['Degradation_Ratio']:.1%}, Stability: {shock_ours['Stability_Assessment']})")
    
    # TFT: RMSE 0.58 -> 1.23
    shock_tft = evaluator.abrupt_shock_test(rmse_before_shock=0.58, rmse_after_shock=1.23)
    print(f"TFT:  RMSE {shock_tft['RMSE_Before']} -> {shock_tft['RMSE_After']} (Degradation: {shock_tft['Degradation_Ratio']:.1%}, Stability: {shock_tft['Stability_Assessment']})")

    print("\n" + "="*70)
    print("4. CROSS-SPECIES GENERALIZATION (Reproducing Table 9)")
    print("="*70)
    
    gen_eval = GeneralizationEvaluator()
    # Simulate 5-seed results for cross-species TPA and LDS
    zero_shot = {
        'Oryza sativa': np.random.normal(81.2, 1.2, SEEDS),
        'Zea mays': np.random.normal(78.5, 1.4, SEEDS)
    }
    fine_tuned = {
        'Oryza sativa': np.random.normal(88.4, 0.7, SEEDS),
        'Oryza sativa_LDS': np.random.normal(0.79, 0.02, SEEDS),
        'Zea mays': np.random.normal(86.1, 0.8, SEEDS),
        'Zea mays_LDS': np.random.normal(0.77, 0.03, SEEDS)
    }
    
    gen_df = gen_eval.summarize_cross_species(zero_shot, fine_tuned)
    print(gen_df.to_string(index=False))
    
    print("\n[SUCCESS] All statistical tests and robustness evaluations executed.")
    print("Note: In the actual repository, replace simulated data with real model outputs.")