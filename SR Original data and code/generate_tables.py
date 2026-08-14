"""
generate_tables.py
Generates Tables 1-4 from the manuscript in LaTeX and Markdown format.
Uses actual experimental results or manuscript-reported values.
"""

import numpy as np


def generate_table1_latex():
    """Table 1: Performance comparison (Section 6)."""
    
    data = {
        'Standard LSTM':      {'RMSE': '5.82±0.14', 'MAE': '4.15±0.11', 'F1': '0.78±0.03', 'Drift': '6.91±0.22', 'Modal': 'N/A'},
        'BiLSTM':             {'RMSE': '5.45±0.12', 'MAE': '3.89±0.09', 'F1': '0.81±0.02', 'Drift': '6.54±0.19', 'Modal': 'N/A'},
        'Neural ODE':         {'RMSE': '4.98±0.15', 'MAE': '3.52±0.13', 'F1': '0.75±0.04', 'Drift': '5.12±0.16', 'Modal': '0.45±0.08'},
        'Liquid NN':          {'RMSE': '4.65±0.11', 'MAE': '3.21±0.08', 'F1': '0.83±0.02', 'Drift': '5.88±0.21', 'Modal': 'N/A'},
        'PI-LSTM':            {'RMSE': '4.12±0.09', 'MAE': '2.95±0.07', 'F1': '0.85±0.02', 'Drift': '4.35±0.14', 'Modal': '0.38±0.06'},
        'ET-NODE-LNN':        {'RMSE': '3.27±0.06', 'MAE': '2.18±0.05', 'F1': '0.92±0.01', 'Drift': '2.45±0.09', 'Modal': '0.12±0.03'},
    }
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Performance comparison of the proposed model and baselines on the synthetic test set. Values represent mean $\pm$ standard deviation over five independent runs.}
\begin{tabular}{lccccc}
\hline
Model & RMSE ($10^{-2}$) & MAE ($10^{-2}$) & F1 Score & Drift Error ($10^{-2}$) & Modal Tracking Error (Hz) \\
\hline
"""
    for name, vals in data.items():
        bold = r'\textbf{' if name == 'ET-NODE-LNN' else ''
        bold_end = '}' if name == 'ET-NODE-LNN' else ''
        latex += f"{bold}{name}{bold_end} & {vals['RMSE']} & {vals['MAE']} & {vals['F1']} & {vals['Drift']} & {vals['Modal']} \\\\\n"
    
    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def generate_table2_latex():
    """Table 2: OC-SVM detector performance (Section 6)."""
    
    data = [
        ('0.05', '74.6±3.2', '3.8±0.9', '1.72±0.35', '0.72±0.03'),
        ('0.10', '86.3±2.5', '2.9±0.7', '1.28±0.28', '0.84±0.02'),
        ('0.20', '94.1±1.8', '2.1±0.5', '0.83±0.19', '0.93±0.02'),
        ('0.40', '98.2±0.9', '1.4±0.4', '0.47±0.12', '0.97±0.01'),
        ('0.60', '99.3±0.5', '1.1±0.3', '0.31±0.08', '0.98±0.01'),
        ('0.80', '99.7±0.3', '0.8±0.2', '0.22±0.06', '0.99±0.01'),
    ]
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Standalone performance of the OC-SVM anomaly detector across PGA levels.}
\begin{tabular}{ccccc}
\hline
PGA (g) & Detection Rate (\%) & False Alarm Rate (\%) & Mean Latency (s) & F1 Score \\
\hline
"""
    for row in data:
        latex += f"{' & '.join(row)} \\\\\n"
    
    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def generate_table3_latex():
    """Table 3: Ablation study (Section 6)."""
    
    data = {
        'Full Model':              {'RMSE': '3.27±0.06', 'Drift': '2.45±0.09', 'Modal': '0.12±0.03', 'F1': '0.92±0.01'},
        'w/o Event Reset':         {'RMSE': '3.45±0.08', 'Drift': '4.82±0.15', 'Modal': '0.13±0.03', 'F1': '0.90±0.02'},
        'w/o Hierarchical Reset':  {'RMSE': '3.52±0.09', 'Drift': '2.58±0.10', 'Modal': '0.13±0.03', 'F1': '0.89±0.02'},
        'w/o Damage Accumulator':  {'RMSE': '3.38±0.07', 'Drift': '2.52±0.09', 'Modal': '0.12±0.03', 'F1': '0.91±0.01'},
        'w/o Monotonicity Loss':   {'RMSE': '3.35±0.06', 'Drift': '2.49±0.10', 'Modal': '0.12±0.03', 'F1': '0.91±0.02'},
        'w/o Physics Constraints': {'RMSE': '3.58±0.07', 'Drift': '2.51±0.10', 'Modal': '0.35±0.05', 'F1': '0.91±0.01'},
        'w/o Attention Mechanism': {'RMSE': '3.62±0.09', 'Drift': '2.48±0.11', 'Modal': '0.12±0.03', 'F1': '0.86±0.03'},
        'w/o Liquid NN':           {'RMSE': '3.85±0.10', 'Drift': '2.60±0.12', 'Modal': '0.14±0.04', 'F1': '0.88±0.02'},
    }
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Ablation study results evaluating the contribution of individual components.}
\begin{tabular}{lcccc}
\hline
Variant & RMSE ($10^{-2}$) & Drift Error ($10^{-2}$) & Modal Tracking Error (Hz) & F1 Score \\
\hline
"""
    for name, vals in data.items():
        latex += f"{name} & {vals['RMSE']} & {vals['Drift']} & {vals['Modal']} & {vals['F1']} \\\\\n"
    
    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def generate_table4_latex():
    """Table 4: Computational complexity (Section 7.1)."""
    
    data = {
        'Standard LSTM':      {'Params': '1.35', 'Train': '2.4', 'Infer': '0.09', 'Mem': '1.1'},
        'BiLSTM':             {'Params': '2.68', 'Train': '4.1', 'Infer': '0.17', 'Mem': '2.0'},
        'Neural ODE':         {'Params': '0.82', 'Train': '9.7', 'Infer': '1.38', 'Mem': '3.6'},
        'Liquid NN':          {'Params': '1.64', 'Train': '6.3', 'Infer': '0.72', 'Mem': '2.9'},
        'PI-LSTM':            {'Params': '1.41', 'Train': '3.2', 'Infer': '0.11', 'Mem': '1.4'},
        'ET-NODE-LNN':        {'Params': '3.95', 'Train': '15.8', 'Infer': '2.46', 'Mem': '6.4'},
    }
    
    latex = r"""
\begin{table}[htbp]
\centering
\caption{Computational complexity comparison.}
\begin{tabular}{lcccc}
\hline
Model & Parameters (M) & Training Time (h) & Inference Speed (ms/batch) & Peak GPU Memory (GB) \\
\hline
"""
    for name, vals in data.items():
        bold = r'\textbf{' if name == 'ET-NODE-LNN' else ''
        bold_end = '}' if name == 'ET-NODE-LNN' else ''
        latex += f"{bold}{name}{bold_end} & {vals['Params']} & {vals['Train']} & {vals['Infer']} & {vals['Mem']} \\\\\n"
    
    latex += r"""
\hline
\end{tabular}
\end{table}
"""
    return latex


def generate_all_tables():
    """Generate all tables and save to files."""
    
    print("Generating Table 1 (Performance Comparison)...")
    with open('table1_performance.tex', 'w') as f:
        f.write(generate_table1_latex())
    
    print("Generating Table 2 (OC-SVM Detector)...")
    with open('table2_ocsvm.tex', 'w') as f:
        f.write(generate_table2_latex())
    
    print("Generating Table 3 (Ablation Study)...")
    with open('table3_ablation.tex', 'w') as f:
        f.write(generate_table3_latex())
    
    print("Generating Table 4 (Computational Complexity)...")
    with open('table4_complexity.tex', 'w') as f:
        f.write(generate_table4_latex())
    
    print("\nAll tables generated successfully!")
    
    # Also print to console
    print("\n" + "="*60)
    print("TABLE 1: Performance Comparison")
    print("="*60)
    print(f"{'Model':<20} {'RMSE':>12} {'MAE':>12} {'F1':>12} {'Drift':>12} {'Modal':>12}")
    print("-"*80)
    
    table1_data = [
        ('Standard LSTM', '5.82±0.14', '4.15±0.11', '0.78±0.03', '6.91±0.22', 'N/A'),
        ('BiLSTM', '5.45±0.12', '3.89±0.09', '0.81±0.02', '6.54±0.19', 'N/A'),
        ('Neural ODE', '4.98±0.15', '3.52±0.13', '0.75±0.04', '5.12±0.16', '0.45±0.08'),
        ('Liquid NN', '4.65±0.11', '3.21±0.08', '0.83±0.02', '5.88±0.21', 'N/A'),
        ('PI-LSTM', '4.12±0.09', '2.95±0.07', '0.85±0.02', '4.35±0.14', '0.38±0.06'),
        ('ET-NODE-LNN', '3.27±0.06', '2.18±0.05', '0.92±0.01', '2.45±0.09', '0.12±0.03'),
    ]
    for row in table1_data:
        print(f"{row[0]:<20} {row[1]:>12} {row[2]:>12} {row[3]:>12} {row[4]:>12} {row[5]:>12}")


if __name__ == '__main__':
    generate_all_tables()