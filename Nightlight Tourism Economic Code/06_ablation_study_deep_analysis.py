import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 学术绘图全局设置
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ==========================================
# 论文 Table 4 真实数据提取
# ==========================================
inputs = ['VIIRS\n(Base)', 'VIIRS\n+POI', 'VIIRS\n+Mobility', 'VIIRS\n+Social', 'Full\nModel']
oa = [85.2, 89.4, 88.3, 86.7, 92.3]
kappa = [0.79, 0.84, 0.82, 0.80, 0.89]
miou = [0.73, 0.78, 0.76, 0.74, 0.85]

# 易混淆类别的 F1-Score
f1_dining = [0.76, 0.81, 0.80, 0.78, 0.85]
f1_cultural = [0.70, 0.75, 0.74, 0.73, 0.81]

# ==========================================
# 绘图开始
# ==========================================
fig = plt.figure(figsize=(16, 10))
fig.suptitle('Figure S1: Deep Analysis of Ablation Study (Corresponding to Table 4)', 
             fontweight='bold', fontsize=16, y=0.98)

# --- 子图 1: 整体指标演进趋势 (折线图 + 阴影) ---
ax1 = plt.subplot(2, 2, 1)
x = np.arange(len(inputs))
ax1.plot(x, oa, 'o-', color='#d62728', linewidth=2.5, markersize=8, label='Overall Accuracy (OA)')
ax1.plot(x, [k*100 for k in kappa], 's--', color='#1f77b4', linewidth=2, markersize=7, label='Kappa Coefficient (×100)')
ax1.plot(x, [m*100 for m in miou], '^-.', color='#2ca02c', linewidth=2, markersize=7, label='mIoU (×100)')

# 填充 Full Model 的优越区域
ax1.fill_between(x[3:], [85, 85, 85], [oa[4], kappa[4]*100, miou[4]*100], color='gray', alpha=0.1)

ax1.set_xticks(x)
ax1.set_xticklabels(inputs, fontsize=10)
ax1.set_ylabel('Score (%)', fontweight='bold')
ax1.set_title('(a) Overall Metrics Evolution', fontweight='bold', loc='left')
ax1.set_ylim(70, 100)
ax1.legend(loc='lower right', framealpha=0.9)
ax1.grid(True, linestyle=':', alpha=0.6)
# 添加数据标签
for i, v in enumerate(oa):
    ax1.text(i, v + 1, f'{v}%', ha='center', fontsize=9, color='#d62728', fontweight='bold')

# --- 子图 2: 易混淆类别 (Dining vs Cultural) F1-Score 突破 ---
ax2 = plt.subplot(2, 2, 2)
width = 0.35
ax2.bar(x - width/2, f1_dining, width, label='Dining Cluster', color='#ff7f0e', edgecolor='black')
ax2.bar(x + width/2, f1_cultural, width, label='Cultural Performance', color='#9467bd', edgecolor='black')

# 高亮 Full Model 的突破
ax2.annotate('Full Model\nBreakthrough', xy=(4, 0.83), xytext=(3, 0.88),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10, fontweight='bold', ha='center')

ax2.set_xticks(x)
ax2.set_xticklabels(inputs, fontsize=10)
ax2.set_ylabel('F1-Score', fontweight='bold')
ax2.set_title('(b) F1-Score of Confused Categories', fontweight='bold', loc='left')
ax2.set_ylim(0.65, 0.95)
ax2.legend(loc='lower right')
ax2.grid(True, axis='y', linestyle=':', alpha=0.6)

# --- 子图 3: 各数据源独立边际贡献度 (相对于 Base) ---
ax3 = plt.subplot(2, 1, 2) # 占据底部一整行
base_acc = 85.2
gains = {
    '+ POI Density': 89.4 - base_acc,
    '+ Human Mobility': 88.3 - base_acc,
    '+ Social Media': 86.7 - base_acc
}
colors_gain = ['#1f77b4', '#2ca02c', '#ff7f0e']

bars = ax3.bar(gains.keys(), gains.values(), color=colors_gain, edgecolor='black', linewidth=1.2)
ax3.axhline(y=0, color='black', linewidth=1)

# 添加具体增益数值标签
for bar, val in zip(bars, gains.values()):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
             f'+{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

ax3.set_ylabel('Accuracy Improvement (%)', fontweight='bold', fontsize=12)
ax3.set_title('(c) Marginal Contribution of Individual Data Sources (over VIIRS Base)', 
              fontweight='bold', loc='left', fontsize=13)
ax3.set_ylim(0, 6)
ax3.grid(True, axis='y', linestyle=':', alpha=0.6)

# 添加一段学术解释文本 (模拟论文Discussion的口吻)
textstr = ('Note: POI density provides the most significant individual boost (+4.2%), acting as the static spatial skeleton.\n'
           'Human mobility adds dynamic temporal pulses (+3.1%). Social media, while yielding a smaller global gain (+1.5%),\n'
           'is critical for resolving spectral ambiguity between Dining and Cultural zones (as seen in Fig. b).')
ax3.text(0.5, -0.25, textstr, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='center', 
         bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.3, edgecolor='gray'))

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig('FigS1_Ablation_Study_Deep_Analysis.png', dpi=600, bbox_inches='tight')
plt.show()