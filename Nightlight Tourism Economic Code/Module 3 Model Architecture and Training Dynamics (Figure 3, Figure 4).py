import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# ==========================================
# Figure 3: DeepLabV3+ 网络结构图 (Network Architecture)
# ==========================================
def plot_model_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # 定义模块样式
    def draw_block(x, y, w, h, text, color):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", edgecolor='black', facecolor=color, linewidth=1.5)
        ax.add_patch(box)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=10, fontweight='bold')
        
    def draw_arrow(x1, y1, x2, y2):
        arrow = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', mutation_scale=20, linewidth=1.5)
        ax.add_patch(arrow)

    # 1. Input (5 Channels)
    draw_block(0.5, 2.5, 1.5, 1, 'Input\n(5-Ch Tensor)', '#e0f7fa')
    # 2. ResNet-50 Backbone
    draw_block(2.5, 2.5, 2, 1, 'ResNet-50\nBackbone', '#b2ebf2')
    # 3. ASPP Module
    draw_block(5.0, 2.5, 2, 1, 'ASPP Module\n(Rates: 6,12,18)', '#80deea')
    # 4. Decoder
    draw_block(7.5, 2.5, 1.5, 1, 'Decoder\n(1x1 Conv)', '#4dd0e1')
    # 5. Output
    draw_block(9.5, 2.5, 1.5, 1, 'Softmax\n(4 Classes)', '#26c6da')
    
    # 侧边：CRF Post-processing
    draw_block(5.0, 0.5, 4, 1, 'CRF Post-processing\n(Reduce Misclassification 23%)', '#ffcc80')
    
    # 连接箭头
    draw_arrow(2.0, 3.0, 2.5, 3.0)
    draw_arrow(4.5, 3.0, 5.0, 3.0)
    draw_arrow(7.0, 3.0, 7.5, 3.0)
    draw_arrow(9.0, 3.0, 9.5, 3.0)
    draw_arrow(6.0, 2.5, 6.0, 1.5)
    draw_arrow(7.0, 1.5, 7.0, 2.5)
    
    ax.set_title('Figure 3: DeepLabV3+ Model Architecture with Multi-Source Input', fontweight='bold', fontsize=12, pad=20)
    plt.tight_layout()
    plt.savefig('Fig3_Model_Architecture.png', dpi=600, bbox_inches='tight')
    plt.show()

# ==========================================
# Figure 4: 训练动态曲线 (Training Dynamics)
# ==========================================
def plot_training_curves():
    epochs = np.arange(1, 76)
    # 模拟论文描述的曲线：Loss 指数下降至 0.15-0.20，Acc 上升至 92% 并稳定
    train_loss = 2.5 * np.exp(-0.08 * epochs) + 0.15 + np.random.normal(0, 0.02, len(epochs))
    val_acc = 0.92 * (1 - np.exp(-0.1 * epochs)) + np.random.normal(0, 0.01, len(epochs))
    val_acc = np.clip(val_acc, 0, 0.95)
    
    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color1 = 'tab:red'
    ax1.set_xlabel('Epochs', fontweight='bold')
    ax1.set_ylabel('Training Loss', color=color1, fontweight='bold')
    ax1.plot(epochs, train_loss, color=color1, linewidth=2, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(0, 3.0)
    
    ax2 = ax1.twinx()
    color2 = 'tab:blue'
    ax2.set_ylabel('Validation Accuracy', color=color2, fontweight='bold')
    ax2.plot(epochs, val_acc, color=color2, linewidth=2, label='Validation Accuracy')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.0)
    
    # 添加图例
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right')
    
    plt.title('Figure 4: Training Dynamics (Loss & Accuracy over 75 Epochs)', fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('Fig4_Training_Curves.png', dpi=600, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    plot_model_architecture()
    plot_training_curves()