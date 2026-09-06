import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE

plt.rcParams['font.size']=10

def plot_figure1_system_architecture():
    """Figure1：Longitudinal SEL Tracking 系统框图，文本绘制框图（和论文图片布局对齐）"""
    fig,ax = plt.subplots(figsize=(11,10))
    ax.set_xlim(0,10)
    ax.set_ylim(0,10)
    ax.axis('off')
    boxes = [
        {"text":"Audio Input Stage\nMicrophone Array → Audio Interface → Pre Amplifier","xy":(0.2,8.6),"w":9.5,"h":1.2},
        {"text":"Signal Processing Stage\nBandpass Filter → Noise Gate → Normalization Module","xy":(0.2,6.8),"w":9.5,"h":1.2},
        {"text":"Feature Extraction Stage\nMFCC / Zero‑Crossing / Chroma / Spectral Contrast → Feature Concat","xy":(0.2,5.0),"w":9.5,"h":1.2},
        {"text":"Liquid‑Mamba Hybrid Processing\nLiquid‑Time‑Constant Solver → Adaptive Integrator → Selective SSM Block","xy":(0.2,3.2),"w":9.5,"h":1.2},
        {"text":"Assessment and Output\nSEL Prediction Head → Softmax Classifier → Visualization Logger","xy":(0.2,1.4),"w":9.5,"h":1.2},
    ]
    for b in boxes:
        rect = plt.matplotlib.patches.Rectangle(b["xy"],b["w"],b["h"],ec="black",linestyle="--",fc="none")
        ax.add_patch(rect)
        ax.text(b["xy"][0]+0.3, b["xy"][1]+0.6, b["text"],va="center")
    # draw arrow
    for ypos in [8.6,6.8,5.0,3.2]:
        ax.annotate("", xy=(5, ypos-0.1), xytext=(5,ypos-1.0), arrowprops=dict(arrowstyle="->"))
    plt.title("Figure1 Overall System Architecture for Longitudinal SEL Tracking")
    plt.tight_layout()
    plt.savefig("fig1_architecture.png",dpi=300)
    plt.close()

def plot_figure2_ltc_time_constant():
    """Figure2 LTC自适应时间常数 +音频能量包络"""
    fig,ax1 = plt.subplots(figsize=(12,5))
    t = np.arange(0,300,1)
    audio_energy = np.random.rand(len(t))*0.8
    tau_eff = np.ones(len(t))*0.7
    # 模拟情绪突变点
    for pos in [60,140,230]:
        tau_eff[pos:pos+10] = 0.3
        audio_energy[pos:pos+10] = 0.95

    ax1.plot(t, audio_energy, color="#2266bb", label="Audio Energy Envelope")
    ax1.set_ylabel("Normalized Audio Energy")
    ax2 = ax1.twinx()
    ax2.plot(t, tau_eff, color="#dd2222", label=r"Adaptive Time Constant $1/\tau(t)$")
    ax2.set_ylabel(r"Adaptive Time Constant ($s$)")
    ax1.vlines([60,140,230], ymin=0,ymax=1,linestyle="--",color="green",alpha=0.6)
    ax1.set_xlabel("Time (seconds)")
    plt.title("Figure2 Evolution of adaptive time constants from LTC module")
    fig.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("fig2_ltc_tau.png",dpi=300)
    plt.close()

def plot_figure3_mamba_selection_heatmap():
    """Figure3 Mamba输入依赖选择权重热力图"""
    fig,ax = plt.subplots(figsize=(13,7))
    mat = np.random.rand(64,500)*0.2
    # 模拟高激活垂直条带
    for col in [80,180,280,380,470]:
        mat[10:50, col:col+15] +=0.7
    im = ax.imshow(mat, cmap="viridis",aspect="auto")
    ax.set_xlabel("Time Step (Semester Duration)")
    ax.set_ylabel("Latent State Dimension")
    plt.colorbar(im, label="Selection Weight Magnitude")
    plt.title("Figure3 Input‑dependent selection weights of Mamba module")
    plt.tight_layout()
    plt.savefig("fig3_mamba_heatmap.png",dpi=300)
    plt.close()

def plot_figure4_tsne_latent():
    """Figure4 t‑SNE 隐空间可视化"""
    n_sample=360
    np.random.seed(42)
    emb = np.random.randn(n_sample,64)
    label_ids = np.hstack([np.zeros(120), np.ones(120), np.ones(120)*2])
    tsne = TSNE(n_components=2, random_state=42)
    proj = tsne.fit_transform(emb)
    fig,ax = plt.subplots(figsize=(10,8))
    colors = ["#4477bb","#ee8844","#55aa66"]
    names = ["Empathetic Perspective‑Taking","Problem‑Solving","Outlook"]
    for cid,name in enumerate(names):
        mask = label_ids == cid
        ax.scatter(proj[mask,0], proj[mask,1], label=name, color=colors[cid],alpha=0.7)
    ax.set_xlabel("t‑SNE Dimension 1")
    ax.set_ylabel("t‑SNE Dimension 2")
    plt.title("Figure4 t‑SNE projection of student latent representations")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig4_tsne.png",dpi=300)
    plt.close()

if __name__ == "__main__":
    plot_figure1_system_architecture()
    plot_figure2_ltc_time_constant()
    plot_figure3_mamba_selection_heatmap()
    plot_figure4_tsne_latent()
