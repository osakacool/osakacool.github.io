import torch
import numpy as np

class GradientAttributionAnalyzer:
    """
    对应论文 Eq. 13 & Figure 4：基于梯度的特征重要性分析。
    证明模型能够解释不同时间尺度下，哪些声学特征驱动了情感预测。
    """
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    def compute_attributions(self, dataloader, time_scale_masks):
        """
        time_scale_masks: dict, 包含 'rapid', 'intermediate', 'longterm' 的布尔掩码，
                          指示 128 个神经元中哪些属于哪个时间尺度（由模块7的GMM聚类得出）。
        """
        self.model.eval()
        # 初始化累加器
        total_attributions = {scale: np.zeros(128) for scale in time_scale_masks.keys()}
        total_samples = 0
        
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device).requires_grad_(True)
            self.model.zero_grad()
            
            outputs, _ = self.model(inputs)
            # 计算输出对输入的梯度 (Eq. 13)
            outputs.sum().backward()
            grads = inputs.grad.detach().cpu().numpy() # Shape: (batch, seq_len=300, 128)
            
            # 按时间尺度聚合梯度幅值
            for scale, mask in time_scale_masks.items():
                # mask shape: (128,) 布尔数组，筛选出属于该时间尺度的神经元
                scale_grads = grads[:, :, mask] 
                # 计算平均绝对梯度作为该尺度下各特征的重要性权重
                attr = np.mean(np.abs(scale_grads), axis=(0, 1)) 
                total_attributions[scale] += attr
            total_samples += inputs.size(0)
            
        # 归一化到 0-1 之间，便于绘制 Figure 4 的热力图
        for scale in total_attributions:
            total_attributions[scale] /= total_samples
            max_val = np.max(total_attributions[scale])
            if max_val > 0:
                total_attributions[scale] /= max_val
                
        return total_attributions