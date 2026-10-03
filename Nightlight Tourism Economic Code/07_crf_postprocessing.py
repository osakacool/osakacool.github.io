import numpy as np
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian

class CRFPostProcessor:
    """
    Conditional Random Field for refining semantic segmentation boundaries.
    Corresponds to Section 5.2: Reduces Dining/Cultural misclassification by 23%.
    """
    def __init__(self, img_shape, num_classes=5):
        self.H, self.W, self.C = img_shape
        self.num_classes = num_classes
        # CRF Hyperparameters (Tuned for 500m resolution nocturnal data)
        self.iterations = 10       # 推理迭代次数
        self.pos_w = 3.0           # 空间位置势能权重
        self.pos_xy_std = 3.0      # 空间位置标准差
        self.bi_w = 10.0           # 双边势能权重 (结合辐射和空间)
        self.bi_xy_std = 50.0      # 双边空间标准差
        self.bi_rgb_std = 13.0     # 双边辐射标准差 (针对VIIRS radiance)

    def process(self, image_tensor, softmax_probs):
        """
        Args:
            image_tensor: (H, W, C) 原始5通道输入特征 (用于双边势能)
            softmax_probs: (H, W, num_classes) DeepLabV3+ 输出的概率图
        Returns:
            refined_map: (H, W) 优化后的类别标签图
        """
        d = dcrf.DenseCRF2D(self.W, self.H, self.num_classes)
        
        # 1. 一元势能 (Unary Potential): 来自深度学习的Softmax输出
        # pydensecrf 需要 (num_classes, H*W) 的格式
        U = unary_from_softmax(softmax_probs.transpose(2, 0, 1))
        d.setUnaryEnergy(U)
        
        # 2. 空间 pairwise 势能 (Spatial Smoothness)
        # 强制相邻像素具有相似的标签
        feats = create_pairwise_gaussian(sdims=(self.pos_xy_std, self.pos_xy_std), shape=(self.H, self.W))
        d.addPairwiseEnergy(feats, compat=self.pos_w)
        
        # 3. 双边势能 (Bilateral Potential): 结合空间与辐射特征
        # 如果辐射差异大（如从Core Zone到Non-Tourism Zone），则允许标签突变
        # 提取前3个通道 (Radiance, POI, Mobility_T1) 作为双边特征
        bilateral_feats = create_pairwise_bilateral(
            sdims=(self.bi_xy_std, self.bi_xy_std), 
            schan=(self.bi_rgb_std,), 
            img=image_tensor[:, :, :3], 
            chdim=2
        )
        d.addPairwiseEnergy(bilateral_feats, compat=self.bi_w)
        
        # 4. 推理 (Inference)
        Q = d.inference(self.iterations)
        refined_map = np.argmax(Q, axis=0).reshape((self.H, self.W))
        
        return refined_map

def calculate_crf_improvement(cm_before, cm_after):
    """计算CRF带来的错分率下降 (Table 6 验证)"""
    # 提取 Dining (1) 和 Cultural (2) 之间的混淆像素
    error_before = cm_before[1, 2] + cm_before[2, 1]
    error_after = cm_after[1, 2] + cm_after[2, 1]
    reduction_rate = (error_before - error_after) / error_before * 100
    print(f"CRF reduced Dining/Cultural misclassification by {reduction_rate:.1f}%")
    return reduction_rate