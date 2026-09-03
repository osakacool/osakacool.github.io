import numpy as np
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

class ParticipantWiseCV:
    """
    对应论文 5.3 节：5-fold participant-wise cross-validation。
    确保训练集和测试集的被试完全独立。
    """
    def __init__(self, participant_ids, n_splits=5):
        """
        participant_ids: List/Array, 每个 session 对应的被试 ID (长度等于 session 数量)
        """
        self.participant_ids = np.array(participant_ids)
        self.n_splits = n_splits
        self.gkf = GroupKFold(n_splits=n_splits)

    def get_fold_indices(self):
        """
        生成 5 折的索引划分
        返回: List of tuples (train_session_indices, test_session_indices)
        """
        # 使用伪标签进行 GroupKFold 划分
        dummy_y = np.zeros(len(self.participant_ids))
        folds = list(self.gkf.split(X=np.arange(len(self.participant_ids)), 
                                    y=dummy_y, 
                                    groups=self.participant_ids))
        return folds

    def create_dataloaders(self, full_dataset, session_to_participant_map, batch_size=32):
        """
        根据 Dataset 和 划分索引，生成 PyTorch DataLoader
        """
        folds = self.get_fold_indices()
        dataloaders_dict = {}
        
        for fold_idx, (train_sess_idx, test_sess_idx) in enumerate(folds):
            # 注意：这里需要将 session 级别的索引映射回 Dataset 中的 sequence 级别索引
            # 为简化，假设 full_dataset 已经按 session 顺序排列，此处需编写映射逻辑
            train_indices = self._map_session_to_sequence_indices(train_sess_idx, full_dataset)
            test_indices = self._map_session_to_sequence_indices(test_sess_idx, full_dataset)
            
            train_subset = Subset(full_dataset, train_indices)
            test_subset = Subset(full_dataset, test_indices)
            
            dataloaders_dict[fold_idx] = {
                'train': DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4),
                'test': DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)
            }
            
        return dataloaders_dict

    def _map_session_to_sequence_indices(self, session_indices, dataset):
        """辅助函数：将 session 索引转换为 300帧滑动窗口的 sequence 索引"""
        # 实际实现需遍历 dataset.processed_sessions 计算累积的 sequence 偏移量
        pass 