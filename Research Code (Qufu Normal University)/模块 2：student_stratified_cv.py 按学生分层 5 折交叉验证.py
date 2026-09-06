# student_stratified_cv.py
import numpy as np
from sklearn.model_selection import GroupKFold
import pandas as pd

def get_student_stratified_5fold(df_meta: pd.DataFrame):
    """
    df_meta columns:
        sample_id, student_id, audio_path, score_empathy, score_outlook, score_problem
        bin_empathy, bin_outlook, bin_problem
    return: list of (train_idx, val_idx) for 5‑fold
    """
    group_kfold = GroupKFold(n_splits=5)
    groups = df_meta["student_id"].values  # group by student id
    fold_splits = []
    for train_idx, val_idx in group_kfold.split(X=df_meta, groups=groups):
        fold_splits.append((train_idx, val_idx))
    return fold_splits


if __name__ == "__main__":
    # demo usage
    mock_meta = pd.DataFrame({
        "student_id": np.repeat(np.arange(120), 30),  # 120学生，每人30条样本
        "sample_id": np.arange(120*30),
    })
    folds = get_student_stratified_5fold(mock_meta)
    for fold, (tr,va) in enumerate(folds):
        stu_tr = np.unique(mock_meta.iloc[tr]["student_id"])
        stu_va = np.unique(mock_meta.iloc[va]["student_id"])
        overlap = np.intersect1d(stu_tr, stu_va)
        print(f"fold {fold}: train students {len(stu_tr)}, val students {len(stu_va)}, overlap={len(overlap)}")
        # overlap必须等于0，代表学生完全隔离，无信息泄露
