import numpy as np
from sklearn.metrics import f1_score

def nutritional_adequacy(generated_nut:np.ndarray, ref_nut:np.ndarray, tol_ratio=0.10):
    """NA营养充足率：生成营养在参考±10%内占比
    generated_nut [N,N_nut]; ref_nut [N,N_nut]
    """
    lower = ref_nut*(1-tol_ratio)
    upper = ref_nut*(1+tol_ratio)
    ok_mask = np.logical_and(generated_nut >= lower, generated_nut <= upper)
    return float(np.mean(ok_mask))*100

def constraint_satisfaction_rate(generated_nut:np.ndarray, constraint_upper:dict):
    """CSR约束满足率，constraint_upper {"sodium":2300,...}"""
    n = generated_nut.shape[0]
    ok_count=0
    for i in range(n):
        valid=True
        for idx,limit in constraint_upper.items():
            if generated_nut[i,idx] > limit:
                valid=False
        if valid:
            ok_count+=1
    return ok_count/n*100

def plan_diversity(food_sets_list):
    """PD膳食多样性：样本间Jaccard距离均值
    food_sets_list list[set()]
    """
    from scipy.spatial.distance import jaccard
    dists=[]
    n=len(food_sets_list)
    for i in range(n):
        for j in range(i+1,n):
            s1,s2 = food_sets_list[i],food_sets_list[j]
            dists.append(jaccard(np.array(list(s1)),np.array(list(s2))))
    return float(np.mean(dists))

def generation_fidelity_f1(pred_food_cat, ref_food_cat):
    """GF生成保真度：macro F1"""
    return f1_score(ref_food_cat, pred_food_cat, average="macro")
