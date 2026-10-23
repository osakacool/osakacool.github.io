# -*- coding:utf-8 -*-
"""
推理demo inference_demo.py
加载训练好的 tri‑model_ensemble.pt
输入运动员参数：sport_type_idx, weight_kg, training_load
输出生成膳食方案、总营养统计
"""
import torch
import numpy as np
from model import TriModelEnsemble

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sport_name_map = {
    0:"耐力运动 Endurance",
    1:"力量运动 Strength",
    2:"团队运动 Team‑Sport",
    3:"短跑",
    4:"体操",
    5:"游泳",
    6:"自行车",
    7:"球类综合"
}


def generate_sport_diet_plan(model, sport_idx:int, weight_kg:float, load_level:float):
    """
    sport_idx:0‑7 运动类别
    weight_kg:运动员体重 50‑110kg
    load_level:0=低, 1=中等,2=高训练负荷
    return:top_food_idx_list, latent_z
    """
    model.eval()
    with torch.no_grad():
        #构造输入batch size=1
        sport_t = torch.tensor([sport_idx], dtype=torch.long).to(device)
        weight_t = torch.tensor([weight_kg], dtype=torch.float32).to(device)
        load_t = torch.tensor([load_level], dtype=torch.float32).to(device)

        #食品序列特征（真实项目替换融合数据集特征）
        single_food = torch.randn(1, 128).to(device)
        multi_day_seq = torch.randn(1,7,128).to(device)
        meal_seq = torch.randn(1,10,128).to(device)

        logits, z_u = model(single_food, multi_day_seq, meal_seq, sport_t, weight_t, load_t)
        prob = torch.softmax(logits, dim=-1)
        topk_val, topk_idx = torch.topk(prob, k=12, dim=-1)

    top_foods = topk_idx[0].cpu().numpy().tolist()
    top_prob = topk_val[0].cpu().numpy().tolist()
    return top_foods, top_prob, z_u.cpu().numpy()


if __name__ == "__main__":
    VOCAB_SIZE = 8215
    model = TriModelEnsemble(nut_dim=128, vocab_size=VOCAB_SIZE).to(device)
    # 加载训练权重
    model.load_state_dict(torch.load("./output_models/tri_model_ensemble.pt", map_location=device))

    # ========== 这里输入运动员参数 ==========
    input_sport_idx = 1      #1力量运动员
    input_weight = 85.2      #体重kg
    input_load = 2.0         #高训练负荷
    # ==============================================

    food_ids, food_probs, latent = generate_sport_diet_plan(model, input_sport_idx, input_weight, input_load)

    print("="*60)
    print(f"运动员信息：{sport_name_map[input_sport_idx]} | 体重：{input_weight} kg | 训练负荷：{input_load}")
    print("生成膳食食品ID列表(top‑12):")
    for fid, p in zip(food_ids, food_probs):
        print(f"  Food‑ID:{fid:5d}  生成概率:{p:.4f}")
    print("="*60)
    print("Latent表征shape：", latent.shape)
