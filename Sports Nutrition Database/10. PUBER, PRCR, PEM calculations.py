# -*- coding:utf-8 -*-
"""
论文公式(6)(7)(8)(9)
μs=(Es × fs) / (4 × Ws)
PUBERs =100 ×[1−Φ((Us−μs) / σs)]
PRCRs =100 ×[Φ((Us−μs) / σs) − Φ((Ls−μs) / σs)]
PEMs =100 ×[μs+σs φ(zs)/(1−Φ(zs)) − Us]/Us , zs=(Us−μs)/σs
"""
import numpy as np
from scipy.stats import norm


def calc_mu_s(E_s: float, f_s: float, W_s: float) -> float:
    """公式6：体重标准化蛋白质摄入 μs  g/kg/day
    E_s:平均每日总能量 kcal
    f_s：蛋白质占能量比例 0~1
    W_s：平均体重 kg
    1g蛋白=4 kcal
    """
    mu = (E_s * f_s) / (4.0 * W_s)
    return mu


def calc_puber_prcr_pem(mu_s: float, sigma_s: float, Ls: float, Us: float):
    """
    :param mu_s: μs 均值 g/kg/day
    :param sigma_s: σs标准差
    :param Ls:参考区间下限
    :param Us:参考区间上限
    :return PUBER, PRCR, PEM (百分比)
    """
    zs = (Us - mu_s) / sigma_s
    phi_zs = norm.pdf(zs)
    cdf_zs = norm.cdf(zs)
    cdf_ls = norm.cdf((Ls - mu_s)/sigma_s)

    puber = 100.0 * (1.0 - cdf_zs)
    prcr = 100.0 * (cdf_zs - cdf_ls)

    # PEM 条件期望相对超标幅度
    if (1.0 - cdf_zs) < 1e-8:
        pem = 0.0
    else:
        cond_exceed = mu_s + sigma_s * (phi_zs / (1.0 - cdf_zs))
        pem = 100.0 * (cond_exceed - Us) / Us
    return round(puber,2), round(prcr,2), round(pem,2)


if __name__ == "__main__":
    # 论文表5的3种运动场景
    sport_scenarios = [
        {
            "name":"Endurance 耐力运动",
            "Es":3450,      #kcal
            "fs":0.212,     #蛋白能量占比
            "Ws":68.5,      #kg
            "sigma_s":0.2664,
            "Ls":1.4, "Us":2.0
        },
        {
            "name":"Strength 力量运动",
            "Es":3120,
            "fs":0.338,
            "Ws":85.2,
            "sigma_s":0.3255,
            "Ls":1.6, "Us":2.2
        },
        {
            "name":"Team Sports 团队运动",
            "Es":3380,
            "fs":0.245,
            "Ws":76.4,
            "sigma_s":0.3099,
            "Ls":1.4, "Us":2.0
        }
    ]

    print(f"{'运动类型':<18}{'μ(g/kg/d)':<12}{'PUBER(%)':<12}{'PRCR(%)':<12}{'PEM(%)':<12}")
    for s in sport_scenarios:
        mu = calc_mu_s(s["Es"], s["fs"], s["Ws"])
        puber,prcr,pem = calc_puber_prcr_pem(mu, s["sigma_s"], s["Ls"], s["Us"])
        print(f"{s['name']:<18}{mu:<12.3f}{puber:<12.2f}{prcr:<12.2f}{pem:<12.2f}")
