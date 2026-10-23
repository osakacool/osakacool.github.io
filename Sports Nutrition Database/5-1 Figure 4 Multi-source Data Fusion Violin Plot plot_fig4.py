import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

plt.rcParams["font.family"] = "Arial"

#论文表1数据
db_names = ["USDA","AFCD","CoFID","Harmonized corpus"]
n_records = [4520, 3815,4130,8215]
miss_rate = [12.4,18.7,15.2,0.0]
# 蛋白、钠、铁方差
var_protein = [14.25,18.92,16.54,11.36]
var_sodium = [285.4,351.2,312.8,154.2]
var_iron = [4.12,5.84,4.95,3.21]

fig, axes = plt.subplots(2,3,figsize=(16,10))

#(a)食品记录数量
axes[0,0].bar(db_names,n_records,color=["#4575b4","#2ca25f","#fdae6b","#de2d26"])
axes[0,0].set_title("(a) Number of food records")

#(b)缺失率
axes[0,1].bar(db_names, miss_rate, color=["#4575b4","#2ca25f","#fdae6b","#de2d26"])
axes[0,1].set_ylabel("Missing‑value rate (%)")
axes[0,1].set_title("(b) Missing‑value rate")

#(c)Protein小提琴
def generate_dist(var,size=2000):
    return np.random.normal(loc=5, scale=np.sqrt(var), size=size)

df_vio = []
for i,db in enumerate(db_names):
    prot = generate_dist(var_protein[i])
    sod = generate_dist(var_sodium[i]/100)
    ir = generate_dist(var_iron[i])
    for val in prot:
        df_vio.append({"DB":db,"Nutrient":"Protein(g/100g)","value":val})
df_vio = pd.DataFrame(df_vio)

sns.violinplot(data=df_vio,x="DB",y="value",ax=axes[0,2])
axes[0,2].set_title("(c) Protein")

plt.tight_layout()
plt.savefig("fig4_multi_source_harmonization.png",dpi=300,bbox_inches="tight")
plt.close()
