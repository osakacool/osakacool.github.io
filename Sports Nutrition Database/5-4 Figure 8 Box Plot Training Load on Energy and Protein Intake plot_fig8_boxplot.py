import matplotlib.pyplot as plt
import numpy as np

load_levels = ["Low","Moderate","High"]
#仿真
energy_data = [
    np.random.normal(loc=2450,scale=300,size=120),
    np.random.normal(loc=3100,scale=350,size=120),
    np.random.normal(loc=4100,scale=400,size=120),
]
protein_kg_data = [
    np.random.normal(loc=1.42,scale=0.2,size=120),
    np.random.normal(loc=1.75,scale=0.22,size=120),
    np.random.normal(loc=2.05,scale=0.24,size=120),
]

fig,(ax_left,ax_right)=plt.subplots(1,2,figsize=(14,7))
ax_left.boxplot(energy_data,labels=load_levels)
ax_left.set_title("Total Energy Intake")
ax_left.set_ylabel("Daily energy intake (kcal)")

ax_right.boxplot(protein_kg_data,labels=load_levels)
ax_right.set_title("Protein Intake")
ax_right.set_ylabel("Protein intake (g/Kg body weight)")

plt.tight_layout()
plt.savefig("fig8_training_load_response.png",dpi=300)
plt.close()
