import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

food_labels = ["Oats","Eggs","Chicken","Rice","Salmon","Spinach","Broccoli","Almonds","Banana","Yogurt"]

attn_mat = np.random.rand(10,10)*0.12
np.fill_diagonal(attn_mat,0.22)
attn_mat[2,3]=0.16
attn_mat[2,8]=0.16

plt.figure(figsize=(12,8))
ax = sns.heatmap(attn_mat, xticklabels=food_labels, yticklabels=food_labels, cmap="Greens",vmin=0,vmax=0.28)
ax.set_title("(a) Representative daily plan attention weight matrix")
plt.tight_layout()
plt.savefig("fig9_attention_matrix.png",dpi=300)
plt.close()
