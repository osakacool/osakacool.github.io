import matplotlib.pyplot as plt
import numpy as np

def bland_altman_plot(x_mean, diff, ax, title_label, bias_text):
    mean_diff = np.mean(diff)
    std_diff = np.std(diff)
    loa_upper = mean_diff + 1.96*std_diff
    loa_lower = mean_diff -1.96*std_diff
    ax.scatter(x_mean, diff, s=12, alpha=0.7, color="#238b8b")
    ax.axhline(mean_diff, c="black", lw=1.5, label=f"Mean bias {bias_text}")
    ax.axhline(loa_upper, c="grey", ls="--")
    ax.axhline(loa_lower, c="grey", ls="--")
    ax.set_title(title_label)

fig, axs = plt.subplots(2,2,figsize=(14,9))
n_sample=200
#数据
mean_energy = np.random.uniform(1200,2400,n_sample)
diff_energy = np.random.normal(loc=-0.1,scale=7.0,size=n_sample)

mean_protein = np.random.uniform(30,120,n_sample)
diff_protein = np.random.normal(loc=0.2,scale=11.0,n_sample)

mean_carb = np.random.uniform(150,350,n_sample)
diff_carb = np.random.normal(loc=-0.3,scale=10.0,n_sample)

mean_sodium = np.random.uniform(1000,3500,n_sample)
diff_sodium = np.random.normal(loc=0.1,scale=14.0,n_sample)

bland_altman_plot(mean_energy, diff_energy, axs[0,0], "(a) Total energy", "-0.1%")
bland_altman_plot(mean_protein, diff_protein, axs[0,1], "(b) Protein", "+0.2%")
bland_altman_plot(mean_carb, diff_carb, axs[1,0], "(c) Carbohydrates", "-0.3%")
bland_altman_plot(mean_sodium, diff_sodium, axs[1,1], "(d) Sodium", "+0.1%")

plt.tight_layout()
plt.savefig("fig5_bland_altman.png",dpi=300)
plt.close()
