import matplotlib.pyplot as plt
import numpy as np

nutrient_names = [
    "Total carbohydrate","Leucine","Sodium","Total protein","Iron",
    "Calcium","Dietary fiber","Omega‑3 fatty acids","Vitamin D","Potassium"
]
shap_values = [0.452,0.385,0.361,0.298,0.274,0.245,0.212,0.198,0.175,0.162]

plt.figure(figsize=(12,7))
y_pos = np.arange(len(nutrient_names))
plt.barh(y_pos, shap_values, color="#238b45")
plt.yticks(y_pos, nutrient_names)
plt.xlabel("Mean absolute SHAP value")
plt.title("(a) Global importance by nutrient")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("fig10_shap_summary.png",dpi=300)
plt.close()
