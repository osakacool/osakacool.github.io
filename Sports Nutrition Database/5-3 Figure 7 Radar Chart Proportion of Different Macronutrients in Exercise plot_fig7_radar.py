import matplotlib.pyplot as plt
import numpy as np

labels = ["Carbohydrate (%)","Protein (%)","Fat (%)"]
angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

#论文表5数据
endurance = [58.4, 21.2, 20.4]
strength = [42.5,33.8,23.7]
teamsport = [51.2, 24.5,24.3]

endurance += endurance[:1]
strength += strength[:1]
teamsport += teamsport[:1]

fig, (ax1,ax2,ax3)=plt.subplots(1,3,subplot_kw=dict(projection="polar"),figsize=(15,6))

ax1.plot(angles,endurance,"b",label="Median")
ax1.fill(angles,endurance,alpha=0.25,color="b")
ax1.set_xticks(angles[:-1])
ax1.set_xticklabels(labels)
ax1.set_title("(a) Endurance Sports")

ax2.plot(angles,strength,"g")
ax2.fill(angles,strength,alpha=0.25,color="g")
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(labels)
ax2.set_title("(b) Strength Sports")

ax3.plot(angles,teamsport,"#ff8800")
ax3.fill(angles,teamsport,alpha=0.25,color="#ff8800")
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(labels)
ax3.set_title("(c) Team Sports")

plt.tight_layout()
plt.savefig("fig7_radar_macro.png",dpi=300)
plt.close()
