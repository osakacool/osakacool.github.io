import shap
import torch
import numpy as np
from model import TriModelEnsemble

model = TriModelEnsemble()
model.load_state_dict(torch.load("trimodel_ensemble.pt",map_location="cpu"))
model.eval()

#
n_sample=200
sf_test = torch.randn(n_sample,128)
md_test = torch.randn(n_sample,7,128)
mp_test = torch.randn(n_sample,10,128)
sport_idx_test = torch.zeros(n_sample,dtype=int)
w_test = torch.ones(n_sample)
load_test = torch.zeros(n_sample)

def model_predict_np(x_np):
    #包装shap需要numpy输入
    x = torch.from_numpy(x_np.astype(np.float32))
    with torch.no_grad():
        logits,_ = model(x, md_test, mp_test, sport_idx_test, w_test, load_test)
    return logits.numpy()

explainer = shap.KernelExplainer(model_predict_np, sf_test.numpy()[:20])
shap_values = explainer.shap_values(sf_test.numpy()[:100])

#绘制图10
shap.summary_plot(shap_values, sf_test.numpy()[:100], show=False)
plt.savefig("shap_summary_plot.png",dpi=300)
