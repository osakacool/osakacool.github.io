import numpy as np
import pandas as pd
import torch


def export_attribution(model, session, transform, cfg, out, device):
    """Eq 19 squared gradients, mapped through the fitted affine scaler/PCA.

    Groups refer to input lags, not inferred physiological pathways or neurons.
    Only held-out windows are used. No claimed feature/band grouping is invented.
    """
    length=cfg['sequence_length'];s=session
    ends=np.arange(length-1,len(s['X']),max(1,length))
    ends=ends[np.isfinite(s['y'][ends])][:cfg.get('attribution_windows',4)]
    if not len(ends):return
    lag=np.arange(length-1,-1,-1)*cfg['frame_step_seconds']
    scales={'rapid_0_to_1s':lag<=1,'intermediate_1_to_10s':(lag>1)&(lag<=10),
            'long_over_10s':lag>10}
    projection=transform.pca.components_/transform.scaler.scale_[None,:]
    values={k:np.zeros(projection.shape[1]) for k,mask in scales.items() if mask.any()}
    model.eval()
    for end in ends:
        x=torch.from_numpy(np.array(s['X'][end-length+1:end+1][None],copy=True)).to(device).requires_grad_(True)
        gradient=torch.autograd.grad(model(x).sum(),x)[0].detach().cpu().numpy()[0]
        original=gradient@projection
        for key,mask in scales.items():
            if mask.any():values[key]+=(original[mask]**2).mean(axis=0)/len(ends)
    rows=[dict(feature=name,lag_scale=key,attribution=float(score),n_windows=len(ends),
               session_id=s['session_id']) for key,val in values.items()
          for name,score in zip(s['feature_names'],val)]
    pd.DataFrame(rows).to_csv(out/'attributions.csv',index=False)
