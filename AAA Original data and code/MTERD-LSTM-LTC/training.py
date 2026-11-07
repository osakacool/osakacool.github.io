from pathlib import Path
import importlib.metadata
import json
import platform
import hashlib
import random
import time
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from .io import load_session, dump_json, sha256
from .preprocessing import FeatureTransform
from models import build_model

class Windows(Dataset):
    def __init__(self, sessions, length, stride=10):
        self.sessions, self.length = sessions, length
        self.indices = [(i,t) for i,s in enumerate(sessions)
                        for t in range(length-1,len(s['X']),stride) if np.isfinite(s['y'][t])]
        if not self.indices:
            raise ValueError('No full windows with observed targets')

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i,t = self.indices[idx]; s=self.sessions[i]
        return torch.from_numpy(np.array(s['X'][t-self.length+1:t+1],copy=True)), torch.tensor(s['y'][t],dtype=torch.float32)

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True, warn_only=True)

def get_device(cfg):
    device = cfg['device']
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if device=='auto' else device)

def load_partition(m, idx, cfg, psycho=False):
    return [load_session(m.iloc[i], cfg, psycho) for i in idx]

def neural_fit(train, val, model_name, cfg, seed, out):
    seed_all(seed); device=get_device(cfg)
    if device.type=='cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(model_name, train[0]['X'].shape[1],cfg).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    # Manuscript L2 applies to matrices; biases and normalization vectors are unpenalized.
    optimizer = torch.optim.Adam([{'params':[p for p in params if p.ndim>1],
                                    'weight_decay':cfg['weight_decay']},
                                  {'params':[p for p in params if p.ndim<=1],'weight_decay':0}],
                                  lr=cfg['learning_rate'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.1,
                                                         patience=cfg['lr_patience'])
    loader_kwargs = dict(batch_size=cfg['batch_size'], num_workers=cfg['num_workers'])
    train_loader = DataLoader(Windows(train,cfg['sequence_length'],cfg['train_stride_frames']),
                               shuffle=True,**loader_kwargs)
    val_loader = DataLoader(Windows(val,cfg['sequence_length'],cfg['train_stride_frames']),
                             shuffle=False,**loader_kwargs)
    best, stale, history = float('inf'),0,[]
    start=time.perf_counter()
    for epoch in range(cfg['max_epochs']):
        model.train(); loss_sum=0; count=0
        for x,y in train_loader:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=nn.functional.mse_loss(model(x),y)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(params,cfg['clip_grad_norm']); optimizer.step()
            loss_sum+=loss.item()*len(y);count+=len(y)
        model.eval(); vs=0; vn=0
        with torch.no_grad():
            for x,y in val_loader:
                x,y=x.to(device),y.to(device)
                vs+=nn.functional.mse_loss(model(x),y,reduction='sum').item();vn+=len(y)
        v=vs/vn
        if not np.isfinite(v):
            raise RuntimeError('Nonfinite validation loss')
        scheduler.step(v)
        history.append(dict(epoch=epoch+1,train_mse=loss_sum/count,val_mse=v,
                             learning_rate=optimizer.param_groups[0]['lr']))
        if v < best:
            best,stale=v,0
            # Store CPU tensors so artifacts can be inspected without a GPU.
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'model':model_name,'config':cfg,'input_dim':train[0]['X'].shape[1],
                        'seed':seed,'best_epoch':epoch+1},out/'checkpoint.pt')
        else:
            stale+=1
        print(f'{out.name} epoch {epoch+1}: val_mse={v:.6g}',flush=True)
        if stale>=cfg['early_stopping_patience']:
            break
    saved=torch.load(out/'checkpoint.pt',map_location=device,weights_only=True)
    model.load_state_dict(saved['state_dict']);model.eval()
    pd.DataFrame(history).to_csv(out/'training_history.csv',index=False)
    return model,dict(training_seconds=time.perf_counter()-start,best_epoch=saved['best_epoch'],
                       best_validation_mse=best,device=str(device),
                       peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type=='cuda' else None)

def window_descriptors(sessions, length, stride):
    x,y=[],[]
    for s in sessions:
        for t in range(length-1,len(s['X']),stride):
            if np.isfinite(s['y'][t]):
                w=s['X'][t-length+1:t+1]
                x.append(np.concatenate([w.mean(0),w.std(0)]));y.append(s['y'][t])
    if not x:
        raise ValueError('No ridge training windows')
    return np.array(x),np.array(y)

def ridge_fit(train,val,cfg,out):
    start=time.perf_counter()
    x,y=window_descriptors(train,cfg['sequence_length'],cfg['train_stride_frames'])
    vx,vy=window_descriptors(val,cfg['sequence_length'],cfg['train_stride_frames'])
    scaler=StandardScaler().fit(x);x=scaler.transform(x);vx=scaler.transform(vx)
    scores=[];best=None;best_loss=float('inf')
    for alpha in cfg['ridge_alphas']:
        m=Ridge(alpha=alpha).fit(x,y);loss=float(np.mean((m.predict(vx)-vy)**2))
        scores.append(dict(alpha=alpha,val_mse=loss))
        if loss<best_loss:
            best,best_loss=m,loss
    joblib.dump((scaler,best),out/'ridge.joblib')
    pd.DataFrame(scores).to_csv(out/'ridge_validation.csv',index=False)
    return (scaler,best),dict(training_seconds=time.perf_counter()-start,
                              best_validation_mse=best_loss,alpha=float(best.alpha),device='cpu')

def predict_session(model, s, cfg, model_name, fold, seed, device):
    length=cfg['sequence_length'];stride=cfg['prediction_stride_frames']
    ends=[t for t in range(length-1,len(s['X']),stride) if np.isfinite(s['y'][t])]
    if not ends:
        raise ValueError('Test session has no valid full windows')
    predictions=[]; diagnostics=[]
    for start in range(0,len(ends),cfg['batch_size']):
        idx=ends[start:start+cfg['batch_size']]
        windows=np.stack([s['X'][t-length+1:t+1] for t in idx])
        if model_name=='psychoacoustic':
            scaler,ridge=model
            z=np.concatenate([windows.mean(1),windows.std(1)],axis=1)
            prediction=ridge.predict(scaler.transform(z))
        else:
            tx=torch.from_numpy(windows).to(device)
            with torch.no_grad():
                if model_name=='lstm_ltc':
                    prediction,diag=model(tx,return_diagnostics=True)
                    tau=diag['forget_tau'].cpu().numpy()
                    f=diag['forget_retention'].cpu().numpy()
                    # Eq 11 is applied to per-window time-averaged tau, explicitly.
                    horizon=cfg['frame_step_seconds']/(-np.expm1(-cfg['frame_step_seconds']/tau))
                    retention_horizon=-cfg['frame_step_seconds']/np.log(np.clip(f,1e-12,1-1e-7))
                    diagnostics.extend([(horizon[i],retention_horizon[i]) for i in range(len(idx))])
                else:
                    prediction=model(tx)
                prediction=prediction.cpu().numpy()
        predictions.extend(prediction.tolist())
    raw=pd.DataFrame({'timestamp':s['time'][ends],'y_pred':predictions})
    raw['second']=np.ceil(raw.timestamp-1e-6).astype(int)
    grouped=raw.groupby('second').y_pred.agg(['mean','count'])
    # Full trailing one-second pools only; drop incomplete warm-up and tail bins.
    expected=int(round(1/(cfg['frame_step_seconds']*stride)))
    if not np.isclose(expected*cfg['frame_step_seconds']*stride,1):
        raise ValueError('Prediction hop must divide one second')
    grouped=grouped[grouped['count']==expected]
    a=s['annotations'];at=a.timestamp.to_numpy(float);ay=a[cfg['target']].to_numpy(float)
    grouped=grouped[(grouped.index>=at[0]) & (grouped.index<=at[-1])]
    if grouped.empty:
        raise ValueError('No complete 1-Hz test bins after window warm-up')
    frame=pd.DataFrame({'model':model_name,'fold':fold,'seed':seed,
                'participant_id':s['participant_id'],'session_id':s['session_id'],
                'timestamp':grouped.index.to_numpy(float),
                'y_true':np.interp(grouped.index,at,ay),'y_pred':grouped['mean'].to_numpy(),
                'pooled_frames':grouped['count'].to_numpy()})
    horizon_rows=[]
    if diagnostics:
        horizons=np.mean([d[0] for d in diagnostics],axis=0)
        retained=np.mean([d[1] for d in diagnostics],axis=0)
        for layer in range(horizons.shape[0]):
            for neuron in range(horizons.shape[1]):
                horizon_rows.append(dict(fold=fold,seed=seed,session_id=s['session_id'],
                   layer=layer,neuron=neuron,horizon_seconds=float(horizons[layer,neuron]),
                   retention_horizon_seconds=float(retained[layer,neuron]),n_windows=len(ends)))
    return frame,horizon_rows

def run_one(manifest, split, cfg, model_name, seed, output):
    out=Path(output)/f'{model_name}_fold{split["fold"]}_seed{seed}'
    if out.exists():
        raise FileExistsError(f'Run already exists: {out}. Use a new output path.')
    out.mkdir(parents=True)
    psycho=model_name=='psychoacoustic'
    if not psycho:
        validated=json.loads((Path(output)/'validation_report.json').read_text())
        key_payload=dict(split=split,features=validated['records'],
                 components=cfg['pca_components'],max_frames=cfg['pca_fit_max_frames'],
                 split_seed=cfg['split_seed'],target=cfg['target'],dt=cfg['frame_step_seconds'])
        key=hashlib.sha256(json.dumps(key_payload,sort_keys=True).encode()).hexdigest()[:16]
        cache=Path(output)/f'preprocessing_fold{split["fold"]}_{key}'
        cache.mkdir(parents=True,exist_ok=True)
        if not (cache/'complete.json').exists():
            transform=FeatureTransform(cfg['pca_components'],cfg['pca_fit_max_frames'],cfg['split_seed'])
            transform.fit(lambda: (load_session(manifest.iloc[i],cfg)['X'] for i in split['train']))
            joblib.dump(transform,cache/'preprocessor.joblib')
            for i,row in enumerate(manifest.itertuples(index=False)):
                session=load_session(row,cfg)
                np.save(cache/f'session_{i}.npy',transform.transform(session['X']))
                np.savez(cache/f'meta_{i}.npz',time=session['time'],y=session['y'],
                         feature_names=np.array(session['feature_names'],dtype=str))
            dump_json(cache/'complete.json',key_payload)
        transform=joblib.load(cache/'preprocessor.joblib')
        joblib.dump(transform,out/'preprocessor.joblib')
        dump_json(out/'pca_report.json',dict(components=cfg['pca_components'],
                    explained_variance=transform.explained_variance,fit_frames=transform.fit_frames,
                    reached_95_percent=transform.explained_variance>=.95,cache=str(cache)))
        def cached_partition(indices):
            sessions=[]
            for i in indices:
                row=manifest.iloc[i]
                with np.load(cache/f'meta_{i}.npz',allow_pickle=False) as f:
                    sessions.append(dict(X=np.load(cache/f'session_{i}.npy',mmap_mode='r'),
                         time=f['time'],y=f['y'],feature_names=f['feature_names'].astype(str).tolist(),
                         annotations=pd.read_csv(row.annotations_path),
                         session_id=row.session_id,participant_id=row.participant_id))
            return sessions
        train,val,test=[cached_partition(split[k]) for k in ['train','val','test']]
        model,report=neural_fit(train,val,model_name,cfg,seed,out)
    else:
        train=load_partition(manifest,split['train'],cfg,True)
        val=load_partition(manifest,split['val'],cfg,True)
        test=load_partition(manifest,split['test'],cfg,True)
        model,report=ridge_fit(train,val,cfg,out)
    predictions=[]; horizons=[];prediction_start=time.perf_counter()
    for s in test:
        frame,h=predict_session(model,s,cfg,model_name,split['fold'],seed,get_device(cfg))
        predictions.append(frame);horizons.extend(h)
    pd.concat(predictions,ignore_index=True).to_csv(out/'predictions.csv',index=False)
    if horizons:
        pd.DataFrame(horizons).to_csv(out/'memory_horizons.csv',index=False)
    if model_name=='lstm_ltc':
        from .attribution import export_attribution
        export_attribution(model,test[0],transform,cfg,out,get_device(cfg))
    versions={}
    for package in ['torch','numpy','pandas','scipy','scikit-learn','opensmile']:
        try: versions[package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: versions[package]=None
    report.update(model=model_name,fold=split['fold'],seed=seed,config=cfg,split=split,
                  prediction_seconds=time.perf_counter()-prediction_start,
                  target=cfg['target'],python=platform.python_version(),packages=versions,
                  cuda_version=torch.version.cuda,
                  gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  input_files=json.loads((Path(output)/'validation_report.json').read_text())['records'])
    dump_json(out/'run.json',report)
    return out
