from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS=['#667c8a','#b68e7e','#97a78e','#a79bb2','#caad7c']

def save(fig,folder,name):
    fig.tight_layout()
    fig.savefig(folder/f'{name}.png',dpi=300,bbox_inches='tight')
    fig.savefig(folder/f'{name}.pdf',bbox_inches='tight')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description='Plot computed outputs; no generated empirical distributions')
    p.add_argument('--metrics',default='results/metrics')
    p.add_argument('--runs',required=True);p.add_argument('--out',default='results/figures')
    p.add_argument('--session',default=None)
    a=p.parse_args();metrics=Path(a.metrics);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    summary=pd.read_csv(metrics/'summary.csv')
    folded=pd.read_csv(metrics/'per_fold.csv');models=sorted(summary.model.unique())
    for name,names in [('predictive_performance',['RMSE','Pearson_r','CCC']),
                       ('temporal_metrics',['TCC','MS_DTW','TEC'])]:
        fig,axes=plt.subplots(1,3,figsize=(12,3.5))
        for ax,metric in zip(axes,names):
            for j,model in enumerate(models):
                r=summary[(summary.model==model)&(summary.metric==metric)].iloc[0]
                values=folded.loc[folded.model==model,metric].to_numpy()
                offsets=np.linspace(-.08,.08,len(values))
                ax.scatter(j+offsets,values,color=COLORS[j%len(COLORS)],s=20,alpha=.8)
                ax.errorbar(j,r['mean'],yerr=r['sd'],fmt='o',color='black',capsize=3)
            ax.set_xticks(range(len(models)),models,rotation=40,ha='right');ax.set_title(metric)
        save(fig,out,name)
    paths=sorted(Path(a.runs).glob('*/predictions.csv'))
    pred=pd.concat([pd.read_csv(p,dtype={'session_id':str,'participant_id':str}) for p in paths])
    session=a.session or sorted(pred.session_id.unique())[0]
    selected=pred[pred.session_id==session]
    if selected.empty:raise ValueError(f'Session not found: {session}')
    fig,ax=plt.subplots(figsize=(10,3.5))
    first=selected.sort_values('timestamp').drop_duplicates('timestamp')
    ax.plot(first.timestamp/60,first.y_true,color='black',lw=1.6,label='Observed target')
    for j,(model,g) in enumerate(selected.groupby('model')):
        avg=g.groupby('timestamp').y_pred.mean()
        ax.plot(avg.index/60,avg.values,label=model,color=COLORS[j%len(COLORS)],lw=1)
    ax.set(xlabel='Time (minutes)',ylabel='Target in original units',title=f'Held-out session {session}')
    ax.legend(fontsize=8,ncol=3);save(fig,out,'session_trajectory')
    gdir=metrics/'gmm'
    if (gdir/'bic.csv').exists():
        bic=pd.read_csv(gdir/'bic.csv');fig,ax=plt.subplots(figsize=(5,3.5))
        for _,g in bic.groupby(['fold','seed']):
            ax.plot(g.k,g.bic,alpha=.5,lw=1)
        ax.set(xlabel='GMM components K',ylabel='BIC');save(fig,out,'gmm_bic')
        h=pd.read_csv(gdir/'neuron_horizons.csv')
        column='horizon_seconds' if 'horizon_seconds' in h else 'retention_horizon_seconds'
        fig,ax=plt.subplots(figsize=(5,3.5));ax.hist(h[column],bins=35,color=COLORS[0],edgecolor='white')
        ax.set(xlabel='Memory horizon (seconds)',ylabel='Neuron observations across runs')
        save(fig,out,'memory_horizons')
    attr_paths=sorted(Path(a.runs).glob('*/attributions.csv'))
    if attr_paths:
        attr=pd.concat([pd.read_csv(p) for p in attr_paths])
        top=attr.groupby('feature').attribution.mean().nlargest(12).index
        matrix=attr[attr.feature.isin(top)].pivot_table(index='feature',columns='lag_scale',
                                                       values='attribution',aggfunc='mean')
        fig,ax=plt.subplots(figsize=(6,5));im=ax.imshow(matrix.values,aspect='auto',cmap='cividis')
        ax.set_yticks(range(len(matrix)),matrix.index,fontsize=7)
        ax.set_xticks(range(len(matrix.columns)),matrix.columns,rotation=30,ha='right')
        fig.colorbar(im,ax=ax,label='Mean squared input gradient');save(fig,out,'feature_attributions')
    print(out)

if __name__=='__main__':main()
