from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import numpy as np
import pandas as pd
import torch
from models import build_model
from mterd.io import config
from mterd.metrics import regression, ccc, tec, dtw_distance
from mterd.preprocessing import FeatureTransform
from mterd.splits import participant_splits
from mterd.training import Windows, predict_session
from mterd.evaluation import summarize_predictions, fold_statistics

torch.set_num_threads(1)

class PipelineBehaviorTests(unittest.TestCase):
    def test_regression_and_metric_identity(self):
        y=np.array([0.,1.,2.,1.]);p=y.copy()
        self.assertEqual(regression(y,p)['RMSE'],0)
        self.assertAlmostEqual(ccc(y,p),1)
        self.assertEqual(dtw_distance(y,p),0)
        self.assertEqual(tec(y,p),1)
        self.assertTrue(np.isnan(ccc(np.ones(4),np.ones(4))))
        with self.assertRaises(ValueError):regression(y,np.array([np.nan]*4))

    def test_participant_disjoint_and_test_once(self):
        cfg=config();m=pd.DataFrame({'participant_id':[f'p{i}' for i in range(10) for _ in range(2)]})
        splits=participant_splits(m,cfg);tests=[]
        for s in splits:
            tr,va,te=[set(s[k]) for k in ['train_participants','val_participants','test_participants']]
            self.assertFalse(tr&va or tr&te or va&te);tests.extend(s['test'])
        self.assertEqual(sorted(tests),list(range(len(m))))

    def test_pca_fit_excludes_unseen_distribution(self):
        train=np.arange(200,dtype=float).reshape(40,5)
        transform=FeatureTransform(components=3,max_frames=30).fit(lambda:iter([train]))
        before=transform.scaler.mean_.copy()
        transform.transform(train+100000)
        np.testing.assert_allclose(before,train.mean(axis=0))
        np.testing.assert_allclose(transform.scaler.mean_,before)

    def test_all_models_gradients_and_tau_bound(self):
        cfg=config();cfg.update(hidden_size=4,num_layers=2,attention_history=3,tcn_dilations=[1,2])
        for name in ['lstm_ltc','standard_lstm','attention_lstm','tcn']:
            model=build_model(name,3,cfg)
            x=torch.randn(2,12,3,requires_grad=True);y=model(x)
            self.assertEqual(tuple(y.shape),(2,));y.square().sum().backward()
            self.assertTrue(torch.isfinite(x.grad).all())
            if name=='lstm_ltc':
                model.eval();_,d=model(x,return_diagnostics=True)
                self.assertGreaterEqual(d['forget_tau'].min().item(),.1)
                self.assertLessEqual(d['forget_tau'].max().item(),10.)
                h=.1/(-torch.expm1(-.1/d['forget_tau']))
                self.assertLessEqual(h.max().item(),10.0501)
                self.assertIsNotNone(model.cells[0].tau_map.weight.grad)

    def test_windows_do_not_cross_sessions(self):
        sessions=[dict(X=np.full((8,2),value,dtype=np.float32),y=np.arange(8,dtype=float)) for value in [1,2]]
        ds=Windows(sessions,length=4,stride=1)
        self.assertEqual(len(ds),10)
        for x,y in ds:
            self.assertTrue(torch.all(x==x[0,0]));self.assertGreaterEqual(y.item(),3)

    def test_pooling_matches_trailing_seconds(self):
        class Endpoint(torch.nn.Module):
            def forward(self,x):return x[:,-1,0]
        cfg=config();cfg.update(sequence_length=3,batch_size=8)
        ts=np.arange(31)/10
        s=dict(X=ts[:,None].astype(np.float32),time=ts,y=ts.copy(),
               annotations=pd.DataFrame({'timestamp':[0.,1.,2.,3.],'regulation_score':[0.,1.,2.,3.]}),
               participant_id='p',session_id='s')
        frame,_=predict_session(Endpoint(),s,cfg,'standard_lstm',0,17,torch.device('cpu'))
        np.testing.assert_allclose(frame.timestamp,[2,3])
        np.testing.assert_allclose(frame.y_pred,[1.55,2.55],atol=1e-6)
        np.testing.assert_allclose(frame.y_true,[2,3])

    def test_statistics_aggregate_seeds_before_folds(self):
        rows=[]
        for model in ['lstm_ltc','standard_lstm']:
            for fold in range(5):
                for seed in [17,29,43]:
                    row=dict(model=model,fold=fold,seed=seed)
                    for k in ['RMSE','Pearson_r','CCC','TCC','MS_DTW','TEC']:
                        row[k]=.1*fold+(.02 if model=='standard_lstm' else 0)+seed/10000
                    rows.append(row)
        summary,comp,avg=fold_statistics(pd.DataFrame(rows))
        self.assertTrue((summary.n_folds==5).all());self.assertEqual(len(avg),10)
        self.assertTrue((comp.n_pairs==5).all())
        self.assertTrue((comp.p_holm.dropna()>=comp.p_raw.dropna()).all())

    def test_duplicate_predictions_rejected(self):
        row=dict(model='lstm_ltc',fold=0,seed=17,participant_id='p',session_id='s',timestamp=1.,y_true=.2,y_pred=.3)
        with self.assertRaises(ValueError):summarize_predictions(pd.DataFrame([row,row]),config())

if __name__=='__main__':unittest.main()
