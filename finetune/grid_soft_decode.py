"""Offline Top-K/path-count grid for differentiable decode feasibility."""
import argparse, os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.validate_soft_decode import load_from_dirs, sampled_mixture_decode
from finetune.dataset import QlibDataset

def main(a):
    os.environ.update({"KRONOS_VAL_DATA_PATHS": str(Path(a.val_data).resolve()),
        "KRONOS_LOOKBACK_WINDOW":"120", "KRONOS_PREDICT_WINDOW":"10",
        "KRONOS_USE_SIZE_PERCENTILE":"1", "KRONOS_NUM_SIZE_BUCKETS":"0",
        "KRONOS_METADATA_PATH":str(Path(a.metadata).resolve()),
        "KRONOS_VAL_SIGNAL_START":"2025-07-03", "KRONOS_VAL_SIGNAL_END":"2026-07-02"})
    device=torch.device(a.device); model,tok=load_from_dirs(Path(a.model_dir),Path(a.tokenizer_dir),device)
    ds=QlibDataset('val'); pos=torch.linspace(0,len(ds)-1,a.samples).long().tolist(); rec=[ds[i] for i in pos]
    x=torch.stack([r[0] for r in rec]).to(device); stamp=torch.stack([r[1] for r in rec]).to(device)
    sector=torch.stack([r[2] for r in rec]).to(device); pct=torch.stack([r[4] for r in rec]).to(device)
    with torch.no_grad():
        s1,s2=tok.encode(x,half=True); out=model(s1[:,:-1],s2[:,:-1],stamp[:,:-1],use_teacher_forcing=True,s1_targets=s1[:,1:],sector_id=sector,size_percentile=pct); l1,l2=out; l1,l2=l1[:,-10:],l2[:,-10:]
    p1=l1.float().softmax(-1); p2=l2.float().softmax(-1); conf=torch.cat((p1.amax(-1),p2.amax(-1)),dim=-1).mean(1)
    hard=tok.decode((l1.argmax(-1),l2.argmax(-1)),half=True).float()
    print('confidence groups: low<0.4, mid=0.4..0.8, high>0.8')
    print('top_k,samples,mean_abs_diff,p95_diff,coverage_mass,ms,low_diff,mid_diff,high_diff')
    for k in (4,8,16):
      mass=(p1.topk(min(k,p1.shape[-1]),-1).values.sum(-1).mean()+p2.topk(min(k,p2.shape[-1]),-1).values.sum(-1).mean())/2
      for n in (8,16,32):
        t=time.perf_counter(); mix=sampled_mixture_decode(tok,l1,l2,k,n); ms=(time.perf_counter()-t)*1000
        d=(mix-hard).abs(); vals=[]
        for lo,hi in ((-1,.4),(.4,.8),(.8,2)):
          z=d[ (conf>=lo)&(conf<hi) ]; vals.append(float(z.mean()) if z.numel() else float('nan'))
        print(f'{k},{n},{d.mean().item():.6f},{torch.quantile(d.flatten(),.95).item():.6f},{mass.item():.6f},{ms:.1f},'+','.join(f'{v:.6f}' for v in vals))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--model-dir',required=True); p.add_argument('--tokenizer-dir',required=True); p.add_argument('--val-data',required=True); p.add_argument('--metadata',required=True); p.add_argument('--device',default='cpu'); p.add_argument('--samples',type=int,default=32); main(p.parse_args())
