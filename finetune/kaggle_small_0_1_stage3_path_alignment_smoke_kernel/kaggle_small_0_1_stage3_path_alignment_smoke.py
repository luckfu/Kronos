import json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path('/kaggle/working/Kronos')
SWANLAB_KEY = os.environ.get('SWANLAB_API_KEY', 'fmEPDGk4IItxgqSZKGL8i')
os.environ['SWANLAB_API_KEY'] = SWANLAB_KEY
subprocess.run(['git', 'clone', '--depth', '1', 'https://github.com/luckfu/Kronos.git', str(ROOT)], check=True)
sys.path.insert(0, str(ROOT))
os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')

# The smoke entrypoint is embedded so this kernel is reproducible even when
# the public repository HEAD predates the local experimental Stage-3 files.
trainer = ROOT / 'finetune' / 'train_stage3_path_alignment.py'
trainer.parent.mkdir(parents=True, exist_ok=True)
trainer.write_text(r'''import argparse, json, time
from pathlib import Path
import torch
try:
 import swanlab
except Exception:
 swanlab = None
from torch.utils.data import DataLoader
from model.kronos import Kronos, KronosTokenizer
from finetune.dataset import QlibDataset
from finetune.stage3_path_alignment import PathAlignmentConfig, DetachedLossEMA, compute_path_alignment_loss

def main(a):
    torch.manual_seed(a.seed); device=torch.device(a.device)
    print('initialization=stage3_from_c2_best_model_only', flush=True)
    print('parent_model=' + str(a.model_dir), flush=True)
    print('optimizer_state=reset', flush=True)
    print('global_step=0', flush=True)
    print('segment=0', flush=True)
    print('validation_mode=full', flush=True)
    print('target_slice=x[:,120:130]', flush=True)
    model=Kronos.from_pretrained(a.model_dir).to(device)
    tok=KronosTokenizer.from_pretrained(a.tokenizer_dir).to(device).eval()
    for p in tok.parameters(): p.requires_grad_(False)
    run = None
    if swanlab is not None:
        run = swanlab.init(project='Kronos', experiment_name='small_0.1_stage3_path_alignment_from_c2_best', config={'lr':a.lr,'batch':a.batch,'segments':a.segments,'top_k':16,'candidates':16,'parent':'c2_best','validation':'full'}, mode='cloud')
        run_url = getattr(run, 'url', getattr(run, 'web_url', ''))
        print('SWANLAB_RUN_URL=' + str(run_url), flush=True)
        if not run_url:
            raise RuntimeError('SwanLab init returned no run URL; refusing to start training')
    model.train(); ds=QlibDataset('train'); loader=DataLoader(ds,batch_size=a.batch,shuffle=False,num_workers=0)
    val_ds=QlibDataset('val'); val_loader=DataLoader(val_ds,batch_size=a.batch,shuffle=False,num_workers=0)
    print('validation_samples=' + str(len(val_ds)), flush=True)
    if len(val_ds) < 100000:
        raise RuntimeError(f'Validation set unexpectedly small: {len(val_ds)}; full validation is required')
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01)
    ema=DetachedLossEMA(0.99); cfg=PathAlignmentConfig(16,16,0.05,0.02,0.99)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); step=0; summary={'segments':[]}
    best_val=float('inf')
    for seg in range(1,a.segments+1):
        ds.set_epoch_seed(seg-1); seen=0; started=time.time()
        for batch in loader:
            x,stamp=batch[0].to(device),batch[1].to(device)
            sec=batch[2].to(device) if len(batch)>2 else None; pct=batch[4].to(device) if len(batch)>4 else None
            with torch.no_grad(): s1,s2=tok.encode(x,half=True)
            logits=model(s1[:,:-1],s2[:,:-1],stamp[:,:-1],use_teacher_forcing=True,s1_targets=s1[:,1:],sector_id=sec,size_percentile=pct)
            # Match C2's forecast objective: the ten prediction positions only.
            ce=model.head.compute_loss(logits[0][:,-10:],logits[1][:,-10:],s1[:,1:][:,-10:],s2[:,1:][:,-10:])[0]
            pa,metrics=compute_path_alignment_loss(tok,logits[0][:,-10:],logits[1][:,-10:],x[:,120:130],cfg,ema)
            loss=ce+pa; opt.zero_grad(set_to_none=True); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1; seen+=len(x)
            if seen>=20000: break
        model.eval(); val_total=0.0; val_n=0
        with torch.no_grad():
            for vb in val_loader:
                vx,vs=vb[0].to(device),vb[1].to(device); vsec=vb[2].to(device) if len(vb)>2 else None; vpct=vb[4].to(device) if len(vb)>4 else None
                vs1,vs2=tok.encode(vx,half=True); vlogits=model(vs1[:,:-1],vs2[:,:-1],vs[:,:-1],use_teacher_forcing=True,s1_targets=vs1[:,1:],sector_id=vsec,size_percentile=vpct)
                vce=model.head.compute_loss(vlogits[0][:,-10:],vlogits[1][:,-10:],vs1[:,1:][:,-10:],vs2[:,1:][:,-10:])[0]
                vpa,_=compute_path_alignment_loss(tok,vlogits[0][:,-10:],vlogits[1][:,-10:],vx[:,120:130],cfg,None)
                val_total += float(vce + vpa)*len(vx); val_n += len(vx)
        model.train(); val_loss=val_total/max(1,val_n)
        row={'segment':seg,'step':step,'samples':seen,'token_forecast_loss':float(ce.detach()),'validation_objective':val_loss,'path_align_loss':float(metrics['path_align_loss']),'horizon_mae':metrics['path_align_horizon_mae'].tolist(),'max_residual':float(metrics['path_align_max_residual']),'grad_norm':float(grad),'elapsed_sec':time.time()-started}
        if val_loss < best_val:
            best_val=val_loss; model.save_pretrained(out/'best_model')
        summary['segments'].append(row); torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'segment':seg,'path_ema':ema.state_dict()},out/'last_state.pt')
        if run is not None: run.log({'segment':seg,'step':step,'token_forecast_loss':row['token_forecast_loss'],'validation_objective':val_loss,'path_align_loss':row['path_align_loss'],'grad_norm':row['grad_norm'],'max_residual':row['max_residual']}, step=step)
        (out/'progress.json').write_text(json.dumps({'segment':seg,'step':step,'samples':seen},indent=2)); (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(row),flush=True)
    model.save_pretrained(out/'last_model')
    if run is not None: run.finish()
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--model-dir',required=True); p.add_argument('--tokenizer-dir',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--batch',type=int,default=64); p.add_argument('--segments',type=int,default=1); p.add_argument('--lr',type=float,default=2e-6); p.add_argument('--seed',type=int,default=20260914); p.add_argument('--device',default='cuda:0'); main(p.parse_args())
''')
alignment = ROOT / 'finetune' / 'stage3_path_alignment.py'
alignment.write_text(r'''from dataclasses import dataclass
import torch
import torch.nn.functional as F
@dataclass
class PathAlignmentConfig:
 top_k:int=16; candidates:int=16; weight:float=.05; huber_delta:float=.02; ema_decay:float=.99
class DetachedLossEMA:
 def __init__(self,decay=.99): self.decay=decay; self.value=None
 def normalize(self,loss):
  cur=loss.detach().float(); self.value=cur if self.value is None else self.decay*self.value+(1-self.decay)*cur
  return loss/self.value.clamp_min(1e-6).to(loss.dtype)
 def state_dict(self): return {'decay':self.decay,'value':None if self.value is None else self.value.detach().cpu()}
def candidate_mixture_decode(tok,a,b,top_k=16,candidates=16):
 k1=min(top_k,a.shape[-1]); k2=min(top_k,b.shape[-1]); p1=a.float().softmax(-1); p2=b.float().softmax(-1); v1,i1=torch.topk(p1,k1,-1); v2,i2=torch.topk(p2,k2,-1); outs=[]; ws=[]
 for n in range(min(candidates,k1*k2)): outs.append(tok.decode((i1[...,n%k1],i2[...,(n//k1)%k2]),half=True).float()); ws.append(v1[...,n%k1]*v2[...,(n//k1)%k2])
 w=torch.stack(ws); w=w/w.sum(0,keepdim=True).clamp_min(1e-8); return (torch.stack(outs)*w[...,None]).sum(0),w.detach()
def compute_path_alignment_loss(tok,a,b,target,cfg,normalizer=None):
 pred,w=candidate_mixture_decode(tok,a,b,cfg.top_k,cfg.candidates); target=target.to(pred.dtype); err=pred-target; per=F.huber_loss(pred,target,delta=cfg.huber_delta,reduction='none').mean((0,2)); loss=per.mean(); norm=normalizer.normalize(loss) if normalizer else loss
 return cfg.weight*norm, {'path_align_loss':loss.detach(),'path_align_horizon_mae':err.detach().abs().mean((0,2)),'path_align_max_residual':err.detach().abs().max()}
''')

def find_one(base, names):
    base = Path(base)
    for name in names:
        hits = list(base.rglob(name))
        if hits:
            return hits[0]
    return None

c2 = Path('/kaggle/input')
best_candidates = [p for p in c2.rglob('model.safetensors') if 'best_model' in str(p)]
model = best_candidates[0] if best_candidates else None
if model is None:
    raise FileNotFoundError('C2 checkpoints/best_model/model.safetensors not found')
model_dir = model.parent
tok = find_one(c2, ['Kronos-Tokenizer-base'])
if tok is None:
    tok = next((p for p in c2.rglob('*') if p.is_dir() and 'tokenizer' in p.name.lower()), None)
if tok is None:
    raise FileNotFoundError('C2 source tokenizer directory not found')
train = find_one('/kaggle/input', ['train_data.pkl'])
meta = find_one('/kaggle/input', ['asset_metadata.csv'])
if train is None or meta is None:
    raise FileNotFoundError('dataset train_data.pkl or asset_metadata.csv not found')
os.environ.update({'KRONOS_TRAIN_DATA_PATHS': str(train), 'KRONOS_METADATA_PATH': str(meta),
                   'KRONOS_LOOKBACK_WINDOW': '120', 'KRONOS_PREDICT_WINDOW': '10',
                   'KRONOS_USE_SIZE_PERCENTILE': '1', 'KRONOS_NUM_SIZE_BUCKETS': '0'})
if os.environ.get('SWANLAB_API_KEY'):
    os.environ['SWANLAB_MODE'] = 'cloud'
out = Path('/kaggle/working/stage3_path_alignment_from_c2_best_smoke')
cmd = [sys.executable, '-u', str(ROOT/'finetune/train_stage3_path_alignment.py'), '--model-dir', str(model_dir), '--tokenizer-dir', str(tok), '--output-dir', str(out), '--segments', '5', '--batch', '64', '--lr', '2e-6', '--device', 'cuda:0']
print(json.dumps({'model_dir': str(model_dir), 'tokenizer_dir': str(tok), 'train_data': str(train), 'metadata': str(meta), 'cmd': cmd}), flush=True)
subprocess.run(cmd, cwd=ROOT, check=True)
print(json.dumps({'output_files': sorted(str(p.relative_to(out)) for p in out.rglob('*') if p.is_file())}), flush=True)
