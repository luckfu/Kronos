"""Standalone Stage-3 path-alignment trainer; leaves legacy train_predictor untouched.

Full holdout validation every segment on the same C2 val pool. Stage3 only
changes the loss (CE on the last 10 forecast steps plus path-align); it does
not switch to quick/large validation. KRONOS_VALIDATION_SAMPLES=0 means use
all total_samples (see finetune/dataset.py).
"""
import argparse, json, os, time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from torch.utils.data import DataLoader
from model.kronos import Kronos, KronosTokenizer
from finetune.dataset import QlibDataset
from stage3_path_alignment import PathAlignmentConfig, DetachedLossEMA, compute_path_alignment_loss

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
    model.train(); ds=QlibDataset('train'); loader=DataLoader(ds,batch_size=a.batch,shuffle=False,num_workers=0)
    os.environ['KRONOS_VALIDATION_SAMPLES']='0'
    val_ds=QlibDataset('val'); val_loader=DataLoader(val_ds,batch_size=a.batch,shuffle=False,num_workers=0)
    print('validation_samples=' + str(val_ds.total_samples), flush=True)
    if len(val_ds) != val_ds.total_samples or val_ds.total_samples < 100000:
        raise RuntimeError(f'Validation set unexpectedly small: {len(val_ds)}; full validation is required')
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01)
    ema=DetachedLossEMA(0.99); cfg=PathAlignmentConfig(16,16,0.05,0.02,0.99)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); step=0; best_val=float('inf')
    summary={'segments':[], 'model_dir':str(a.model_dir), 'tokenizer_dir':str(a.tokenizer_dir), 'seed':a.seed, 'validation_mode':'full'}
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
        (out/'progress.json').write_text(json.dumps({'segment':seg,'step':step,'samples':seen},indent=2)); (out/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(row),flush=True)
    model.save_pretrained(out/'last_model')

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--model-dir',required=True); p.add_argument('--tokenizer-dir',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--batch',type=int,default=64); p.add_argument('--segments',type=int,default=5); p.add_argument('--lr',type=float,default=2e-6); p.add_argument('--seed',type=int,default=20260914); p.add_argument('--device',default='cuda:0'); main(p.parse_args())
