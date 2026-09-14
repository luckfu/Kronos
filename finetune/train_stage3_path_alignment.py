"""Standalone Stage-3 smoke trainer; leaves legacy train_predictor untouched."""
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
    model=Kronos.from_pretrained(a.model_dir).to(device); tok=KronosTokenizer.from_pretrained(a.tokenizer_dir).to(device).eval()
    for p in tok.parameters(): p.requires_grad_(False)
    model.train(); ds=QlibDataset('train'); loader=DataLoader(ds,batch_size=a.batch,shuffle=False,num_workers=0)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.01); ema=DetachedLossEMA(0.99); cfg=PathAlignmentConfig(16,16,0.05,0.02,0.99)
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); step=0
    summary = {'segments': [], 'model_dir': str(a.model_dir), 'tokenizer_dir': str(a.tokenizer_dir), 'seed': a.seed}
    for seg in range(1,a.segments+1):
        ds.set_epoch_seed(seg-1)
        seen=0; t=time.time()
        for batch in loader:
            x,stamp=batch[0].to(device),batch[1].to(device); sec=batch[2].to(device) if len(batch)>2 else None; pct=batch[4].to(device) if len(batch)>4 else None
            with torch.no_grad(): s1,s2=tok.encode(x,half=True)
            logits=model(s1[:,:-1],s2[:,:-1],stamp[:,:-1],use_teacher_forcing=True,s1_targets=s1[:,1:],sector_id=sec,size_percentile=pct)
            ce=model.head.compute_loss(logits[0],logits[1],s1[:,1:],s2[:,1:])[0]
            pa,metrics=compute_path_alignment_loss(tok,logits[0][:,-10:],logits[1][:,-10:],x[:,120:130],cfg,ema)
            loss=ce+pa; opt.zero_grad(set_to_none=True); loss.backward(); grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); step+=1; seen+=len(x)
            if seen>=20000: break
        torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'step':step,'segment':seg,'path_ema':ema.state_dict()},out/'last_state.pt')
        row = {'segment':seg,'step':step,'samples':seen,'token_loss':float(ce.detach()),'path_align_loss':float(metrics['path_align_loss']),'horizon_mae':metrics['path_align_horizon_mae'].tolist(),'max_residual':float(metrics['path_align_max_residual']),'grad_norm':float(grad),'elapsed_sec':time.time()-t}
        summary['segments'].append(row)
        (out/'progress.json').write_text(json.dumps({'segment': seg, 'step': step, 'samples': seen}, ensure_ascii=False, indent=2))
        (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(json.dumps(row,ensure_ascii=False),flush=True)
    model.save_pretrained(out/'last_model')

if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--model-dir',required=True); p.add_argument('--tokenizer-dir',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--batch',type=int,default=64); p.add_argument('--segments',type=int,default=5); p.add_argument('--lr',type=float,default=2e-6); p.add_argument('--seed',type=int,default=20260914); p.add_argument('--device',default='cuda:0'); main(p.parse_args())
