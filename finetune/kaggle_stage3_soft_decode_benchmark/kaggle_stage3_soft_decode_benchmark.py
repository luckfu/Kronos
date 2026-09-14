import json, os, subprocess, sys
from pathlib import Path
print(json.dumps({'phase':'discover_inputs'}, ensure_ascii=False), flush=True)
repo=Path('/kaggle/working/Kronos'); subprocess.run(['git','clone','--depth','1','https://github.com/luckfu/Kronos.git',str(repo)],check=True)
sys.path.insert(0,str(repo))
root=Path('/kaggle/input')
models=list(root.glob('**/small_0.1_stage2_cosine_refinement/**/best_model/config.json'))
model_dir=next((p.parent for p in models if (p.parent/'model.safetensors').exists()),None)
if model_dir is None: raise RuntimeError(f'C2 best_model not found under {root}: {models}')
tokenizer_dir=Path('/kaggle/working/Kronos-Tokenizer-base')
local_tok=next(iter(root.glob('**/Kronos-Tokenizer-base/config.json')),None)
if local_tok is not None:
    import shutil; shutil.copytree(local_tok.parent, tokenizer_dir, dirs_exist_ok=True)
else:
    raise RuntimeError('C2 Kernel source does not contain Kronos-Tokenizer-base')
val=next(root.glob('**/processed_datasets/val_data.pkl'),None); meta=next(root.glob('**/asset_metadata.csv'),None)
if val is None or meta is None: raise RuntimeError('validation data or metadata not found')
import time, torch
os.environ.update({'KRONOS_VAL_DATA_PATHS':str(val),'KRONOS_LOOKBACK_WINDOW':'120','KRONOS_PREDICT_WINDOW':'10','KRONOS_USE_SIZE_PERCENTILE':'1','KRONOS_NUM_SIZE_BUCKETS':'0','KRONOS_METADATA_PATH':str(meta),'KRONOS_VAL_SIGNAL_START':'2025-07-03','KRONOS_VAL_SIGNAL_END':'2026-07-02'})
from model.kronos import KronosTokenizer
from finetune.dataset import QlibDataset
from model.kronos import Kronos
dev=torch.device('cuda:0'); model=Kronos.from_pretrained(str(model_dir)).to(dev).eval(); tok=KronosTokenizer.from_pretrained(str(tokenizer_dir)).to(dev).eval()
for p in tok.parameters(): p.requires_grad_(False)
ds=QlibDataset('val'); rec=[ds[i] for i in range(64)]; x=torch.stack([r[0] for r in rec]).to(dev); st=torch.stack([r[1] for r in rec]).to(dev); sec=torch.stack([r[2] for r in rec]).to(dev); pct=torch.stack([r[4] for r in rec]).to(dev)
with torch.no_grad(): s1,s2=tok.encode(x,half=True)
torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(); t=time.perf_counter(); out=model(s1[:,:-1],s2[:,:-1],st[:,:-1],use_teacher_forcing=True,s1_targets=s1[:,1:],sector_id=sec,size_percentile=pct); torch.cuda.synchronize(); forward_ms=(time.perf_counter()-t)*1000; l1,l2=out; l1,l2=l1[:,-10:],l2[:,-10:]
torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(); t=time.perf_counter(); p1=l1.float().softmax(-1); p2=l2.float().softmax(-1); v1,i1=torch.topk(p1,16,-1); v2,i2=torch.topk(p2,16,-1); outs=[]; ws=[]
for n in range(16):
    o=tok.decode((i1[...,n%16],i2[...,n]),half=True).float(); outs.append(o); ws.append(v1[...,n%16]*v2[...,n])
w=torch.stack(ws); w=w/w.sum(0,keepdim=True).clamp_min(1e-8); mix=(torch.stack(outs)*w[...,None]).sum(0); torch.cuda.synchronize(); decode_ms=(time.perf_counter()-t)*1000; mem=torch.cuda.max_memory_allocated(dev)/2**30
loss=mix.square().mean(); model.zero_grad(set_to_none=True); torch.cuda.synchronize(); t=time.perf_counter(); loss.backward(); torch.cuda.synchronize(); backward_ms=(time.perf_counter()-t)*1000
print(json.dumps({'phase':'benchmark_finished','batch':64,'top_k':16,'candidates':16,'forward_ms':round(forward_ms,2),'mixture_decode_ms':round(decode_ms,2),'backward_ms':round(backward_ms,2),'peak_decode_gb':round(mem,3),'loss':float(loss.detach())},ensure_ascii=False),flush=True)
