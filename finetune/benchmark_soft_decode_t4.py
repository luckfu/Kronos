"""CUDA-only benchmark for the Stage 3 candidate-mixture decode."""
import argparse, os, sys, time
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finetune.validate_soft_decode import load_from_dirs, sampled_mixture_decode
from finetune.dataset import QlibDataset

def main(a):
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    os.environ.update({'KRONOS_VAL_DATA_PATHS':str(Path(a.val_data).resolve()),'KRONOS_LOOKBACK_WINDOW':'120','KRONOS_PREDICT_WINDOW':'10','KRONOS_USE_SIZE_PERCENTILE':'1','KRONOS_NUM_SIZE_BUCKETS':'0','KRONOS_METADATA_PATH':str(Path(a.metadata).resolve()),'KRONOS_VAL_SIGNAL_START':'2025-07-03','KRONOS_VAL_SIGNAL_END':'2026-07-02'})
    dev=torch.device('cuda:0'); model,tok=load_from_dirs(Path(a.model_dir),Path(a.tokenizer_dir),dev); ds=QlibDataset('val'); rec=[ds[i] for i in range(a.batch)]; x=torch.stack([r[0] for r in rec]).to(dev); st=torch.stack([r[1] for r in rec]).to(dev); sec=torch.stack([r[2] for r in rec]).to(dev); pct=torch.stack([r[4] for r in rec]).to(dev)
    with torch.no_grad(): s1,s2=tok.encode(x,half=True)
    torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(); t0=time.perf_counter(); out=model(s1[:,:-1],s2[:,:-1],st[:,:-1],use_teacher_forcing=True,s1_targets=s1[:,1:],sector_id=sec,size_percentile=pct); torch.cuda.synchronize(); forward_ms=(time.perf_counter()-t0)*1000; l1,l2=out; l1,l2=l1[:,-10:],l2[:,-10:]
    torch.cuda.reset_peak_memory_stats(dev); torch.cuda.synchronize(); t0=time.perf_counter(); mix=sampled_mixture_decode(tok,l1,l2,a.top_k,a.candidates); torch.cuda.synchronize(); decode_ms=(time.perf_counter()-t0)*1000; decode_mem=torch.cuda.max_memory_allocated(dev)/2**30
    loss=mix.float().square().mean(); model.zero_grad(set_to_none=True); torch.cuda.synchronize(); t0=time.perf_counter(); loss.backward(); torch.cuda.synchronize(); backward_ms=(time.perf_counter()-t0)*1000; print({'batch':a.batch,'top_k':a.top_k,'candidates':a.candidates,'forward_ms':round(forward_ms,2),'mixture_decode_ms':round(decode_ms,2),'backward_ms':round(backward_ms,2),'peak_decode_gb':round(decode_mem,3),'loss':float(loss.detach())})
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--model-dir',required=True); p.add_argument('--tokenizer-dir',required=True); p.add_argument('--val-data',required=True); p.add_argument('--metadata',required=True); p.add_argument('--batch',type=int,default=64); p.add_argument('--top-k',type=int,default=16); p.add_argument('--candidates',type=int,default=16); main(p.parse_args())
