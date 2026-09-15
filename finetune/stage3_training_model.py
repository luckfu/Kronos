"""DDP-owned Stage3 objective; leaves production Kronos.forward unchanged."""
from dataclasses import asdict
import torch
from torch import nn

from finetune.stage3_path_alignment import (
    DetachedLossEMA, PathAlignmentConfig, compute_path_alignment_loss,
)


class Stage3TrainingModel(nn.Module):
    def __init__(self, predictor, tokenizer, config=None, lookback=120, horizon=10,
                 synchronize_ema=True):
        super().__init__()
        self.predictor = predictor
        self.tokenizer = tokenizer.requires_grad_(False).eval()
        self.config = config or PathAlignmentConfig()
        self.lookback, self.horizon = int(lookback), int(horizon)
        self.synchronize_ema = bool(synchronize_ema)
        self.path_ema = DetachedLossEMA(self.config.ema_decay)

    def train(self, mode=True):
        super().train(mode)
        # DDP/model.train must never enable dropout in the frozen tokenizer.
        self.tokenizer.eval()
        return self

    def forward(self, x, stamp=None, sector_id=None, size_percentile=None):
        if x.shape[1:] != (self.lookback + self.horizon + 1, 6):
            raise ValueError('Stage3 requires lookback + horizon + 1 rows and six features')
        with torch.no_grad():
            s1, s2 = self.tokenizer.encode(x.float(), half=True)
        context = self.predictor.encode_context(
            s1[:, :-1], s2[:, :-1], None if stamp is None else stamp[:, :-1],
            sector_id=sector_id, size_percentile=size_percentile,
        )
        start, end = self.lookback - 1, self.lookback + self.horizon - 1
        target_slice = slice(self.lookback, self.lookback + self.horizon)
        logits1 = self.predictor.predict_s1(context[:, start:end])
        logits2 = self.predictor.predict_s2(context, s1[:, 1:], is_causal=True)[:, start:end]
        token_loss = self.predictor.head.compute_loss(
            logits1, logits2, s1[:, target_slice], s2[:, target_slice],
        )[0]
        if self.config.weight == 0 and self.training:
            # CE-only control: the candidate decode is pure overhead at lambda 0.
            # Validation still decodes so raw path metrics stay comparable with C3.
            zeros_h = token_loss.new_zeros((self.horizon,))
            zeros_hf = token_loss.new_zeros((self.horizon, 6))
            with torch.no_grad():
                logp1 = logits1.float().log_softmax(-1)
                s1_entropy = -(logp1.exp() * logp1).sum(-1).mean()
            metrics = {
                'token_loss': token_loss.detach(),
                'raw_path_loss': token_loss.new_zeros(()),
                'normalized_path_loss': token_loss.new_zeros(()),
                'weighted_path_loss': token_loss.new_zeros(()),
                'total_loss': token_loss.detach(),
                'top16_joint_mass': token_loss.new_zeros(()),
                's1_entropy': s1_entropy,
                's2_conditional_entropy_topk_s1': token_loss.new_zeros(()),
                'horizon_mae': zeros_h,
                'max_residual': token_loss.new_zeros(()),
                'prediction_mean_hf': zeros_hf,
                'prediction_second_moment_hf': zeros_hf.square(),
                'target_mean_hf': x[:, target_slice].detach().mean(0),
                'target_second_moment_hf': x[:, target_slice].detach().square().mean(0),
            }
            return token_loss, metrics
        path_loss, details = compute_path_alignment_loss(
            self.predictor, self.tokenizer, context, logits1, x[:, target_slice],
            self.config, self.path_ema if self.training else None,
            position_start=start, is_causal=True,
            synchronize_ema=self.synchronize_ema and self.training,
        )
        # compute_path_alignment_loss already applies lambda_path once.
        total = token_loss + path_loss
        with torch.no_grad():
            logp1 = logits1.float().log_softmax(-1)
            logp2 = details['s2_conditional_logits'].float().log_softmax(-1)
            p1_top = logp1.exp().topk(logp2.shape[-2], dim=-1).values
            conditional_entropy = -(logp2.exp() * logp2).sum(-1)
            conditional_entropy = (conditional_entropy * p1_top).sum(-1) / p1_top.sum(-1)
            metrics = {
                'token_loss': token_loss.detach(),
                'raw_path_loss': details['path_align_loss'],
                'normalized_path_loss': details['path_align_normalized'],
                'weighted_path_loss': path_loss.detach(),
                'total_loss': total.detach(),
                'top16_joint_mass': details['selected_joint_logp'].detach().exp().sum(-1).mean(),
                's1_entropy': -(logp1.exp() * logp1).sum(-1).mean(),
                's2_conditional_entropy_topk_s1': conditional_entropy.mean(),
                'horizon_mae': details['path_align_horizon_mae'],
                'max_residual': details['path_align_max_residual'],
                'prediction_mean_hf': details['path_align_prediction'].detach().mean(0),
                'prediction_second_moment_hf': details['path_align_prediction'].detach().square().mean(0),
                'target_mean_hf': x[:, target_slice].detach().mean(0),
                'target_second_moment_hf': x[:, target_slice].detach().square().mean(0),
            }
        return total, metrics

    def checkpoint_state(self, optimizer, step, segment):
        """Predictor keys stay production-compatible; training state is separate."""
        return {
            'schema': 'stage3_conditional_joint_causal_v2',
            'model': self.predictor.state_dict(), 'optimizer': optimizer.state_dict(),
            'path_ema': self.path_ema.state_dict(), 'step': int(step), 'segment': int(segment),
            'path_config': asdict(self.config), 'lookback': self.lookback, 'horizon': self.horizon,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }

    def load_checkpoint_state(self, state, optimizer):
        """Only resume this Stage3 schema, never import C2 optimizer state."""
        if state.get('schema') != 'stage3_conditional_joint_causal_v2':
            raise ValueError('Not a compatible Stage3 training checkpoint')
        if (state['path_config'] != asdict(self.config) or
                state['lookback'] != self.lookback or state['horizon'] != self.horizon):
            raise ValueError('Stage3 objective configuration changed on resume')
        self.predictor.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        self.path_ema.load_state_dict(state['path_ema'])
        torch.set_rng_state(state['torch_rng_state'].cpu())
        if state.get('cuda_rng_state') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state(state['cuda_rng_state'].cpu())
        return int(state['step']), int(state['segment'])


def gradient_metrics(predictor):
    groups = {
        's1_grad_norm': predictor.head.proj_s1,
        's2_grad_norm': predictor.head.proj_s2,
        'dependency_grad_norm': predictor.dep_layer,
        'backbone_grad_norm': predictor.transformer,
    }
    result = {}
    for name, module in groups.items():
        terms = [p.grad.detach().float().square().sum() for p in module.parameters() if p.grad is not None]
        result[name] = torch.stack(terms).sum().sqrt() if terms else next(predictor.parameters()).new_zeros(())
    return result
