"""DDP-owned Stage3 objective; leaves production Kronos.forward unchanged."""
from dataclasses import asdict
import torch
from torch import nn

from finetune.stage3_ce_rank import Stage3CERankConfig, compute_rank_terms, compute_weighted_ce_objective
from finetune.stage3_path_alignment import (
    DetachedLossEMA, PathAlignmentConfig, compute_path_alignment_loss,
)
from finetune.stage3_vol_alignment import VolAlignmentConfig, compute_vol_alignment_loss, CLOSE


class Stage3TrainingModel(nn.Module):
    def __init__(self, predictor, tokenizer, config=None, lookback=120, horizon=10,
                 synchronize_ema=True, ce_rank_config=None, vol_config=None):
        super().__init__()
        self.predictor = predictor
        self.tokenizer = tokenizer.requires_grad_(False).eval()
        self.config = config or PathAlignmentConfig()
        self.vol_config = vol_config or VolAlignmentConfig(weight=0.0)
        self.ce_rank_config = ce_rank_config or Stage3CERankConfig()
        self.lookback, self.horizon = int(lookback), int(horizon)
        self.synchronize_ema = bool(synchronize_ema)
        self.path_ema = DetachedLossEMA(self.config.ema_decay)
        self.vol_ema = DetachedLossEMA(self.vol_config.ema_decay)

    def train(self, mode=True):
        super().train(mode)
        self.tokenizer.eval()
        return self

    def _ce_rank_forward(self, x, s1, s2, context, date_ids, feature_means, feature_stds):
        seq_end = self.lookback + self.horizon - 1
        target_slice = slice(self.lookback, self.lookback + self.horizon)
        targets1 = s1[:, 1:seq_end + 1]
        targets2 = s2[:, 1:seq_end + 1]
        logits1 = self.predictor.predict_s1(context[:, :seq_end])
        logits2 = self.predictor.predict_s2(
            context, s1[:, 1:], is_causal=True,
        )[:, :seq_end]
        ce_terms = compute_weighted_ce_objective(
            self.predictor.head, logits1, logits2, targets1, targets2,
            self.lookback, self.horizon, self.ce_rank_config,
        )
        ce_objective = ce_terms['objective']
        rank_loss = ce_objective.new_zeros(())
        rank_diag = {}
        if (
            self.training
            and self.ce_rank_config.lambda_rank != 0
            and date_ids is not None
            and feature_means is not None
        ):
            rank_loss, rank_diag = compute_rank_terms(
                self.predictor, self.tokenizer, context, logits1, x,
                feature_means, feature_stds, date_ids,
                self.lookback, self.horizon, self.ce_rank_config,
            )
        total = ce_objective + self.ce_rank_config.lambda_rank * rank_loss
        with torch.no_grad():
            logp1 = logits1[:, self.lookback - 1:].float().log_softmax(-1)
            s1_entropy = -(logp1.exp() * logp1).sum(-1).mean()
            metrics = {
                'token_loss': ce_objective.detach(),
                'ce_objective': ce_objective.detach(),
                'history_loss': ce_terms['history_loss'].detach(),
                'weighted_forecast_loss': ce_terms['weighted_forecast_loss'].detach(),
                'rank_loss': rank_loss.detach(),
                'raw_path_loss': ce_objective.new_zeros(()),
                'normalized_path_loss': ce_objective.new_zeros(()),
                'weighted_path_loss': ce_objective.new_zeros(()),
                'total_loss': total.detach(),
                'top16_joint_mass': ce_objective.new_zeros(()),
                's1_entropy': s1_entropy,
                's2_conditional_entropy_topk_s1': ce_objective.new_zeros(()),
                'horizon_mae': ce_objective.new_zeros((self.horizon,)),
                'max_residual': ce_objective.new_zeros(()),
                'prediction_mean_hf': ce_objective.new_zeros((self.horizon, 6)),
                'prediction_second_moment_hf': ce_objective.new_zeros((self.horizon, 6)),
                'target_mean_hf': x[:, target_slice].detach().mean(0),
                'target_second_moment_hf': x[:, target_slice].detach().square().mean(0),
            }
            metrics.update({key: value for key, value in rank_diag.items() if key != 'rank_loss'})
        return total, metrics

    def forward(self, x, stamp=None, sector_id=None, size_percentile=None,
                date_ids=None, feature_means=None, feature_stds=None):
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
        if self.ce_rank_config.enabled:
            if self.config.weight != 0 or self.vol_config.weight != 0:
                raise ValueError('P1 CE+rank requires lambda_path=0 and lambda_vol=0')
            return self._ce_rank_forward(x, s1, s2, context, date_ids, feature_means, feature_stds)
        logits1 = self.predictor.predict_s1(context[:, start:end])
        logits2 = self.predictor.predict_s2(context, s1[:, 1:], is_causal=True)[:, start:end]
        token_loss = self.predictor.head.compute_loss(
            logits1, logits2, s1[:, target_slice], s2[:, target_slice],
        )[0]
        if self.vol_config.weight != 0:
            if self.config.weight != 0:
                raise ValueError('vol alignment requires lambda_path=0')
            if feature_means is None or feature_stds is None:
                raise ValueError('vol alignment requires feature_means and feature_stds')
            vol_loss, details = compute_vol_alignment_loss(
                self.predictor, self.tokenizer, context, logits1, x[:, target_slice],
                x[:, self.lookback - 1, CLOSE], feature_means, feature_stds,
                self.vol_config, self.vol_ema if self.training else None,
                position_start=start, is_causal=True,
                synchronize_ema=self.synchronize_ema and self.training,
            )
            total = token_loss + vol_loss
            self._live_token_loss = token_loss
            self._live_vol_loss = vol_loss
            zeros_h = token_loss.new_zeros((self.horizon,))
            zeros_hf = token_loss.new_zeros((self.horizon, 6))
            with torch.no_grad():
                logp1 = logits1.float().log_softmax(-1)
                s1_entropy = -(logp1.exp() * logp1).sum(-1).mean()
            return total, {
                'token_loss': token_loss.detach(),
                'raw_path_loss': details['vol_align_loss'],
                'normalized_path_loss': details['vol_align_normalized'],
                'weighted_path_loss': vol_loss.detach(),
                'total_loss': total.detach(),
                'top16_joint_mass': details['selected_joint_logp'].detach().exp().sum(-1).mean(),
                's1_entropy': s1_entropy,
                's2_conditional_entropy_topk_s1': token_loss.new_zeros(()),
                'horizon_mae': zeros_h,
                'max_residual': token_loss.new_zeros(()),
                'prediction_mean_hf': zeros_hf,
                'prediction_second_moment_hf': zeros_hf.square(),
                'target_mean_hf': x[:, target_slice].detach().mean(0),
                'target_second_moment_hf': x[:, target_slice].detach().square().mean(0),
                'vol_calibration_ratio': details['vol_calibration_ratio'],
                'pred_path_vol': details['pred_path_vol'],
                'realized_path_vol': details['realized_path_vol'],
                'mixture_mean_path_vol': details['mixture_mean_path_vol'],
            }
        if self.config.weight == 0:
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
        payload = {
            'schema': 'stage3_vol_calibration_v2' if self.vol_config.weight else 'stage3_conditional_joint_causal_v2',
            'model': self.predictor.state_dict(), 'optimizer': optimizer.state_dict(),
            'path_ema': self.path_ema.state_dict(), 'vol_ema': self.vol_ema.state_dict(),
            'step': int(step), 'segment': int(segment),
            'path_config': asdict(self.config), 'vol_config': asdict(self.vol_config),
            'ce_rank_config': asdict(self.ce_rank_config),
            'lookback': self.lookback, 'horizon': self.horizon,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        return payload

    def load_checkpoint_state(self, state, optimizer):
        expected = 'stage3_vol_calibration_v2' if self.vol_config.weight else 'stage3_conditional_joint_causal_v2'
        if state.get('schema') != expected:
            raise ValueError('Not a compatible Stage3 training checkpoint')
        if (state['path_config'] != asdict(self.config) or
                state.get('vol_config', asdict(VolAlignmentConfig(weight=0.0))) != asdict(self.vol_config) or
                state.get('ce_rank_config', asdict(Stage3CERankConfig())) != asdict(self.ce_rank_config) or
                state['lookback'] != self.lookback or state['horizon'] != self.horizon):
            raise ValueError('Stage3 objective configuration changed on resume')
        self.predictor.load_state_dict(state['model'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        self.path_ema.load_state_dict(state['path_ema'])
        if 'vol_ema' in state:
            self.vol_ema.load_state_dict(state['vol_ema'])
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
