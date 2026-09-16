"""Stage3 trainable-mask probes: freeze C2 except one module family."""
from __future__ import annotations

KNOWN_MASKS = (
    'all',
    'dependency_layer',
    'condition_branch',
    'blocks_7_8',
    'forecast_head',
)


def mask_predicate(mask: str):
    if mask == 'all':
        return lambda name: True
    if mask == 'dependency_layer':
        return lambda name: name.startswith('dep_layer.')
    if mask == 'condition_branch':
        return lambda name: name.startswith(('sector_emb.', 'size_emb.', 'size_mlp.'))
    if mask == 'blocks_7_8':
        return lambda name: name.startswith(('transformer.6.', 'transformer.7.'))
    if mask == 'forecast_head':
        return lambda name: name.startswith('head.')
    raise ValueError(f'Unknown trainable mask: {mask}')


def apply_trainable_mask(predictor, mask: str):
    """Freeze every parameter, then restore originally-trainable matches."""
    if mask not in KNOWN_MASKS:
        raise ValueError(f'Unknown trainable mask: {mask}')
    originally_trainable = {
        name: bool(parameter.requires_grad)
        for name, parameter in predictor.named_parameters()
    }
    matches = mask_predicate(mask)
    for parameter in predictor.parameters():
        parameter.requires_grad = False
    trainable_names = []
    for name, parameter in predictor.named_parameters():
        if matches(name) and originally_trainable[name]:
            parameter.requires_grad = True
            trainable_names.append(name)
    if not trainable_names:
        raise ValueError(f'Trainable mask {mask} selected no parameters')
    frozen_names = [
        name for name, parameter in predictor.named_parameters()
        if not parameter.requires_grad
    ]
    trainable_count = sum(
        parameter.numel() for parameter in predictor.parameters() if parameter.requires_grad
    )
    total_count = sum(parameter.numel() for parameter in predictor.parameters())
    return {
        'mask': mask,
        'trainable_names': trainable_names,
        'frozen_names': frozen_names,
        'trainable_tensors': len(trainable_names),
        'frozen_tensors': len(frozen_names),
        'trainable_parameters': trainable_count,
        'total_parameters': total_count,
        'trainable_fraction': trainable_count / max(total_count, 1),
    }


def snapshot_frozen_parameters(predictor):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in predictor.named_parameters()
        if not parameter.requires_grad
    }


def assert_frozen_parameters_unchanged(predictor, snapshot):
    current = dict(predictor.named_parameters())
    if set(snapshot) != {name for name, parameter in current.items() if not parameter.requires_grad}:
        raise RuntimeError('Frozen parameter set changed after applying the trainable mask')
    for name, baseline in snapshot.items():
        parameter = current[name]
        if parameter.requires_grad:
            raise RuntimeError(f'Frozen parameter became trainable: {name}')
        if not torch_equal(parameter, baseline):
            raise RuntimeError(f'Frozen parameter drifted: {name}')


def torch_equal(parameter, baseline):
    import torch
    return bool(torch.equal(parameter.detach().cpu(), baseline.cpu()))
