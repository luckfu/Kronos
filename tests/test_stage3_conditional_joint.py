"""CPU correctness checks; no optimizer steps, network access or OOS data."""
import torch
import pytest
from model import Kronos, KronosTokenizer
from finetune.stage3_path_alignment import (
    candidate_mixture_decode, compute_path_alignment_loss, PathAlignmentConfig,
)


def tiny_model():
    torch.manual_seed(71)
    return Kronos(4, 4, 2, 32, 4, 64, 0., 0., 0., 0., False)


def tiny_tokenizer():
    tok = KronosTokenizer(6, 32, 4, 64, 2, 2, 0., 0., 0., 4, 4,
                          0.25, 1., 1., 1., 4).eval()
    tok.requires_grad_(False)
    return tok


@pytest.mark.parametrize('training', [True, False])
@pytest.mark.parametrize('length,start', [(17, 6), (130, 119)])
def test_conditional_matches_original_full_sequence(training, length, start):
    model = tiny_model().train(training)
    context = torch.randn(2, length, 32)
    ids = torch.randint(0, 16, (2, 10, 4))
    actual = model.predict_s2_candidates(context, ids, position_start=start)
    assert actual.shape == (2, 10, 4, 16)
    for k in range(4):
        full_ids = torch.randint(0, 16, (2, length))
        full_ids[:, start:start + 10] = ids[..., k]
        expected = model.head.cond_forward(
            model.dep_layer(context, model.embedding.emb_s1(full_ids)))
        torch.testing.assert_close(actual[:, :, k], expected[:, start:start + 10])


@pytest.mark.parametrize('training', [True, False])
def test_extracted_context_preserves_teacher_forced_forward(training):
    model = tiny_model().train(training)
    ids = torch.randint(0, 16, (2, 17))
    targets = torch.randint(0, 16, ids.shape)
    # Original operations, independent of the extracted context method.
    context = model.token_drop(model.embedding((ids, ids)))
    for layer in model.transformer:
        context = layer(context)
    context = model.norm(context)
    expected = (model.head(context), model.head.cond_forward(
        model.dep_layer(context, model.embedding.emb_s1(targets))))
    actual = model(ids, ids, use_teacher_forcing=True, s1_targets=targets)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e)


def test_joint_selection_decode_shapes_and_probability():
    model, tok = tiny_model().eval(), tiny_tokenizer()
    ids = torch.randint(0, 16, (2, 17))
    context = model.encode_context(ids, ids)
    logits = model.predict_s1(context[:, -10:])
    pred, weights, extra = candidate_mixture_decode(model, tok, context, logits)
    assert pred.shape == (2, 10, 6)
    assert weights.shape == (2, 10, 16)
    assert extra['candidate_paths'].shape == (2, 10, 6, 16)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 10))
    p1, i1 = logits.log_softmax(-1).topk(16)
    p2 = model.predict_s2_candidates(context, i1).log_softmax(-1)
    joint = p1.unsqueeze(-1) + p2
    expected_logp, flat_ids = joint.flatten(-2).topk(16)
    torch.testing.assert_close(extra['selected_joint_logp'], expected_logp)
    assert torch.equal(extra['selected_s1_ids'], i1.gather(-1, flat_ids // 16))
    assert torch.equal(extra['selected_s2_ids'], flat_ids % 16)
    torch.testing.assert_close(pred, (extra['candidate_paths'] * weights.unsqueeze(-2)).sum(-1))


def test_path_only_gradients_and_conditional_sensitivity():
    model, tok = tiny_model().eval(), tiny_tokenizer()
    ids = torch.randint(0, 16, (2, 17))
    context = model.encode_context(ids, ids)
    logits = model.predict_s1(context[:, -10:])
    loss, _ = compute_path_alignment_loss(model, tok, context, logits,
                                          torch.randn(2, 10, 6), PathAlignmentConfig())
    loss.backward()
    for module in (model.head.proj_s1, model.head.proj_s2, model.dep_layer, model.transformer):
        grads = [p.grad for p in module.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        assert sum(g.square().sum() for g in grads) > 0
    assert all(p.grad is None for p in tok.parameters())
    candidates = torch.tensor([0, 1]).expand(2, 10, 2)
    cond = model.predict_s2_candidates(context, candidates)
    assert (cond[..., 0, :] - cond[..., 1, :]).abs().max() > 0


def test_joint_logit_finite_difference_and_movement():
    model, tok = tiny_model().eval(), tiny_tokenizer()
    ids = torch.randint(0, 16, (2, 17))
    ctx = model.encode_context(ids, ids)
    pred, _, extra = candidate_mixture_decode(model, tok, ctx, model.predict_s1(ctx[:, -10:]))
    # Fixed support and decoder anchors: differentiable part of this surrogate.
    lp = extra['selected_joint_logp'].detach().double().requires_grad_()
    paths = extra['candidate_paths'].double()
    target = torch.randn_like(pred).double()
    def objective(z):
        return ((paths * z.softmax(-1).unsqueeze(-2)).sum(-1) - target).square().mean()
    gradient, = torch.autograd.grad(objective(lp), lp)
    delta = torch.zeros_like(lp); delta[0, 0, 0] = 1e-5
    numeric = (objective(lp + delta) - objective(lp - delta)) / 2e-5
    torch.testing.assert_close(numeric, gradient[0, 0, 0], atol=1e-9, rtol=1e-5)
    before = (paths * lp.softmax(-1).unsqueeze(-2)).sum(-1)
    after = (paths * (lp + delta * 10000).softmax(-1).unsqueeze(-2)).sum(-1)
    direction = paths[0, 0, :, 0] - before[0, 0]
    assert torch.dot(after[0, 0] - before[0, 0], direction) > 0


def test_next_token_forecast_slice():
    source_row = torch.arange(131)
    next_token_targets = source_row[1:]
    assert torch.equal(next_token_targets[119:129], source_row[120:130])
    assert not torch.equal(next_token_targets[-10:], source_row[120:130])


@pytest.mark.parametrize('which', ['s1', 's2'])
def test_actual_candidate_pipeline_logit_finite_difference(which):
    model, tok = tiny_model().eval(), tiny_tokenizer()
    context = torch.randn(2, 17, 32)
    logits1 = torch.randn(2, 10, 16, requires_grad=True)
    logits2 = torch.randn(2, 10, 16, 16, requires_grad=True)
    target = torch.randn(2, 10, 6)
    class FixedConditional:
        def predict_s2_candidates(self, context, ids, position_start=None, is_causal=None):
            # IDs, not candidate ranks, select the corresponding conditional.
            return logits2.gather(2, ids.unsqueeze(-1).expand(-1, -1, -1, 16))
    def objective():
        prediction, _, _ = candidate_mixture_decode(FixedConditional(), tok, context, logits1)
        return (prediction - target).square().mean()
    parameter = logits1 if which == 's1' else logits2
    gradient, = torch.autograd.grad(objective(), parameter)
    index = int(gradient.abs().argmax())
    original = parameter.detach().flatten()[index].clone()
    with torch.no_grad():
        parameter.view(-1)[index] = original + 0.001
    plus = objective().detach()
    with torch.no_grad():
        parameter.view(-1)[index] = original - 0.001
    minus = objective().detach()
    with torch.no_grad():
        parameter.view(-1)[index] = original
    torch.testing.assert_close((plus - minus) / 0.002, gradient.flatten()[index],
                               atol=1e-4, rtol=0.03)
