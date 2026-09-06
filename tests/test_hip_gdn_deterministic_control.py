"""Opt-in fixed-order GDN reduction must match the existing independent oracle."""
import pytest
import torch

from test_gated_delta_rule import _run_cuda_gated_delta_rule, _torch_gated_delta_rule


@pytest.mark.parametrize('split', [1, 2, 4])
@pytest.mark.parametrize('history', [False, True])
@torch.inference_mode()
def test_fixed_order_replay_and_reference(monkeypatch, split, history):
    if not torch.cuda.is_available():
        pytest.skip('GPU required')
    monkeypatch.setenv('EXL3_DIAGNOSTIC_GDN_FIXED_ORDER', '1')
    monkeypatch.setenv('EXL3_HIP_GDN_PREFILL_VSPLIT', str(split))
    torch.manual_seed(1234)
    device = 'cuda:0'
    length, kh, vh, dim = 17, 2, 6, 128
    qkv = (torch.randn(1, length, (2 * kh + vh) * dim, device=device) * .25).bfloat16()
    g = torch.randn(1, length, vh, device=device) * .5 - 1.
    beta = torch.randn(1, length, vh, device=device).sigmoid().bfloat16()
    initial = torch.randn(3, length if history else 1, vh, dim, dim, device=device) * .05
    slots = torch.tensor([1], device=device, dtype=torch.int32)
    args = (qkv, g, beta, initial, slots, history, kh, vh, dim, dim)
    reference = _torch_gated_delta_rule(*[a.cpu() if isinstance(a, torch.Tensor) else a for a in args])
    base = None
    for _ in range(12):
        outputs = _run_cuda_gated_delta_rule(*args)
        if base is None:
            base = outputs
        for actual, previous, expected in zip(outputs, base, reference):
            assert torch.equal(actual.contiguous().view(torch.uint8), previous.contiguous().view(torch.uint8))
            torch.testing.assert_close(actual.cpu(), expected, rtol=.05, atol=.05)
        for unused_slot in (0, 2):
            assert torch.equal(outputs[1][unused_slot], initial[unused_slot])
