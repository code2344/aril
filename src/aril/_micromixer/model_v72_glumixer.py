# Adapted from MicroMixer-4 (Apache-2.0): package imports were adjusted.
"""V72 GLU-Mixer -- V71-GLC + SwiGLU channel-mix (param-parity swap).

V72 is a single-change ablation over the V71-GLC champion backbone
(d=128, 7 blocks, glc_kernel=129 -> 994,458 params). The only architectural
difference vs V71-GLC is that the per-block GELU ChannelMLP
(Linear d->2d->d, GELU) is replaced by a SwiGLU MLP
(Linear d->h gate, Linear d->h up, Linear h->d down) sized to keep total
parameter count below the V71-GLC budget.

  GELU ChannelMLP(d=128): fc1 + fc2 = 65,920 params/block
  SwiGLU(d=128, h=170):   gate + up + down = 65,748 params/block
                          -> saves 172/block x 7 = 1,204 params
  v72_glu_1m total:       994,458 - 1,204 = 993,254 params (<1M, exact)

The hidden width formula is chosen so that the SwiGLU parameter count
bracketingly matches the GELU MLP (3dh + 2h + d == 2d^2 + 3d + eps):

    h = (4*d*d + 2*d) // (3*d + 2)
        d=128 -> h = 170

Everything else (V71-GLC token-mix, V59 selectivity sidecar, V67 frozen
content pool, RMSNorm, byte-tied embeddings) is inherited unchanged from
MicroMixerV71RTM. No attention, no recurrence, no SSM.

Self-test: 10/10 PASS.
"""

from __future__ import annotations

import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_v37_qag import RMSNorm
from .model_v71_rtm import (
    V71Config,
    MicroMixerV71RTM,
    v71_rtm_glc_1m,
    count_parameters,
)


# ============================================================================
# SwiGLU channel-mix (the only architectural change vs V71-GLC)
# ============================================================================

def swiglu_hidden(d: int) -> int:
    """SwiGLU hidden width that bracketingly matches GELU ChannelMLP params.

        GELU ChannelMLP(e=2): 2*d^2 + 3d
        SwiGLU:               3*d*h + 2*h + d      (h = hidden width)

    Setting SwiGLU <= GELU:
        3dh + 2h + d <= 2d^2 + 3d
        h(3d + 2)   <= 2d^2 + 2d
        h            <= (2d^2 + 2d) / (3d + 2)

    We pick h = floor((4d^2 + 2d) / (3d + 2)) (slightly above the equality
    point because the GELU bias term grows as 3d, not 2d -- this keeps the
    swap within ~0.3% param parity and lands at exactly 170 for d=128).
    """
    return (4 * d * d + 2 * d) // (3 * d + 2)


class SwiGLUChannelMLP(nn.Module):
    """SwiGLU channel-mix:

        out = down_proj( silu(gate_proj(x)) * up_proj(x) )

    House Linear init: xavier_uniform_ weight, zeros_ bias (matches V71's
    `_init_weights`). Init is applied inside __init__ so replacement modules
    constructed after the model-wide _init_weights loop are still correct.
    """

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden, bias=True)
        self.up_proj = nn.Linear(d_model, hidden, bias=True)
        self.down_proj = nn.Linear(hidden, d_model, bias=True)
        # House Linear init (matches V71 _init_weights).
        for lin in (self.gate_proj, self.up_proj, self.down_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ============================================================================
# V72 GLU-Mixer model
# ============================================================================

class MicroMixerV72GLU(MicroMixerV71RTM):
    """V72 GLU-Mixer: V71-GLC backbone with each block's GELU ChannelMLP
    replaced by SwiGLUChannelMLP.

    The GELU ChannelMLP is constructed first by `super().__init__()`, then
    discarded by the per-block swap below. This is acceptable: it costs
    only a few extra milliseconds at construction time, and it lets V72
    inherit V71's full forward / generate / _init_weights pipeline
    unchanged. The replacement modules are assigned BEFORE any state_dict
    load (which happens externally, after __init__ returns), so the SwiGLU
    weights are the ones that get loaded. Init happens inside
    SwiGLUChannelMLP.__init__, so nothing else is needed here.
    """

    def __init__(self, config: V71Config):
        super().__init__(config)
        # Swap channel_mix in every block. The V71Config is reused with
        # tmix="glc"; the only change is the channel-mix module.
        for b in self.blocks:
            b.channel_mix = SwiGLUChannelMLP(config.d_model, swiglu_hidden(config.d_model))


# ============================================================================
# Presets
# ============================================================================

def v72_glu_100k() -> V71Config:
    """~83K GLC+SwiGLU preset (d=48, 3 blocks, glc_kernel=129)."""
    return V71Config(tmix="glc", d_model=48, num_blocks=3, max_seq_len=512)


def v72_glu_1m() -> V71Config:
    """~993K GLC+SwiGLU preset (d=128, 7 blocks, glc_kernel=129).

    Exact param count: 993,254 (V71-GLC-1M = 994,458 minus 1,204 saved by
    SwiGLU vs GELU: 172/block x 7 blocks).
    """
    return V71Config(tmix="glc", d_model=128, num_blocks=7, max_seq_len=1024)


# ============================================================================
# Self-test (10 invariants, mirrors V71 structure)
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 72)
    print("MicroMixer-4 V72 GLU-Mixer -- self-test")
    print("(V71-GLC backbone + SwiGLU channel-mix, single-change ablation)")
    print("=" * 72)

    PASS_COUNT = 0
    TOTAL = 10
    V = 256

    # === [1] Param budgets: v72_glu_1m exact (993,254); v72_glu_100k <1M ===
    print("\n[1] Parameter budgets (v72_glu_1m EXACT 993,254; v72_glu_100k <1M):")
    m_1m = MicroMixerV72GLU(v72_glu_1m())
    n_1m = count_parameters(m_1m)
    m_100k = MicroMixerV72GLU(v72_glu_100k())
    n_100k = count_parameters(m_100k)
    expected_1m = 993_254
    ok_1m = (n_1m == expected_1m) and (n_1m < 1_000_000)
    ok_100k = n_100k < 1_000_000
    print(f"  v72_glu_1m:   {n_1m:>10,} (expected {expected_1m:>10,}) "
          f"[{'<1M' if n_1m < 1_000_000 else 'OVER'}] [{'OK' if ok_1m else 'FAIL'}]")
    print(f"  v72_glu_100k: {n_100k:>10,} "
          f"[{'<1M' if n_100k < 1_000_000 else 'OVER'}] [{'OK' if ok_100k else 'FAIL'}]")
    assert ok_1m, f"v72_glu_1m expected {expected_1m}, got {n_1m}"
    assert ok_100k, f"v72_glu_100k >= 1M: {n_100k}"
    print("  [PASS]")
    PASS_COUNT += 1

    # === [2] Forward + backward, finite grads (B=4, S=64) ===
    print("\n[2] Forward + backward (v72_glu_1m, B=4, S=64):")
    cfg = v72_glu_1m()
    m = MicroMixerV72GLU(cfg)
    B, S = 4, 64
    ids = torch.randint(4, V, (B, S))
    targets = torch.cat([ids[:, 1:], torch.full((B, 1), 0, dtype=torch.long)], dim=1)
    m.train()
    logits, loss, aux = m(ids, None, None, None, targets)
    assert logits.shape == (B, S, V)
    assert torch.isfinite(loss).item()
    loss.backward()
    finite = all(torch.isfinite(p.grad).all().item()
                 for p in m.parameters() if p.grad is not None)
    assert finite, "non-finite gradients"
    print(f"  loss={loss.item():.4f} acc={aux['token_acc'].item():.4f} "
          f"finite_grads={finite} logits={tuple(logits.shape)} [PASS]")
    PASS_COUNT += 1

    # === [3] Causality: perturb last byte -> past logits unchanged ===
    print("\n[3] Causality (perturb last byte, check past logits):")
    m = MicroMixerV72GLU(v72_glu_1m())
    m.eval()
    S_t = 64
    ids_a = torch.randint(4, V, (1, S_t))
    ids_b = ids_a.clone()
    ids_b[0, -1] = (ids_b[0, -1] + 1) % V
    with torch.no_grad():
        la = m(ids_a, None, None, None, None)[0]
        lb = m(ids_b, None, None, None, None)[0]
    diff = (la[:, :-1] - lb[:, :-1]).abs().max().item()
    print(f"  v72_glu_1m (S={S_t}): past-logits diff = {diff:.2e}")
    assert diff == 0.0, f"causality violated: {diff}"
    print("  [PASS]")
    PASS_COUNT += 1

    # === [4] Interface contract (5-arg forward, aux keys, generate kwargs,
    #         deepcopy, out_norm) ===
    print("\n[4] Interface contract (5-arg forward, aux keys, generate kwargs, deepcopy, out_norm):")
    import inspect
    sig = inspect.signature(MicroMixerV72GLU.forward)
    params = list(sig.parameters.keys())
    assert params == ["self", "input_ids", "q_mask", "a_mask", "all_mask", "targets"], \
        f"forward signature mismatch: {params}"
    gsig = inspect.signature(MicroMixerV72GLU.generate)
    gparams = list(gsig.parameters.keys())
    assert gparams == ["self", "input_ids", "q_mask", "a_mask", "all_mask",
                       "max_new_tokens", "temperature", "top_k", "top_p",
                       "repetition_penalty", "no_repeat_ngram_size", "eos_token_id"], \
        f"generate signature mismatch: {gparams}"
    # aux with targets
    _, _, aux_full = m(ids, None, None, None, targets)
    expected_aux = {"ce_loss", "ce_unweighted", "main_loss", "token_acc", "loss_weight_mean"}
    assert expected_aux.issubset(aux_full.keys()), f"aux missing keys: {aux_full.keys()}"
    # aux when targets=None
    _, _, aux_none = m(ids, None, None, None, None)
    assert set(aux_none.keys()) == {"token_acc"}, f"aux(targets=None) wrong: {aux_none.keys()}"
    # out_norm + config + embed present
    assert isinstance(m.out_norm, RMSNorm), "out_norm not RMSNorm"
    assert isinstance(m.config, V71Config), "config not V71Config"
    assert isinstance(m.embed, nn.Embedding), "embed not nn.Embedding"
    # deepcopy works on a FRESH model (V67 ReMixerLayer sets last_pool_pre during
    # forward -- non-leaf tensor that cannot be deepcopy'd after a forward call;
    # the deepcopy contract applies to a model that has not yet been called.)
    m_fresh = MicroMixerV72GLU(v72_glu_1m())
    m_fresh.eval()
    m_deep = copy.deepcopy(m_fresh)
    m_deep.eval()
    test_ids2 = torch.randint(4, V, (2, 32))
    with torch.no_grad():
        out_orig = m_fresh(test_ids2, None, None, None, None)[0]
        out_deep = m_deep(test_ids2, None, None, None, None)[0]
    deep_diff = (out_orig - out_deep).abs().max().item()
    assert deep_diff == 0.0, f"deepcopy not bit-identical: {deep_diff}"
    print(f"  forward 5 args OK; aux keys = {sorted(aux_full.keys())}")
    print(f"  generate kwargs OK; deepcopy(fresh) bit-identical (diff={deep_diff:.0e})")
    print("  [PASS]")
    PASS_COUNT += 1

    # === [5] Module scan: no attention / recurrence / SSM ===
    print("\n[5] Module scan (no attention/recurrence/SSM):")
    forbidden_types = (nn.MultiheadAttention, nn.RNN, nn.LSTM, nn.GRU,
                       nn.RNNCell, nn.LSTMCell, nn.GRUCell)
    has_forbidden_type = any(
        isinstance(mod, forbidden_types) for _, mod in m.named_modules()
    )
    has_forbidden_name = any(
        ("attention" in n.lower() or "attn" in n.lower()
         or "s4" in n.lower() or "mamba" in n.lower())
        for n, _ in m.named_modules()
    )
    assert not has_forbidden_type, "forbidden module type present"
    assert not has_forbidden_name, "forbidden module name present"
    n_swiglu = sum(1 for _, mod in m.named_modules() if isinstance(mod, SwiGLUChannelMLP))
    print(f"  forbidden types/names present: {has_forbidden_type or has_forbidden_name}")
    print(f"  SwiGLUChannelMLP count: {n_swiglu} (expect 7 in 1M preset)")
    assert n_swiglu == 7, f"expected 7 SwiGLU blocks, got {n_swiglu}"
    print("  [PASS] no attention/recurrence/SSM")
    PASS_COUNT += 1

    # === [6] SwiGLU param parity: each block channel_mix has exactly 65,748
    #         params at d=128 ===
    print("\n[6] SwiGLU param parity (each block channel_mix == 65,748 at d=128):")
    m = MicroMixerV72GLU(v72_glu_1m())
    expected_block = 65_748
    all_match = True
    for i, block in enumerate(m.blocks):
        cm_params = sum(p.numel() for p in block.channel_mix.parameters())
        ok = cm_params == expected_block
        if not ok:
            all_match = False
        print(f"  block {i}: channel_mix = {cm_params:,} params "
              f"[{'OK' if ok else 'FAIL ' + str(expected_block)}]")
    assert all_match, "SwiGLU per-block param count mismatch"
    # Also verify gate/up/down sub-structure.
    cm0 = m.blocks[0].channel_mix
    assert isinstance(cm0.gate_proj, nn.Linear) and cm0.gate_proj.out_features == 170
    assert isinstance(cm0.up_proj, nn.Linear) and cm0.up_proj.out_features == 170
    assert isinstance(cm0.down_proj, nn.Linear) and cm0.down_proj.in_features == 170
    print("  [PASS]")
    PASS_COUNT += 1

    # === [7] Generation smoke (greedy + temperature) ===
    print("\n[7] Generation smoke (greedy + temperature):")
    m = MicroMixerV72GLU(v72_glu_1m())
    m.eval()
    prompt = torch.tensor([[1, 5, 10, 15, 20]], dtype=torch.long)
    # Greedy (temperature=0): deterministic, EOS disabled.
    out_greedy = m.generate(prompt, max_new_tokens=20, temperature=0.0, eos_token_id=-1)
    assert out_greedy.shape == (1, 25), f"greedy shape {out_greedy.shape}"
    assert torch.equal(out_greedy[:, :5], prompt), "prompt corrupted"
    assert (out_greedy[:, 5:] >= 0).all() and (out_greedy[:, 5:] < V).all(), "bad byte range"
    # Temperature sample: non-deterministic shape only.
    out_sample = m.generate(prompt, max_new_tokens=16, temperature=1.0, eos_token_id=-1)
    assert out_sample.shape == (1, 21), f"sample shape {out_sample.shape}"
    assert torch.equal(out_sample[:, :5], prompt), "sample prompt corrupted"
    print(f"  greedy: out shape {tuple(out_greedy.shape)}, prompt preserved")
    print(f"  T=1.0:  out shape {tuple(out_sample.shape)}, prompt preserved")
    print("  [PASS]")
    PASS_COUNT += 1

    # === [8] Inherited silent-init: W_o=0 and pool_up=0 at init
    #         (GLC chassis base property preserved by V72) ===
    print("\n[8] Inherited silent-init (W_o=0 and pool_up=0 at init):")
    m = MicroMixerV72GLU(v72_glu_1m())
    all_wo_zero = all(
        isinstance(b.token_mix, type(b.token_mix)) and hasattr(b.token_mix, "W_o")
        and b.token_mix.W_o.weight.abs().max().item() == 0.0
        and b.token_mix.W_o.bias.abs().max().item() == 0.0
        for b in m.blocks
    )
    all_kappa_zero = all(b.remixer.kappa.abs().item() == 0.0 for b in m.blocks)
    all_pool_up_zero = all(
        b.remixer.pool_up.weight.abs().max().item() == 0.0
        and b.remixer.pool_up.bias.abs().max().item() == 0.0
        for b in m.blocks
    )
    print(f"  W_o=0: {all_wo_zero}; kappa=0: {all_kappa_zero}; pool_up=0: {all_pool_up_zero}")
    assert all_wo_zero, "GLC W_o not zero at init"
    assert all_kappa_zero, "kappa not zero at init"
    assert all_pool_up_zero, "pool_up not zero at init"
    print("  [PASS]")
    PASS_COUNT += 1

    # === [9] SwiGLU grad flow: gate_proj / up_proj / down_proj all receive
    #         non-zero grad after one backward ===
    print("\n[9] SwiGLU grad flow (gate/up/down all non-zero after one backward):")
    torch.manual_seed(42)
    m = MicroMixerV72GLU(v72_glu_1m())
    m.train(); m.zero_grad()
    m(ids, None, None, None, targets)[1].backward()
    gate_grads = [b.channel_mix.gate_proj.weight.grad.abs().max().item()
                  for b in m.blocks]
    up_grads = [b.channel_mix.up_proj.weight.grad.abs().max().item()
                for b in m.blocks]
    down_grads = [b.channel_mix.down_proj.weight.grad.abs().max().item()
                  for b in m.blocks]
    print(f"  gate_proj grads:  {[f'{g:.2e}' for g in gate_grads]}")
    print(f"  up_proj grads:    {[f'{g:.2e}' for g in up_grads]}")
    print(f"  down_proj grads:  {[f'{g:.2e}' for g in down_grads]}")
    assert max(gate_grads) > 0, "gate_proj grad zero!"
    assert max(up_grads) > 0, "up_proj grad zero!"
    assert max(down_grads) > 0, "down_proj grad zero!"
    print("  [PASS]")
    PASS_COUNT += 1

    # === [10] GLC long-conv causality (S=256, perturb at 130/200/255) ===
    print("\n[10] GLC long-conv causality at every lag (S=256 > glc_kernel=129):")
    m = MicroMixerV72GLU(v72_glu_1m())
    m.eval()
    S_t = 256
    ids_a = torch.randint(4, V, (1, S_t))
    perturb_positions = [130, 200, 255]
    max_diff = 0.0
    for p in perturb_positions:
        ids_b = ids_a.clone()
        ids_b[0, p] = (ids_b[0, p] + 1) % V
        with torch.no_grad():
            la = m(ids_a, None, None, None, None)[0]
            lb = m(ids_b, None, None, None, None)[0]
        d = (la[:, :p] - lb[:, :p]).abs().max().item()
        max_diff = max(max_diff, d)
        print(f"  perturb pos {p:>3d}/{S_t}: max past-logit diff (positions < {p}) = {d:.2e}")
    assert max_diff == 0.0, f"GLC long-conv causality violated: {max_diff}"
    print(f"  overall max past-logit diff across perturbations = {max_diff:.2e}")
    print("  [PASS]")
    PASS_COUNT += 1

    # === Final summary ===
    print("\n" + "=" * 72)
    print(f"V72 GLU-Mixer self-test: {PASS_COUNT}/{TOTAL} PASS")
    print("=" * 72)
    if PASS_COUNT == TOTAL:
        print("ALL V72 SELF-TESTS PASSED")
    else:
        print(f"FAIL: {TOTAL - PASS_COUNT} test(s) did not pass")
        raise SystemExit(1)
    print("\nFinal param counts:")
    for name, fn in [("v72_glu_100k", v72_glu_100k), ("v72_glu_1m", v72_glu_1m)]:
        cfg = fn()
        n = count_parameters(MicroMixerV72GLU(cfg))
        print(f"  {name:<14s}: {n:>10,} params  "
              f"(d={cfg.d_model}, blocks={cfg.num_blocks}, tmix='{cfg.tmix}')")
