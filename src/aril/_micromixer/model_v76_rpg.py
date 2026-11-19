# Adapted from MicroMixer-4 (Apache-2.0): package imports were adjusted.
"""MicroMixer-4 V76 RPG-Mixer -- RoPE-Conv value+gate (the loop-attractor fix).

V76 is the direct successor of V75-RPC. It is a single structural change
over V75-RPC: **the per-position RoPE rotation is applied to BOTH the conv
value stream `v` AND the GLU gate `g`** (pre-conv, pre-multiply). The
rotation adds **zero parameters** (cos/sin are non-persistent buffers
inherited from V75-RPC); state_dict keys are bit-identical to V72, preserving
FMSP strict-load compatibility.

== Motivation (design doc .omo/plans/v75-rpc-design.md §6 pivot ①,
   user decision 2026-07-21) ==

V75-RPC rotates only the conv value `v`. It improved chatter distinct2
0.798 -> 0.863, but loops persisted in a subset of prompts. The
user-approved strengthening is to also rotate the gate `g` with the
SAME RoPE tables. This makes the GLU product v*g explicitly
position-dependent on BOTH factors (not just one), which strengthens the
loop-attractor break: a candidate period-P orbit now requires BOTH
v_t ≡ v_{t-P} AND g_t ≡ g_{t-P}, which is still measure-zero under the
geometric RoPE frequency spread.

== The fix ==

In `GLCTokenMix.forward`, between the chunk and the conv:

    u    = Linear(128 -> 256)(x)
    v, g = u[..., :128], u[..., 128:]
    v    = rope_rotate(v)                  # V75
    g    = rope_rotate(g)                  # NEW in V76 -- the only change
    v    = CausalDepthwiseConv1d(k=129)(v)
    out  = W_o(v * g)                       # W_o zero-init (silent at init)

== Parameter budget ==

V72-GLU:                   993,254
RoPE cos/sin buffers:      0 (non-persistent; not in state_dict; INHERITED
                            from V75-RPC's GLCTokenMixRPC.__init__)
V76-RPG total:             993,254  (margin 6,746)

Self-test [1] hard-asserts exact equality with V72's count.

== Init behavior ==

W_o is zero-init (inherited from GLCTokenMix) so at init the token_mix
contributes 0 and V76 forward is bit-identical to V72 forward. The RoPE
mechanism activates as W_o learns away from 0. The periodicity-break
self-test [4] un-zeros W_o to a fixed seed to exercise the mechanism.

Self-test: 11/11 PASS (10 mirror V75's, plus [11] gate-rotation-active).
"""

from __future__ import annotations

import copy
import math
import os
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_v37_qag import RMSNorm
from .model_v71_rtm import (
    V71Config,
    GLCTokenMix,
    MicroMixerV71RTM,
    count_parameters,
)
from .model_v72_glumixer import (
    MicroMixerV72GLU,
    SwiGLUChannelMLP,
    swiglu_hidden,
    v72_glu_1m,
    v72_glu_100k,
)
from .model_v75_rpc import (
    GLCTokenMixRPC,
    MicroMixerV75RPC,
    v75_rpc_1m,
    v75_rpc_100k,
)


# ============================================================================
# GLCTokenMixRPG -- GLCTokenMixRPC + RoPE rotation on the gate stream too
# ============================================================================

class GLCTokenMixRPG(GLCTokenMixRPC):
    """GLCTokenMix with RoPE rotation on BOTH `v` AND `g` between chunk and
    conv / multiply.

        u    = Linear(d, 2d)(x)
        v, g = u.chunk(2, dim=-1)
        v    = _rope(v)                       # inherited from V75-RPC
        g    = _rope(g)                       # NEW in V76 -- per-position pair rotation on gate
        v    = CausalDepthwiseConv1d(d, k)(v)
        out  = W_o(v * g)                     # W_o zero-init (silent at init)

    RoPE: split a d-channel tensor into d/2 pairs (2i, 2i+1). Pair i at
    position t rotates by angle t*theta_i, theta_i = 10000^(-2i/(d/2)).
    Cos/sin tables are INHERITED from GLCTokenMixRPC.__init__ (registered
    with persistent=False there) -- 0 learned parameters and state_dict
    keys are identical to GLCTokenMix (V72), preserving FMSP strict-load
    compatibility.

    This subclass does NOT re-register any buffers and does NOT re-init
    any weights; it only overrides `forward` to add the gate rotation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.proj_in(x)
        v, g = u.chunk(2, dim=-1)
        v = self._rope(v)                     # V75-RPC: RoPE value rotation
        g = self._rope(g)                     # NEW in V76: RoPE gate rotation
        v = self.conv(v)
        return self.W_o(v * g)


# ============================================================================
# V76 RPG-Mixer model
# ============================================================================

class MicroMixerV76RPG(MicroMixerV72GLU):
    """V76 RPG-Mixer: V72-GLU backbone with each block's GLCTokenMix
    replaced by GLCTokenMixRPG (RoPE-conv value+gate).

    The GLCTokenMix is constructed first by `super().__init__()` (V71
    chassis + V72 SwiGLU swap), then discarded by the per-block swap
    below. State_dict keys are bit-identical to V72 (RoPE cos/sin are
    non-persistent buffers) -- strict-load from a V72 checkpoint succeeds
    and yields a V76 model with RoPE activated once W_o is non-zero.

    Everything else (V71 GLC structure, V59 selectivity sidecar, V67
    frozen content pool, RMSNorm, byte-tied embeddings, forward/generate
    contract) is inherited unchanged from MicroMixerV72GLU / V71RTM.
    """

    def __init__(self, config: V71Config):
        super().__init__(config)
        # Swap token_mix in every block (mirrors V72's channel_mix swap and
        # V75's token_mix swap). GLCTokenMixRPG inherits __init__ from
        # GLCTokenMixRPC, which applies the V71 house init to proj_in/conv
        # and zeroes W_o, and registers the RoPE cos/sin buffers as
        # non-persistent. No buffer re-registration or weight re-init is
        # performed by GLCTokenMixRPG itself.
        for b in self.blocks:
            b.token_mix = GLCTokenMixRPG(
                config.d_model, config.glc_kernel, config.max_seq_len
            )


# ============================================================================
# Presets
# ============================================================================

# V76 follows the V75 (and V72) preset pattern: only the 1M champion-budget
# preset and a small 100K smoke/debug arm. d_model and num_blocks match
# V72/V75 exactly; the only difference is the RoPE rotation in the token_mix
# (now on both v and g), which is parameter-free and preset-independent.

def v76_rpg_100k() -> V71Config:
    """~83K GLC+SwiGLU+RoPE(v,g) preset (d=48, 3 blocks, glc_kernel=129).

    Mirror of v75_rpc_100k / v72_glu_100k for smoke / debugging. Param
    count is identical to V72-100K / V75-100K (RoPE adds 0 params).
    """
    return V71Config(tmix="glc", d_model=48, num_blocks=3, max_seq_len=512)


def v76_rpg_1m() -> V71Config:
    """~993K GLC+SwiGLU+RoPE(v,g) preset (d=128, 7 blocks, glc_kernel=129).

    Exact param count: 993,254 (identical to V72-GLU-1M and V75-RPC-1M --
    RoPE cos/sin buffers are non-persistent and add 0 learnable parameters).
    """
    return V71Config(tmix="glc", d_model=128, num_blocks=7, max_seq_len=1024)


# ============================================================================
# Self-test (11 invariants, mirrors V75 discipline + new gate-rotation test)
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 72)
    print("MicroMixer-4 V76 RPG-Mixer -- self-test")
    print("(V72-GLU backbone + RoPE-conv value+gate rotation, single-change ablation)")
    print("=" * 72)

    PASS_COUNT = 0
    TOTAL = 11
    V = 256

    # === [1] Param budget: v76_rpg_1m EXACTLY 993,254 AND equal to V72-GLU
    #         AND equal to V75-RPC ===
    print("\n[1] Parameter budget (v76_rpg_1m EXACT 993,254 == V72-GLU == V75-RPC):")
    m_v76_1m = MicroMixerV76RPG(v76_rpg_1m())
    n_v76_1m = count_parameters(m_v76_1m)
    m_v72_1m = MicroMixerV72GLU(v72_glu_1m())
    n_v72_1m = count_parameters(m_v72_1m)
    m_v75_1m = MicroMixerV75RPC(v75_rpc_1m())
    n_v75_1m = count_parameters(m_v75_1m)
    m_v76_100k = MicroMixerV76RPG(v76_rpg_100k())
    n_v76_100k = count_parameters(m_v76_100k)
    expected_1m = 993_254
    ok_exact = (n_v76_1m == expected_1m)
    ok_equal_v72 = (n_v76_1m == n_v72_1m)
    ok_equal_v75 = (n_v76_1m == n_v75_1m)
    ok_under_1m = (n_v76_1m < 1_000_000)
    ok_100k = (n_v76_100k < 1_000_000)
    print(f"  v76_rpg_1m:  {n_v76_1m:>10,} (expected {expected_1m:>10,}) "
          f"[{'<1M' if ok_under_1m else 'OVER'}] [{'OK' if ok_exact else 'FAIL'}]")
    print(f"  v72_glu_1m:  {n_v72_1m:>10,} (reference)")
    print(f"  v75_rpc_1m:  {n_v75_1m:>10,} (reference)")
    print(f"  v76 == v72:  {ok_equal_v72}")
    print(f"  v76 == v75:  {ok_equal_v75}")
    print(f"  v76_rpg_100k: {n_v76_100k:>10,} [{'<1M' if ok_100k else 'OVER'}] "
          f"[{'OK' if ok_100k else 'FAIL'}]")
    assert ok_exact, f"v76_rpg_1m expected {expected_1m}, got {n_v76_1m}"
    assert ok_equal_v72, f"v76_rpg_1m ({n_v76_1m}) != v72_glu_1m ({n_v72_1m})"
    assert ok_equal_v75, f"v76_rpg_1m ({n_v76_1m}) != v75_rpc_1m ({n_v75_1m})"
    assert ok_under_1m, f"v76_rpg_1m >= 1M: {n_v76_1m}"
    assert ok_100k, f"v76_rpg_100k >= 1M: {n_v76_100k}"
    print("  [PASS]")
    PASS_COUNT += 1

    # === [2] Forward + backward, finite grads; RoPE path grad flows to
    #         proj_in / conv / W_o (gate rotation must not break grad flow) ===
    # Same un-zero-W_o trick as V75 [2]: at init W_o=0 kills the gradient to
    # proj_in/conv. We un-zero W_o (fixed seed) so all three modules receive
    # non-zero grad. W_o itself receives grad even at zero-init.
    print("\n[2] Forward + backward (v76_rpg_1m, B=4, S=64): RoPE grad flow:")
    cfg = v76_rpg_1m()
    m = MicroMixerV76RPG(cfg)
    torch.manual_seed(7)
    for b in m.blocks:
        nn.init.xavier_uniform_(b.token_mix.W_o.weight)
        nn.init.zeros_(b.token_mix.W_o.bias)
    B, S = 4, 64
    ids = torch.randint(4, V, (B, S))
    targets = torch.cat([ids[:, 1:], torch.full((B, 1), 0, dtype=torch.long)], dim=1)
    m.train()
    logits, loss, aux = m(ids, None, None, None, targets)
    assert logits.shape == (B, S, V)
    assert torch.isfinite(loss).item()
    loss.backward()
    finite = all(
        torch.isfinite(p.grad).all().item()
        for p in m.parameters() if p.grad is not None
    )
    assert finite, "non-finite gradients"
    # RoPE path grad flow: proj_in / conv / W_o of each block's token_mix
    # must receive non-zero grad (RoPE is parameter-free but passes grad;
    # the gate rotation is an additional RoPE call but still parameter-free
    # and must not break the chain rule).
    proj_in_grads = [
        b.token_mix.proj_in.weight.grad.abs().max().item() for b in m.blocks
    ]
    conv_grads = [
        b.token_mix.conv.conv.weight.grad.abs().max().item() for b in m.blocks
    ]
    wo_grads = [
        b.token_mix.W_o.weight.grad.abs().max().item() for b in m.blocks
    ]
    print(f"  loss={loss.item():.4f} acc={aux['token_acc'].item():.4f} "
          f"finite_grads={finite} logits={tuple(logits.shape)}")
    print(f"  proj_in grads: {[f'{g:.2e}' for g in proj_in_grads]}")
    print(f"  conv grads:    {[f'{g:.2e}' for g in conv_grads]}")
    print(f"  W_o grads:     {[f'{g:.2e}' for g in wo_grads]}")
    assert max(proj_in_grads) > 0, "proj_in grad zero (RoPE path not wired)!"
    assert max(conv_grads) > 0, "conv grad zero (RoPE path not wired)!"
    assert max(wo_grads) > 0, "W_o grad zero at init!"
    print("  [PASS]")
    PASS_COUNT += 1

    # === [3] Causality: perturbing future bytes changes no past logits
    #         (rotation is per-position pointwise on BOTH v and g, must
    #         not leak) ===
    print("\n[3] Causality (perturb last byte, check past logits):")
    m = MicroMixerV76RPG(v76_rpg_1m())
    m.eval()
    S_t = 64
    ids_a = torch.randint(4, V, (1, S_t))
    ids_b = ids_a.clone()
    ids_b[0, -1] = (ids_b[0, -1] + 1) % V
    with torch.no_grad():
        la = m(ids_a, None, None, None, None)[0]
        lb = m(ids_b, None, None, None, None)[0]
    diff = (la[:, :-1] - lb[:, :-1]).abs().max().item()
    print(f"  v76_rpg_1m (S={S_t}): past-logits diff = {diff:.2e}")
    assert diff == 0.0, f"causality violated: {diff}"
    print("  [PASS]")
    PASS_COUNT += 1

    # === [4] Periodicity break (the fix mechanism; copy V75's approach) ===
    # Same setup as V75 [4]: un-zero W_o (same seed in V72/V75/V76), copy
    # state V72 -> V75 -> V76 to isolate the RoPE rotation as the only
    # difference. Feed a periodic byte seq (period 20, length 1020 -- the
    # largest multiple of period that fits max_seq_len, since 7 blocks of
    # k=129 causal conv compound the warmup to t >= 7*128 = 896). Verify
    # V76 Δ > 0.1; print V75 and V72 Δ for the record (no assert on them).
    print("\n[4] Periodicity break (period-20 byte seq, len 1020):")
    torch.manual_seed(42)
    m_v76 = MicroMixerV76RPG(v76_rpg_1m())
    torch.manual_seed(42)
    m_v75 = MicroMixerV75RPC(v75_rpc_1m())
    torch.manual_seed(42)
    m_v72 = MicroMixerV72GLU(v72_glu_1m())
    # Copy ALL params from V72 -> V75 -> V76 -> identical state isolating RoPE.
    m_v75.load_state_dict(m_v72.state_dict())
    m_v76.load_state_dict(m_v72.state_dict())
    # Un-zero W_o in V72 (fixed seed), then copy into V75 / V76 to keep all
    # three models bit-identical at the W_o level.
    torch.manual_seed(7)
    for b in m_v72.blocks:
        nn.init.xavier_uniform_(b.token_mix.W_o.weight)
        nn.init.zeros_(b.token_mix.W_o.bias)
    for b72, b75, b76 in zip(m_v72.blocks, m_v75.blocks, m_v76.blocks):
        b75.token_mix.W_o.weight.data.copy_(b72.token_mix.W_o.weight.data)
        b75.token_mix.W_o.bias.data.copy_(b72.token_mix.W_o.bias.data)
        b76.token_mix.W_o.weight.data.copy_(b72.token_mix.W_o.weight.data)
        b76.token_mix.W_o.bias.data.copy_(b72.token_mix.W_o.bias.data)
    m_v76.eval(); m_v75.eval(); m_v72.eval()
    # Causal depthwise conv k=129: each block's conv needs k-1=128 positions
    # of VALID input to reach steady state. With 7 blocks, warmup COMPOUNDS:
    # the residual stream after block k is periodic only for t >= (k+1)*128.
    # After all 7 blocks the asymptotic regime starts at t >= 7*128 = 896.
    # We need S large enough that the warmed-up window has usable length:
    # S_p=1020 (= largest multiple of period that fits max_seq_len), so
    # warmup_start = 7*128 + 20 = 916 ensures both t and t-period are in
    # the steady-state regime.
    period = 20
    S_p = (cfg.max_seq_len // period) * period  # 1020 at max_seq_len=1024
    n_blocks = cfg.num_blocks  # 7
    kernel = cfg.glc_kernel    # 129
    warmup_start = n_blocks * (kernel - 1) + period  # 7*128 + 20 = 916
    torch.manual_seed(99)
    pattern = torch.randint(4, V, (period,))
    seq = pattern.repeat(S_p // period)
    ids_p = seq.unsqueeze(0)
    with torch.no_grad():
        logits_v76_p = m_v76(ids_p, None, None, None, None)[0]
        logits_v75_p = m_v75(ids_p, None, None, None, None)[0]
        logits_v72_p = m_v72(ids_p, None, None, None, None)[0]
    sl = slice(warmup_start, S_p)
    prev_sl = slice(warmup_start - period, S_p - period)
    diff_v76 = (logits_v76_p[0, sl] - logits_v76_p[0, prev_sl]).abs().max().item()
    diff_v75 = (logits_v75_p[0, sl] - logits_v75_p[0, prev_sl]).abs().max().item()
    diff_v72 = (logits_v72_p[0, sl] - logits_v72_p[0, prev_sl]).abs().max().item()
    # For the record: max over ALL lag-`period` positions (warmup included).
    diff_v76_all = (logits_v76_p[0, period:] - logits_v76_p[0, :-period]).abs().max().item()
    diff_v75_all = (logits_v75_p[0, period:] - logits_v75_p[0, :-period]).abs().max().item()
    diff_v72_all = (logits_v72_p[0, period:] - logits_v72_p[0, :-period]).abs().max().item()
    print(f"  warmed-up window t in [{warmup_start}, {S_p}) "
          f"({S_p - warmup_start} positions, post-7-block steady state):")
    print(f"    V76-RPG:  max |Δlogit| at lag {period} = {diff_v76:.4f} "
          f"(threshold > 0.1) [{'OK' if diff_v76 > 0.1 else 'FAIL'}]")
    print(f"    V75-RPC:  max |Δlogit| at lag {period} = {diff_v75:.4f} "
          f"(for record; V75 also breaks periodicity)")
    print(f"    V72-GLU:  max |Δlogit| at lag {period} = {diff_v72:.4f} "
          f"(for record; shift-equivariance -> ~0 in steady state)")
    print(f"  all positions (warmup transient included, for the record):")
    print(f"    V76-RPG:  {diff_v76_all:.4f}    V75-RPC: {diff_v75_all:.4f}    "
          f"V72-GLU: {diff_v72_all:.4f}")
    assert diff_v76 > 0.1, (
        f"V76 periodicity break FAILED (warmed-up window): "
        f"max |Δlogit| = {diff_v76:.4f} (needs > 0.1). "
        f"RoPE(v,g) is not breaking the period-{period} orbit."
    )
    # Sanity: V72 must be ~0 in the steady-state window (else the test does
    # not discriminate V76 from V72 -- some other op is breaking periodicity).
    assert diff_v72 < 1e-3, (
        f"V72 shows non-trivial Δ in steady-state window ({diff_v72:.4f}); "
        f"test does not discriminate V76 from V72."
    )
    print("  [PASS]")
    PASS_COUNT += 1

    # === [5] RoPE equivariance: verify the inherited _rope math directly
    #         (same checks as V75 [5]; both v and g rotations use the same
    #         _rope, so verifying it once suffices) ===
    print("\n[5] RoPE equivariance (_rope math, spot + shift checks; inherited):")
    tm = MicroMixerV76RPG(v76_rpg_1m()).blocks[0].token_mix  # GLCTokenMixRPG
    B_t, S_t, D = 1, 32, cfg.d_model
    torch.manual_seed(0)
    v = torch.randn(B_t, S_t, D)
    v_rot = tm._rope(v)  # (B, S, D) -- the SAME _rope used for both v and g

    # Build the per-pair R(t) operator from the same formula _rope uses.
    n_pairs = D // 2
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, n_pairs, dtype=torch.float32) * 2.0 / n_pairs)
    )

    # (a) Spot positions: _rope(v)[t] == R(t) · v[t]
    spot_diffs = []
    for t in [0, 5, 15, 31]:
        angles = float(t) * inv_freq
        cos = angles.cos()
        sin = angles.sin()
        v_t = v[0, t]
        v_even = v_t[0::2]
        v_odd = v_t[1::2]
        exp_even = v_even * cos - v_odd * sin
        exp_odd = v_even * sin + v_odd * cos
        expected_t = torch.stack([exp_even, exp_odd], dim=-1).flatten()
        actual_t = v_rot[0, t]
        d_spot = (expected_t - actual_t).abs().max().item()
        spot_diffs.append(d_spot)
    max_spot = max(spot_diffs)
    print(f"  (a) _rope(v)[t] == R(t)·v[t] for t in [0,5,15,31]: "
          f"max |diff| = {max_spot:.2e}")

    # (b) Shift equivariance: v_shift[t+P] = v[t] (zero-padded on the left).
    #     Then _rope(v_shift)[t+P] should equal R(P) · _rope(v)[t] for all t.
    P = 7
    v_shift = torch.cat([torch.zeros(B_t, P, D), v], dim=1)  # (B, S+P, D)
    v_shift_rot = tm._rope(v_shift)
    # R(P) · v_rot
    angles_P = float(P) * inv_freq
    cos_P = angles_P.cos()
    sin_P = angles_P.sin()
    v_rot_even = v_rot[..., 0::2]
    v_rot_odd = v_rot[..., 1::2]
    exp_even = v_rot_even * cos_P - v_rot_odd * sin_P
    exp_odd = v_rot_even * sin_P + v_rot_odd * cos_P
    expected = torch.stack([exp_even, exp_odd], dim=-1).flatten(-2)  # (B, S, D)
    actual = v_shift_rot[:, P:P + S_t]  # (B, S, D)
    shift_diff = (expected - actual).abs().max().item()
    print(f"  (b) R(P)·_rope(v)[t] == _rope(shifted v)[t+P] for P={P}: "
          f"max |diff| = {shift_diff:.2e}")
    assert max_spot < 1e-5, f"RoPE spot check failed: {max_spot}"
    assert shift_diff < 1e-5, f"RoPE shift equivariance failed: {shift_diff}"
    print("  [PASS]")
    PASS_COUNT += 1

    # === [6] Sidecar silence at init + W_o=0 -> RPG block ≡ V72 base at init
    #         (gate rotation silent when W_o=0) ===
    print("\n[6] Silent-init (W_o=0, kappa=0, pool_up=0): V76 ≡ V72 at init")
    torch.manual_seed(42)
    m_v76_init = MicroMixerV76RPG(v76_rpg_1m())
    torch.manual_seed(42)
    m_v72_init = MicroMixerV72GLU(v72_glu_1m())
    # Copy V72's params into V76 (state_dicts have identical keys).
    m_v76_init.load_state_dict(m_v72_init.state_dict())
    all_wo_zero = all(
        hasattr(b.token_mix, "W_o")
        and b.token_mix.W_o.weight.abs().max().item() == 0.0
        and b.token_mix.W_o.bias.abs().max().item() == 0.0
        for b in m_v76_init.blocks
    )
    all_kappa_zero = all(
        b.remixer.kappa.abs().item() == 0.0 for b in m_v76_init.blocks
    )
    all_pool_up_zero = all(
        b.remixer.pool_up.weight.abs().max().item() == 0.0
        and b.remixer.pool_up.bias.abs().max().item() == 0.0
        for b in m_v76_init.blocks
    )
    print(f"  W_o=0: {all_wo_zero}; kappa=0: {all_kappa_zero}; "
          f"pool_up=0: {all_pool_up_zero}")
    assert all_wo_zero, "GLC W_o not zero at init"
    assert all_kappa_zero, "kappa not zero at init"
    assert all_pool_up_zero, "pool_up not zero at init"
    # Bit-identical forward at init (W_o=0 makes RoPE silent -- both v and g
    # rotations are multiplied through to a zero W_o).
    m_v76_init.eval(); m_v72_init.eval()
    test_ids = torch.randint(4, V, (2, 64))
    with torch.no_grad():
        out_v76 = m_v76_init(test_ids, None, None, None, None)[0]
        out_v72 = m_v72_init(test_ids, None, None, None, None)[0]
    init_diff = (out_v76 - out_v72).abs().max().item()
    print(f"  max |V76 - V72| at init = {init_diff:.2e} "
          f"(W_o=0 -> RoPE silent -> V76 ≡ V72)")
    assert init_diff == 0.0, (
        f"V76 not bit-identical to V72 at init: {init_diff}"
    )
    print("  [PASS]")
    PASS_COUNT += 1

    # === [7] Generation smoke (greedy + temperature; EOS handling) ===
    print("\n[7] Generation smoke (greedy + temperature; EOS handling):")
    m = MicroMixerV76RPG(v76_rpg_1m())
    m.eval()
    prompt = torch.tensor([[1, 5, 10, 15, 20]], dtype=torch.long)
    out_greedy = m.generate(prompt, max_new_tokens=20, temperature=0.0,
                            eos_token_id=-1)
    assert out_greedy.shape == (1, 25), f"greedy shape {out_greedy.shape}"
    assert torch.equal(out_greedy[:, :5], prompt), "prompt corrupted"
    assert (out_greedy[:, 5:] >= 0).all() and (out_greedy[:, 5:] < V).all(), \
        "bad byte range"
    # Temperature sample shape only (non-deterministic).
    out_sample = m.generate(prompt, max_new_tokens=16, temperature=1.0,
                            eos_token_id=-1)
    assert out_sample.shape == (1, 21), f"sample shape {out_sample.shape}"
    assert torch.equal(out_sample[:, :5], prompt), "sample prompt corrupted"
    print(f"  greedy:  out shape {tuple(out_greedy.shape)}, prompt preserved")
    print(f"  T=1.0:   out shape {tuple(out_sample.shape)}, prompt preserved")
    print("  [PASS]")
    PASS_COUNT += 1

    # === [8] Module scan: no attention / recurrence / SSM ===
    print("\n[8] Module scan (no attention/recurrence/SSM):")
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
    n_rpg = sum(
        1 for _, mod in m.named_modules() if isinstance(mod, GLCTokenMixRPG)
    )
    n_swiglu = sum(
        1 for _, mod in m.named_modules() if isinstance(mod, SwiGLUChannelMLP)
    )
    print(f"  forbidden types/names present: {has_forbidden_type or has_forbidden_name}")
    print(f"  GLCTokenMixRPG count:  {n_rpg} (expect 7 in 1M preset)")
    print(f"  SwiGLUChannelMLP count: {n_swiglu} (expect 7 in 1M preset)")
    assert n_rpg == 7, f"expected 7 GLCTokenMixRPG blocks, got {n_rpg}"
    assert n_swiglu == 7, f"expected 7 SwiGLU blocks, got {n_swiglu}"
    print("  [PASS] no attention/recurrence/SSM")
    PASS_COUNT += 1

    # === [9] Interface contract (5-arg forward, aux keys, generate kwargs,
    #         deepcopy, out_norm present) ===
    print("\n[9] Interface contract (5-arg forward, aux keys, generate kwargs, deepcopy, out_norm):")
    import inspect
    sig = inspect.signature(MicroMixerV76RPG.forward)
    params = list(sig.parameters.keys())
    assert params == ["self", "input_ids", "q_mask", "a_mask", "all_mask", "targets"], \
        f"forward signature mismatch: {params}"
    gsig = inspect.signature(MicroMixerV76RPG.generate)
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
    assert m.config.d_model == 128
    assert m.config.max_seq_len == 1024
    assert m.config.pad_token_id == 0
    # deepcopy works on a FRESH model (V67 ReMixerLayer sets last_pool_pre
    # during forward; deepcopy contract applies to a model that has not yet
    # been called, as in V67 / V71 / V72 / V75.)
    m_fresh = MicroMixerV76RPG(v76_rpg_1m())
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

    # === [10] Buffer check: cos/sin non-persistent; state_dict keys
    #          identical to V72 (171 keys; FMSP strict-load compatibility) ===
    print("\n[10] Buffer check (cos/sin non-persistent; state_dict keys == V72):")
    sd_v76 = m_fresh.state_dict()
    sd_v72 = MicroMixerV72GLU(v72_glu_1m()).state_dict()
    keys_v76 = set(sd_v76.keys())
    keys_v72 = set(sd_v72.keys())
    only_v76 = keys_v76 - keys_v72
    only_v72 = keys_v72 - keys_v76
    # RoPE buffers must NOT be in the state_dict.
    assert "rope_cos" not in sd_v76 and "rope_sin" not in sd_v76, \
        "RoPE buffers leaked into state_dict (should be non-persistent)"
    # Probe: confirm the module HAS the buffers as named attributes.
    buf_names = dict(m_fresh.named_buffers())
    has_rope_cos = any(
        n.endswith("token_mix.rope_cos") for n in buf_names
    )
    has_rope_sin = any(
        n.endswith("token_mix.rope_sin") for n in buf_names
    )
    # State_dict keys must match V72 exactly (171 keys).
    keys_match = (keys_v76 == keys_v72)
    print(f"  V76 state_dict keys: {len(keys_v76)}")
    print(f"  V72 state_dict keys: {len(keys_v72)}")
    print(f"  only in V76: {sorted(only_v76)}")
    print(f"  only in V72: {sorted(only_v72)}")
    print(f"  rope_cos present as buffer: {has_rope_cos} "
          f"(persistent=False -> not in state_dict)")
    print(f"  rope_sin present as buffer: {has_rope_sin} "
          f"(persistent=False -> not in state_dict)")
    print(f"  state_dict keys identical to V72: {keys_match}")
    assert keys_match, (
        f"state_dict keys differ from V72: only_v76={sorted(only_v76)}, "
        f"only_v72={sorted(only_v72)}"
    )
    assert len(keys_v76) == 171, f"expected 171 keys, got {len(keys_v76)}"
    assert has_rope_cos and has_rope_sin, "RoPE buffers missing"
    # Round-trip: V72 checkpoint loads into V76 with strict=True.
    m_v76_load = MicroMixerV76RPG(v76_rpg_1m())
    m_v76_load.load_state_dict(sd_v72, strict=True)
    # Verify RoPE buffers survive (they are reconstructed by __init__, not
    # loaded from state_dict, so they remain populated).
    buf_after = dict(m_v76_load.named_buffers())
    rope_after = next(
        b for n, b in buf_after.items() if n.endswith("token_mix.rope_cos")
    )
    assert rope_after.shape == (cfg.max_seq_len, cfg.d_model // 2)
    assert torch.isfinite(rope_after).all(), "RoPE cos buffer corrupted after load"
    print(f"  strict-load V72 state_dict into V76: OK (RoPE buffers preserved)")
    print("  [PASS]")
    PASS_COUNT += 1

    # === [11] Gate rotation ACTIVE: V76 forward differs from V75 (gate
    #          rotation changes the logits), AND disabling gate rotation
    #          makes V76 bit-identical to V75 (v-path parity) ===
    print("\n[11] Gate rotation active (V76 != V75 with gate rotated; "
          "V76 == V75 with gate rotation disabled):")
    # Build V76 and V75 with identical initial weights (state_dicts have the
    # same keys, so a strict copy isolates the forward difference to the
    # gate rotation alone).
    torch.manual_seed(42)
    m_v76_chk = MicroMixerV76RPG(v76_rpg_1m())
    torch.manual_seed(42)
    m_v75_chk = MicroMixerV75RPC(v75_rpc_1m())
    # Copy V75's state into V76 (strict load works: keys are identical).
    m_v76_chk.load_state_dict(m_v75_chk.state_dict(), strict=True)
    # Un-zero W_o in V75 (fixed seed), then mirror into V76 so both models
    # see the same non-silent W_o at forward time (otherwise W_o=0 makes
    # both models' token_mix output zero and they would trivially match).
    torch.manual_seed(7)
    for b in m_v75_chk.blocks:
        nn.init.xavier_uniform_(b.token_mix.W_o.weight)
        nn.init.zeros_(b.token_mix.W_o.bias)
    for b75, b76 in zip(m_v75_chk.blocks, m_v76_chk.blocks):
        b76.token_mix.W_o.weight.data.copy_(b75.token_mix.W_o.weight.data)
        b76.token_mix.W_o.bias.data.copy_(b75.token_mix.W_o.bias.data)
    m_v76_chk.eval(); m_v75_chk.eval()
    # Sanity: W_o is now non-zero in both models.
    wo_nonzero = all(
        b.token_mix.W_o.weight.abs().max().item() > 0
        for b in m_v76_chk.blocks
    )
    assert wo_nonzero, "W_o still zero in V76 -- cannot exercise gate rotation"
    # Feed a non-trivial input.
    chk_ids = torch.randint(4, V, (2, 128))
    with torch.no_grad():
        out_v76 = m_v76_chk(chk_ids, None, None, None, None)[0]
        out_v75 = m_v75_chk(chk_ids, None, None, None, None)[0]
    # (a) Logits MUST differ -- proves the gate rotation is exercised.
    diff_active = (out_v76 - out_v75).abs().max().item()
    print(f"  (a) max |V76 - V75| (gate rotation ACTIVE, W_o != 0) = "
          f"{diff_active:.4e} (threshold > 1e-3)")
    assert diff_active > 1e-3, (
        f"V76 forward does not differ from V75 with gate rotation active: "
        f"max |Δ| = {diff_active:.4e} (needs > 1e-3). "
        f"The gate rotation is not changing the forward."
    )
    # (b) Disable gate rotation by monkeypatching each token_mix's forward
    #     to V75's (rotate v only, not g). Then V76 must be bit-identical
    #     to V75 -- proves v-path parity (the only V76/V75 difference is
    #     the g rotation).
    def _v75_like_forward(self, x: torch.Tensor) -> torch.Tensor:
        # V75's GLCTokenMixRPC.forward (rotate v only, not g).
        u = self.proj_in(x)
        v, g = u.chunk(2, dim=-1)
        v = self._rope(v)
        v = self.conv(v)
        return self.W_o(v * g)

    patched = []
    for b in m_v76_chk.blocks:
        # Bind the function as a method on this specific instance.
        b.token_mix.forward = types.MethodType(_v75_like_forward, b.token_mix)
        patched.append(b.token_mix)
    # Re-run V76 with gate rotation disabled.
    with torch.no_grad():
        out_v76_disabled = m_v76_chk(chk_ids, None, None, None, None)[0]
    diff_disabled = (out_v76_disabled - out_v75).abs().max().item()
    print(f"  (b) max |V76 - V75| (gate rotation DISABLED via monkeypatch) = "
          f"{diff_disabled:.2e} (expect 0 -- v-path parity)")
    assert diff_disabled == 0.0, (
        f"V76 forward still differs from V75 with gate rotation disabled: "
        f"max |Δ| = {diff_disabled:.2e}. "
        f"The v-path is not bit-identical to V75."
    )
    # Sanity: confirm one of the patched forwards is actually GLCTokenMixRPG
    # (the class -- only its forward method was swapped, not the class).
    assert isinstance(patched[0], GLCTokenMixRPG), (
        "patched token_mix is not a GLCTokenMixRPG instance -- monkeypatch "
        "did not target the right module"
    )
    print(f"  -> gate rotation is the SOLE V76/V75 forward difference")
    print("  [PASS]")
    PASS_COUNT += 1

    # === Final summary ===
    print("\n" + "=" * 72)
    print(f"V76 RPG-Mixer self-test: {PASS_COUNT}/{TOTAL} PASS")
    print("=" * 72)
    if PASS_COUNT == TOTAL:
        print("ALL V76 SELF-TESTS PASSED")
    else:
        print(f"FAIL: {TOTAL - PASS_COUNT} test(s) did not pass")
        raise SystemExit(1)
    print("\nFinal param counts:")
    for name, fn in [("v76_rpg_100k", v76_rpg_100k), ("v76_rpg_1m", v76_rpg_1m)]:
        c = fn()
        n = count_parameters(MicroMixerV76RPG(c))
        print(f"  {name:<14s}: {n:>10,} params  "
              f"(d={c.d_model}, blocks={c.num_blocks}, tmix='{c.tmix}')")
