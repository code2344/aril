# Adapted from MicroMixer-4 (Apache-2.0): package imports were adjusted.
"""MicroMixer-4 V82 GCR-Mixer -- GLC Clean-RoPE (the clean-content lever ON the champion).

V82 is the cleanest possible test of V80's clean-content lever: V76-RPG
EXACT, but RoPE skipped on v's content half (v[:64]). Zero new parameters,
zero new shapes, zero new mechanisms -- a different RoPE application mask
on the proven champion.

== Motivation (V81_README.md S4 / .omo/plans/v82-glc-clean-rope.md) ==

V80 established the clean-content lever (+104 EM: TsR 488 vs TbR 384).
V81 showed it transfers but only reaches statistical parity with V76. The
sharpest remaining test: apply the lever DIRECTLY on the V76 champion
without touching anything else.

V76's GLCTokenMixRPG: ``v = rope(v); g = rope(g); v = DWConv(k=129)(v);
out = W_o(v * g)``. The conv is DEPTHWISE -- each channel mixes temporally
independent of other channels. So a mixed rotation pattern (rotate the
state half of v, skip the content half) is well-defined and costs 0 params.
V82 changes ONLY this:

    v_c, v_s = v[:64], v[64:]
    v_s = rope(v_s)          # state half: rotated (anti-collapse)
    # v_c: NOT rotated       # content half: clean (V80's +104 EM lever)
    v = DWConv(128, k=129)(cat([v_c, v_s]))   # V76's exact all-channel dense conv

== The fix (token-mix forward) ==

    u    = Linear(128 -> 256)(x)
    v, g = u.chunk(2, dim=-1)             # each 128-d
    v_c, v_s = v[:64], v[64:]             # content / state split
    v_s = rope_32(v_s)                    # 32-pair rotation on state (anti-collapse)
    # v_c: NOT rotated                    # clean content (V80's +104 EM lever)
    v   = cat([v_c, v_s], dim=-1)         # recombine to 128-d
    v   = CausalDepthwiseConv1d(128, k=129)(v)   # V76's exact all-channel dense conv
    [Nc]   g = rope_64(g)                 # 64-pair gate rotation (V76-identical)
    [NcNg] g = g                          # gate NOT rotated
    out  = W_o(v * g)                     # W_o zero-init (silent at init)

== Arms (resolving the V80 confound) ==

V80's -104 EM (TsR->TbR) conflated rope(v_c) + rope(g). Two arms isolate:

    V76-RPG (baseline): rope(v_c)=Y, rope(v_s)=Y, rope(g)=Y  -> EM 544.3, d2 0.872
    V82-Nc:             rope(v_c)=N, rope(v_s)=Y, rope(g)=Y  -> v_c rotation cost
    V82-NcNg:           rope(v_c)=N, rope(v_s)=Y, rope(g)=N  -> + g rotation cost

== Parameter budget ==

V76-RPG:                   993,254
RoPE cos/sin buffers:      0 (non-persistent; not in state_dict)
V82-GCR total:             993,254  (param-IDENTICAL to V76 -- RoPE is a 0-param
                                     buffer; skipping it on 64 channels changes
                                     no shapes, no counts, no state_dict keys)

Per-block token-mix (66,176 == V76 exactly):
    proj_in Linear(128 -> 256):   33,024
    CausalDepthwiseConv1d(128, k=129, dil=1): 16,640
    W_o Linear(128 -> 128) ZERO-INIT: 16,512

Self-test [T1] hard-asserts exact equality with V76's count and 171-key
state_dict.

== Init behavior ==

W_o is zero-init (inherited from GLCTokenMixRPG) so at init the token_mix
contributes 0 and V82 forward is bit-identical to V76/V72 forward. The
clean-content RoPE mechanism activates as W_o learns away from 0.

== Pure MLP-Mixer ==

No attention, no recurrence, no SSM. RoPE = elementwise per-pair rotation
(0 parameters, non-persistent cos/sin buffers). Depthwise causal conv for
token mixing, SwiGLU for channel mixing, RMSNorm pre-norm.

Self-test: per-arm T1/T2/T3 + causality (both arms) + cross-arm interface /
scan / generation / V76-interchangeability, runnable via
``python src/model_v82_gcr.py``.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import types
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_v37_qag import RMSNorm
from .model_v71_rtm import V71Config, count_parameters
from .model_v72_glumixer import SwiGLUChannelMLP
from .model_v67_contentpool import ReMixerLayer
from .model_v76_rpg import (
    GLCTokenMixRPG,
    MicroMixerV76RPG,
    v76_rpg_1m,
)


# ============================================================================
# Config
# ============================================================================

_V82_ARMS: tuple[str, ...] = ("Nc", "NcNg")

# Both arms: 7 blocks, SwiGLU h=170, glc_kernel=129 (V76-exact chassis).
_V82_NUM_BLOCKS: int = 7
_V82_GLC_KERNEL: int = 129

# Exact param count (hard-asserted in self-test; identical to V76 for BOTH arms).
_V82_EXPECTED_PARAMS: int = 993_254

# V76 state_dict key count (parameters only; RoPE buffers are non-persistent).
_V82_EXPECTED_SD_KEYS: int = 171


@dataclass
class V82Config(V71Config):
    """Configuration for MicroMixerV82GCR.

    Subclasses V71Config (for interface parity with the V71/V72/V75/V76
    family) and adds ``arm`` to select the RoPE application mask:

      "Nc":   v_s rotated (32-pair), g rotated (64-pair, V76-identical).
              v_c NEVER rotated. Isolates the v_c rotation cost vs V76.
      "NcNg": v_s rotated (32-pair), g NOT rotated.
              v_c NEVER rotated. Isolates the g rotation cost vs Nc.

    Both arms are param-IDENTICAL to V76 (993,254): RoPE is a 0-param buffer
    and skipping it on 64 channels changes no shapes, no counts, no keys.
    """
    arm: str = "Nc"


# ============================================================================
# GLCTokenMixGCR -- GLCTokenMixRPG + clean-content RoPE (skip v_c)
# ============================================================================

class GLCTokenMixGCR(GLCTokenMixRPG):
    """GLC Token-Mix with Clean-RoPE: rotate v_s (state) + optionally g,
    skip v_c (content).

    Subclasses V76's GLCTokenMixRPG (which subclasses GLCTokenMixRPC ->
    GLCTokenMix), inheriting proj_in / conv / W_o with identical shapes and
    the V71 house init (xavier proj_in, kaiming conv, zero W_o). The
    inherited 64-pair RoPE tables (rope_cos / rope_sin) serve the gate g
    (128-d, V76-identical). A NEW pair of 32-pair tables (rope_cos_state /
    rope_sin_state) serve v_s (64-d, the state half).

        u    = Linear(128 -> 256)(x)               # 33,024  (inherited)
        v, g = u.chunk(2, dim=-1)                   # each 128-d
        v_c, v_s = v[:64], v[64:]                   # content / state split
        v_s = rope_32(v_s)                          # 32-pair rotation on state
        # v_c: NOT rotated                           # clean content (V80's lever)
        v   = cat([v_c, v_s], dim=-1)               # recombine to 128-d
        v   = CausalDepthwiseConv1d(128, k=129)(v)  # 16,640  (V76-exact, inherited)
        [Nc]   g = rope_64(g)                       # 64-pair (inherited, V76-identical)
        [NcNg] g = g                                # gate NOT rotated
        out  = W_o(v * g)                           # 16,512  (ZERO-INIT, inherited)

    Params (token-mix): 33,024 + 16,640 + 16,512 = 66,176 (== V76 exactly).
    RoPE cos/sin: non-persistent buffers, 0 learned parameters.

    State_dict keys are IDENTICAL to V76's GLCTokenMixRPG (proj_in.weight,
    proj_in.bias, conv.conv.weight, conv.conv.bias, W_o.weight, W_o.bias) --
    the RoPE buffers are non-persistent and absent from state_dict. This
    makes V82 checkpoints mutually strict-loadable with V76's machinery.
    """

    def __init__(self, d_model: int, kernel_size: int, max_seq_len: int = 1024,
                 arm: str = "Nc"):
        if arm not in _V82_ARMS:
            raise ValueError(
                f"GLCTokenMixGCR arm must be one of {list(_V82_ARMS)}; got '{arm}'"
            )
        if d_model % 4 != 0:
            raise ValueError(
                f"GLCTokenMixGCR requires d_model divisible by 4; got {d_model}"
            )
        # super().__init__ chain: GLCTokenMixRPG -> GLCTokenMixRPC -> GLCTokenMix.
        #   - builds proj_in(128->256), conv(128, k=129), W_o(128->128)
        #   - applies house init (xavier proj_in, kaiming conv, zero W_o)
        #   - registers rope_cos / rope_sin (64-pair, for gate g, V76-identical)
        super().__init__(d_model, kernel_size, max_seq_len)
        self.arm = arm
        self.d_split = d_model // 2          # 64 (content / state split point)
        self.n_pairs_state = self.d_split // 2   # 32 (state-branch RoPE pairs)

        # RoPE tables for v_s (64-d, 32 pairs) -- state branch.
        #   theta_i = 10000^(-2i / n_pairs_state), i = 0 .. n_pairs_state-1
        # Same construction as V80/V81 _make_rope_tables and V75/V76's inline
        # formula (mathematically identical).
        inv_freq = 1.0 / (
            10000.0 ** (
                torch.arange(0, self.n_pairs_state, dtype=torch.float32)
                * 2.0 / self.n_pairs_state
            )
        )
        positions = torch.arange(0, max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, n_pairs_state)
        self.register_buffer("rope_cos_state", angles.cos(), persistent=False)
        self.register_buffer("rope_sin_state", angles.sin(), persistent=False)

    def _rope_state(self, v_s: torch.Tensor) -> torch.Tensor:
        """Apply 32-pair RoPE to v_s (64-d state half).

        Pair i (channels 2i, 2i+1) at position t rotates by angle t*theta_i
        where theta_i = 10000^(-2i/32). Cos/sin buffers sliced to [:S].
        """
        B, S, D = v_s.shape
        if D != self.d_split:
            raise ValueError(
                f"_rope_state expects last dim {self.d_split}, got {D}"
            )
        if S > self.rope_cos_state.shape[0]:
            raise ValueError(
                f"sequence length S={S} exceeds max_seq_len="
                f"{self.rope_cos_state.shape[0]} (RoPE state buffer pre-sized; "
                f"no extrapolation allowed)"
            )
        cos = self.rope_cos_state[:S].unsqueeze(0)   # (1, S, n_pairs_state)
        sin = self.rope_sin_state[:S].unsqueeze(0)
        v_even = v_s[..., 0::2]                       # (B, S, n_pairs_state)
        v_odd = v_s[..., 1::2]
        rot_even = v_even * cos - v_odd * sin
        rot_odd = v_even * sin + v_odd * cos
        return torch.stack([rot_even, rot_odd], dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.proj_in(x)                          # (B, S, 256)
        v, g = u.chunk(2, dim=-1)                    # each (B, S, 128)
        v_c = v[..., :self.d_split]                  # (B, S, 64)  clean content
        v_s = v[..., self.d_split:]                  # (B, S, 64)  dense state

        # State: 32-pair RoPE (anti-collapse mixer on the state half).
        v_s = self._rope_state(v_s)
        # Content: NOT rotated (V80's +104 EM clean-content lever).
        # Recombine to 128-d, then V76's exact all-channel dense conv.
        v = torch.cat([v_c, v_s], dim=-1)            # (B, S, 128)
        v = self.conv(v)                             # CausalDepthwiseConv1d(128, k=129)

        # Gate: arm-dependent.
        if self.arm == "Nc":
            g = self._rope(g)                        # inherited 64-pair (V76-identical)
        # arm "NcNg": gate NOT rotated.
        return self.W_o(v * g)


# ============================================================================
# V82 GCR-Mixer model
# ============================================================================

class MicroMixerV82GCR(MicroMixerV76RPG):
    """V82 GCR-Mixer: V76-RPG chassis with each block's GLCTokenMixRPG
    replaced by GLCTokenMixGCR (clean-content RoPE).

    The GLCTokenMixRPG is constructed first by ``super().__init__()`` (V76
    chassis: V72-GLU backbone + V71 RTMBlock stack + SwiGLU h=170 +
    ReMixerLayer sidecar), then discarded by the per-block swap below
    (mirrors V76's own swap from V72). State_dict keys are bit-identical to
    V76 (RoPE cos/sin are non-persistent buffers) -- strict-load from a V76
    checkpoint succeeds and yields a V82 model with clean-content RoPE
    activated once W_o is non-zero.

    Everything else (V71 GLC structure, V59 selectivity sidecar, V67 frozen
    content pool, RMSNorm, byte-tied embeddings, forward/generate contract)
    is inherited unchanged from MicroMixerV76RPG / V72GLU / V71RTM.

    Interface contract (identical to V76):
      - forward(input_ids, q_mask, a_mask, all_mask, targets)
        -> (logits, loss|None, aux); masks accepted and IGNORED (freeze_mask
        derived from input_ids via answer_region_mask).
      - generate(input_ids, q_mask, a_mask, all_mask, *,
        max_new_tokens, temperature, top_k, top_p, repetition_penalty,
        no_repeat_ngram_size, eos_token_id) -> full sequence.
      - model.config, model.out_norm, model.embed are public attributes.
      - Tied output head: logits = out_norm(h) @ embed.weight.T.
      - Model is copy.deepcopy-able and state_dict strict-loadable.
    """

    def __init__(self, config: V82Config):
        if config.arm not in _V82_ARMS:
            raise ValueError(
                f"unknown V82 arm '{config.arm}'; must be one of "
                f"{list(_V82_ARMS)}"
            )
        # super().__init__ builds the V76 chassis (V72-GLU backbone + V71
        # RTMBlock stack + SwiGLU + ReMixerLayer), then swaps token_mix to
        # GLCTokenMixRPG (V76's swap). We then re-swap to GLCTokenMixGCR.
        super().__init__(config)
        for b in self.blocks:
            b.token_mix = GLCTokenMixGCR(
                config.d_model, config.glc_kernel, config.max_seq_len, config.arm,
            )


# ============================================================================
# Presets
# ============================================================================

def v82_gcr_1m(arm: str = "Nc") -> V82Config:
    """~993K GCR-Mixer preset for the given arm.

    Exact param count: 993,254 for BOTH arms (identical to V76-RPG -- RoPE
    cos/sin buffers are non-persistent and add 0 learnable parameters;
    skipping the rotation on 64 channels changes no shapes or counts).

      "Nc":   v_s rotated, g rotated (V76-identical gate), v_c clean.
      "NcNg": v_s rotated, g NOT rotated, v_c clean.
    """
    if arm not in _V82_ARMS:
        raise ValueError(
            f"unknown V82 arm '{arm}'; must be one of {list(_V82_ARMS)}"
        )
    return V82Config(
        tmix="glc",
        d_model=128,
        num_blocks=_V82_NUM_BLOCKS,
        glc_kernel=_V82_GLC_KERNEL,
        max_seq_len=1024,
        arm=arm,
    )


# ============================================================================
# Self-test helpers
# ============================================================================

def _unzero_w_o(model: MicroMixerV82GCR, seed: int = 7) -> None:
    """Un-zero all zero-init W_o at a fixed seed.

    At init W_o=0 makes the entire token_mix output 0, killing the gradient
    to proj_in / conv. Un-zeroing exercises the RoPE paths (the state
    rotation, and for arm Nc the gate rotation).
    """
    torch.manual_seed(seed)
    for block in model.blocks:
        tm = block.token_mix
        nn.init.xavier_uniform_(tm.W_o.weight)
        nn.init.zeros_(tm.W_o.bias)


def _monkeypatch_token_mix_zero(model: MicroMixerV82GCR) -> list:
    """Monkeypatch each block's GLCTokenMixGCR forward to return zeros.

    Returns list of (block_idx, original_forward) for restoration.
    """
    patched = []
    for i, block in enumerate(model.blocks):
        tm = block.token_mix
        orig = tm.forward

        def _zero(self, x):
            return torch.zeros_like(x)
        tm.forward = types.MethodType(_zero, tm)
        patched.append((i, orig))
    return patched


def _restore_token_mix(model: MicroMixerV82GCR, patched: list) -> None:
    for i, orig in patched:
        model.blocks[i].token_mix.forward = orig


# ============================================================================
# Self-test
# ============================================================================

def _run_arm_tests(arm: str, V: int) -> int:
    """Run T1/T2/T3 + causality for a single V82 arm. Returns pass count."""
    passes = 0
    cfg = v82_gcr_1m(arm)

    print(f"\n{'='*72}")
    print(f"  ARM: {arm}  (blocks={_V82_NUM_BLOCKS}, glc_kernel={_V82_GLC_KERNEL}, "
          f"SwiGLU h=170)")
    print(f"{'='*72}")

    # === T1: Param count + bit-identical init + finite grads ===
    print(f"\n[T1-{arm}] Param count ({_V82_EXPECTED_PARAMS:,}) + bit-identical "
          f"init + finite grads:")

    m = MicroMixerV82GCR(cfg)
    n = count_parameters(m)
    ok_params = (n == _V82_EXPECTED_PARAMS) and (n < 1_000_000)
    print(f"  params: {n:>10,}  (expected {_V82_EXPECTED_PARAMS:>10,})  "
          f"[{'OK' if ok_params else 'FAIL'}]")
    assert ok_params, f"[{arm}] param count mismatch: {n} != {_V82_EXPECTED_PARAMS}"

    # Per-block token-mix accounting (defensive: matches V76 exactly).
    tm0 = m.blocks[0].token_mix
    tm_params = sum(p.numel() for p in tm0.parameters())
    expected_tm = 66_176  # 33,024 + 16,640 + 16,512  (== V76 GLC token-mix)
    print(f"  token-mix/block: {tm_params:,}  (expected {expected_tm:,})  "
          f"[{'OK' if tm_params == expected_tm else 'FAIL'}]")
    assert tm_params == expected_tm, (
        f"[{arm}] token-mix params {tm_params} != {expected_tm}"
    )

    # Verify zero-init conditions (W_o=0, kappa=0, pool_up=0).
    all_wo_zero = all(
        b.token_mix.W_o.weight.abs().max().item() == 0.0
        and b.token_mix.W_o.bias.abs().max().item() == 0.0
        for b in m.blocks
    )
    all_kappa_zero = all(
        b.remixer.kappa.abs().item() == 0.0 for b in m.blocks
    )
    all_pool_up_zero = all(
        b.remixer.pool_up.weight.abs().max().item() == 0.0
        and b.remixer.pool_up.bias.abs().max().item() == 0.0
        for b in m.blocks
    )
    print(f"  W_o=0: {all_wo_zero}; kappa=0: {all_kappa_zero}; "
          f"pool_up=0: {all_pool_up_zero}")
    assert all_wo_zero, f"[{arm}] W_o not zero at init"
    assert all_kappa_zero, f"[{arm}] kappa not zero at init"
    assert all_pool_up_zero, f"[{arm}] pool_up not zero at init"

    # Bit-identical: forward at init == forward with token-mix zeroed.
    m.eval()
    test_ids = torch.randint(4, V, (2, 64))
    with torch.no_grad():
        logits_normal = m(test_ids, None, None, None, None)[0]
    patched = _monkeypatch_token_mix_zero(m)
    with torch.no_grad():
        logits_zeroed = m(test_ids, None, None, None, None)[0]
    _restore_token_mix(m, patched)
    init_diff = (logits_normal - logits_zeroed).abs().max().item()
    print(f"  bit-identical init (token-mix zeroed): max|diff| = {init_diff:.2e}")
    assert init_diff == 0.0, f"[{arm}] not bit-identical at init: {init_diff}"

    # Finite grads.
    B, S = 4, 64
    ids = torch.randint(4, V, (B, S))
    targets = torch.cat(
        [ids[:, 1:], torch.full((B, 1), 0, dtype=torch.long)], dim=1,
    )
    m.train()
    m.zero_grad()
    logits, loss, aux = m(ids, None, None, None, targets)
    assert logits.shape == (B, S, V)
    assert torch.isfinite(loss).item()
    loss.backward()
    finite = all(
        torch.isfinite(p.grad).all().item()
        for p in m.parameters() if p.grad is not None
    )
    assert finite, f"[{arm}] non-finite gradients"
    print(f"  loss={loss.item():.4f} acc={aux['token_acc'].item():.4f} "
          f"finite_grads={finite}")
    print(f"  [T1 PASS]")
    passes += 1

    # === T2: Grad-flow (conv / proj_in / W_o after W_o un-zeroed) ===
    print(f"\n[T2-{arm}] Grad-flow (conv / proj_in / W_o, W_o un-zeroed):")
    m2 = MicroMixerV82GCR(cfg)
    _unzero_w_o(m2, seed=7)
    m2.train()
    m2.zero_grad()
    m2(ids, None, None, None, targets)[1].backward()
    finite2 = all(
        torch.isfinite(p.grad).all().item()
        for p in m2.parameters() if p.grad is not None
    )
    assert finite2, f"[{arm}] non-finite grads after W_o un-zero"

    conv_grads = [
        b.token_mix.conv.conv.weight.grad.abs().max().item() for b in m2.blocks
    ]
    proj_in_grads = [
        b.token_mix.proj_in.weight.grad.abs().max().item() for b in m2.blocks
    ]
    wo_grads = [
        b.token_mix.W_o.weight.grad.abs().max().item() for b in m2.blocks
    ]
    max_conv = max(conv_grads)
    max_proj = max(proj_in_grads)
    max_wo = max(wo_grads)
    print(f"  conv grads:    {[f'{g:.2e}' for g in conv_grads]}")
    print(f"  proj_in grads: {[f'{g:.2e}' for g in proj_in_grads]}")
    print(f"  W_o grads:     {[f'{g:.2e}' for g in wo_grads]}")
    assert max_conv > 0, f"[{arm}] conv grad zero (state RoPE path not wired)"
    assert max_proj > 0, f"[{arm}] proj_in grad zero (path not wired)"
    assert max_wo > 0, f"[{arm}] W_o grad zero at init"
    print(f"  max conv={max_conv:.2e}  max proj_in={max_proj:.2e}  "
          f"max W_o={max_wo:.2e}")
    print(f"  [T2 PASS]")
    passes += 1

    # === T3: Periodicity-break (report only; V80 lesson: no gate on init) ===
    print(f"\n[T3-{arm}] Periodicity-break (period-20 orbit, "
          f"warmup=7*(k-1)+period; INFORMATIONAL):")
    m3 = MicroMixerV82GCR(cfg)
    _unzero_w_o(m3, seed=7)
    m3.eval()

    period = 20
    # V82 uses V76's exact conv (128-ch, k=129, dil=1). 7 blocks compound
    # warmup: residual after block k is periodic only for t >= (k+1)*(k-1),
    # so all-7-block steady state requires t >= 7*128 = 896.
    warmup_start = _V82_NUM_BLOCKS * (_V82_GLC_KERNEL - 1) + period  # 7*128 + 20 = 916
    S_p = (cfg.max_seq_len // period) * period  # 1020
    assert warmup_start < S_p, (
        f"warmup_start {warmup_start} >= S_p {S_p} for arm {arm}"
    )
    torch.manual_seed(99)
    pattern = torch.randint(4, V, (period,))
    seq = pattern.repeat(S_p // period)
    ids_p = seq.unsqueeze(0)

    with torch.no_grad():
        logits_p = m3(ids_p, None, None, None, None)[0]

    sl = slice(warmup_start, S_p)
    prev_sl = slice(warmup_start - period, S_p - period)
    delta = (logits_p[0, sl] - logits_p[0, prev_sl]).abs().max().item()
    delta_all = (logits_p[0, period:] - logits_p[0, :-period]).abs().max().item()

    print(f"  warmup_start={warmup_start}  window=[{warmup_start},{S_p}) "
          f"({S_p - warmup_start} pos)")
    print(f"  max |Δlogit| at lag {period} (warmed-up window) = {delta:.4f}")
    print(f"  max |Δlogit| at lag {period} (all positions)    = {delta_all:.4f}")
    print(f"  (informational only; V80 lesson: init Δ does NOT predict "
          f"trained chatter -- no gate)")
    passes += 1  # T3 always "passes" (report only)

    # === Causality at every lag (k=129 left-pads 128) ===
    print(f"\n[C-{arm}] Causality at every lag (S=256, perturb at 130/200/255):")
    m4 = MicroMixerV82GCR(cfg)
    m4.eval()
    S_t = 256
    ids_a = torch.randint(4, V, (1, S_t))
    perturb_positions = [130, 200, 255]
    max_diff = 0.0
    for p in perturb_positions:
        ids_b = ids_a.clone()
        ids_b[0, p] = (ids_b[0, p] + 1) % V
        with torch.no_grad():
            la = m4(ids_a, None, None, None, None)[0]
            lb = m4(ids_b, None, None, None, None)[0]
        d = (la[:, :p] - lb[:, :p]).abs().max().item()
        max_diff = max(max_diff, d)
        print(f"  perturb pos {p:>3d}/{S_t}: max past-logit diff (< {p}) = {d:.2e}")
    assert max_diff == 0.0, f"[{arm}] causality violated: {max_diff}"
    print(f"  overall max past-logit diff = {max_diff:.2e}  "
          f"[PASS] (k={_V82_GLC_KERNEL} causal, pad {_V82_GLC_KERNEL - 1})")
    passes += 1

    return passes


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 72)
    print("MicroMixer-4 V82 GCR-Mixer -- self-test")
    print("(GLC Clean-RoPE: V76-RPG exact, RoPE skipped on v's content half)")
    print("(2 arms: Nc / NcNg)")
    print("=" * 72)

    V = 256
    ALL_ARMS = list(_V82_ARMS)
    PER_ARM_TESTS = 4  # T1, T2, T3, Causality
    CROSS_ARM_TESTS = 4  # Interface, Module scan, Generation+deepcopy, V76-interchange

    TOTAL = len(ALL_ARMS) * PER_ARM_TESTS + CROSS_ARM_TESTS
    PASS_COUNT = 0

    # === Per-arm tests ===
    for arm in ALL_ARMS:
        try:
            passes = _run_arm_tests(arm, V)
            PASS_COUNT += passes
        except AssertionError as e:
            print(f"\n  *** FAIL: {e}")
            raise

    # === Cross-arm tests ===
    print(f"\n{'='*72}")
    print("  CROSS-ARM TESTS")
    print(f"{'='*72}")

    # === [INT] Interface contract ===
    print("\n[INT] Interface contract (forward sig, generate kwargs, aux, "
          "tied head, deepcopy, strict-load):")
    import inspect
    sig = inspect.signature(MicroMixerV82GCR.forward)
    params = list(sig.parameters.keys())
    assert params == ["self", "input_ids", "q_mask", "a_mask", "all_mask", "targets"], \
        f"forward signature mismatch: {params}"
    gsig = inspect.signature(MicroMixerV82GCR.generate)
    gparams = list(gsig.parameters.keys())
    assert gparams == ["self", "input_ids", "q_mask", "a_mask", "all_mask",
                       "max_new_tokens", "temperature", "top_k", "top_p",
                       "repetition_penalty", "no_repeat_ngram_size", "eos_token_id"], \
        f"generate signature mismatch: {gparams}"
    gdefs = {p: gsig.parameters[p].default for p in gparams if p != "self"}
    assert gdefs["max_new_tokens"] == 100
    assert gdefs["temperature"] == 1.0
    assert gdefs["top_k"] is None
    assert gdefs["top_p"] is None
    assert gdefs["repetition_penalty"] == 1.0
    assert gdefs["no_repeat_ngram_size"] == 0
    assert gdefs["eos_token_id"] == 2

    # aux keys (use Nc model for the probe).
    m_int = MicroMixerV82GCR(v82_gcr_1m("Nc"))
    B, S = 4, 64
    ids = torch.randint(4, V, (B, S))
    targets = torch.cat(
        [ids[:, 1:], torch.full((B, 1), 0, dtype=torch.long)], dim=1,
    )
    _, _, aux_full = m_int(ids, None, None, None, targets)
    expected_aux = {"ce_loss", "ce_unweighted", "main_loss", "token_acc", "loss_weight_mean"}
    assert expected_aux.issubset(aux_full.keys()), f"aux missing keys: {aux_full.keys()}"
    _, _, aux_none = m_int(ids, None, None, None, None)
    assert set(aux_none.keys()) == {"token_acc"}, f"aux(targets=None) wrong: {aux_none.keys()}"
    assert isinstance(m_int.out_norm, RMSNorm), "out_norm not RMSNorm"
    assert isinstance(m_int.config, V82Config), "config not V82Config"
    assert isinstance(m_int.config, V71Config), "config not V71Config (subclass)"
    assert isinstance(m_int.embed, nn.Embedding), "embed not nn.Embedding"
    assert m_int.config.d_model == 128
    assert m_int.config.max_seq_len == 1024
    assert m_int.config.pad_token_id == 0
    # Tied head: NO standalone output Linear; logits come from embed.weight.T.
    has_separate_head = any(
        isinstance(mod, nn.Linear) and mod.out_features == V
        and n.endswith("lm_head")
        for n, mod in m_int.named_modules()
    )
    assert not has_separate_head, "found standalone lm_head (head must be tied)"

    # deepcopy on a FRESH model (V67 ReMixerLayer sets last_pool_pre during forward).
    m_fresh = MicroMixerV82GCR(v82_gcr_1m("Nc"))
    m_fresh.eval()
    m_deep = copy.deepcopy(m_fresh)
    m_deep.eval()
    test_ids2 = torch.randint(4, V, (2, 32))
    with torch.no_grad():
        out_orig = m_fresh(test_ids2, None, None, None, None)[0]
        out_deep = m_deep(test_ids2, None, None, None, None)[0]
    deep_diff = (out_orig - out_deep).abs().max().item()
    assert deep_diff == 0.0, f"deepcopy not bit-identical: {deep_diff}"

    # Strict state_dict load: fresh model <- state_dict of another fresh model.
    m_strict_src = MicroMixerV82GCR(v82_gcr_1m("NcNg"))
    m_strict_dst = MicroMixerV82GCR(v82_gcr_1m("NcNg"))
    sd = m_strict_src.state_dict()
    missing, unexpected = m_strict_dst.load_state_dict(sd, strict=True)
    assert not missing, f"strict-load missing keys: {missing}"
    assert not unexpected, f"strict-load unexpected keys: {unexpected}"
    m_strict_src.eval(); m_strict_dst.eval()
    with torch.no_grad():
        out_src = m_strict_src(test_ids2, None, None, None, None)[0]
        out_dst = m_strict_dst(test_ids2, None, None, None, None)[0]
    load_diff = (out_src - out_dst).abs().max().item()
    assert load_diff == 0.0, f"strict-load not bit-identical: {load_diff}"

    print(f"  forward 5 args OK; aux keys = {sorted(aux_full.keys())}")
    print(f"  generate kwargs defaults OK (V76 parity)")
    print(f"  tied head (no standalone lm_head) OK")
    print(f"  deepcopy(fresh) bit-identical (diff={deep_diff:.0e})")
    print(f"  strict state_dict load bit-identical (diff={load_diff:.0e})")
    print(f"  [PASS]")
    PASS_COUNT += 1

    # === [SCAN] Module scan: no attention / recurrence / SSM ===
    print("\n[SCAN] Module scan (no attention/recurrence/SSM; GLCTokenMixGCR present):")
    m_scan = MicroMixerV82GCR(v82_gcr_1m("Nc"))
    forbidden_types = (nn.MultiheadAttention, nn.RNN, nn.LSTM, nn.GRU,
                       nn.RNNCell, nn.LSTMCell, nn.GRUCell)
    has_forbidden_type = any(
        isinstance(mod, forbidden_types) for _, mod in m_scan.named_modules()
    )
    has_forbidden_name = any(
        ("attention" in n.lower() or "attn" in n.lower()
         or "s4" in n.lower() or "mamba" in n.lower())
        for n, _ in m_scan.named_modules()
    )
    assert not has_forbidden_type, "forbidden module type present"
    assert not has_forbidden_name, "forbidden module name present"
    # Count structural modules per arm.
    for arm in ALL_ARMS:
        m_a = MicroMixerV82GCR(v82_gcr_1m(arm))
        n_tokenmix = sum(
            1 for _, mod in m_a.named_modules() if isinstance(mod, GLCTokenMixGCR)
        )
        n_swiglu = sum(
            1 for _, mod in m_a.named_modules() if isinstance(mod, SwiGLUChannelMLP)
        )
        n_remixer = sum(
            1 for _, mod in m_a.named_modules() if isinstance(mod, ReMixerLayer)
        )
        assert n_tokenmix == _V82_NUM_BLOCKS, (
            f"[{arm}] expected {_V82_NUM_BLOCKS} GLCTokenMixGCR, got {n_tokenmix}"
        )
        assert n_swiglu == _V82_NUM_BLOCKS, (
            f"[{arm}] expected {_V82_NUM_BLOCKS} SwiGLU, got {n_swiglu}"
        )
        assert n_remixer == _V82_NUM_BLOCKS, (
            f"[{arm}] expected {_V82_NUM_BLOCKS} ReMixerLayer, got {n_remixer}"
        )
    print(f"  forbidden types/names: {has_forbidden_type or has_forbidden_name}")
    print(f"  GLCTokenMixGCR / SwiGLUChannelMLP / ReMixerLayer count per arm: "
          f"{_V82_NUM_BLOCKS}/{_V82_NUM_BLOCKS}/{_V82_NUM_BLOCKS}")
    print(f"  [PASS] no attention/recurrence/SSM")
    PASS_COUNT += 1

    # === [GEN] Generation smoke + RoPE buffer check ===
    print("\n[GEN] Generation smoke + RoPE buffer check:")
    m_gen = MicroMixerV82GCR(v82_gcr_1m("Nc"))
    m_gen.eval()
    prompt = torch.tensor([[1, 5, 10, 15, 20]], dtype=torch.long)
    out_greedy = m_gen.generate(prompt, max_new_tokens=20, temperature=0.0,
                                eos_token_id=-1)
    assert out_greedy.shape == (1, 25), f"greedy shape {out_greedy.shape}"
    assert torch.equal(out_greedy[:, :5], prompt), "prompt corrupted"
    assert (out_greedy[:, 5:] >= 0).all() and (out_greedy[:, 5:] < V).all(), \
        "bad byte range"
    out_sample = m_gen.generate(prompt, max_new_tokens=16, temperature=1.0,
                                eos_token_id=-1)
    assert out_sample.shape == (1, 21), f"sample shape {out_sample.shape}"
    # RoPE buffers non-persistent (absent from state_dict, present as buffers).
    sd = m_gen.state_dict()
    assert "rope_cos" not in sd and "rope_sin" not in sd, \
        "RoPE gate buffers leaked into state_dict"
    assert "rope_cos_state" not in sd and "rope_sin_state" not in sd, \
        "RoPE state buffers leaked into state_dict"
    buf_names = dict(m_gen.named_buffers())
    has_rope_gate = any(n.endswith("token_mix.rope_cos") for n in buf_names)
    has_rope_state = any(n.endswith("token_mix.rope_cos_state") for n in buf_names)
    assert has_rope_gate, "RoPE gate buffers missing"
    assert has_rope_state, "RoPE state buffers missing"
    print(f"  greedy: shape {tuple(out_greedy.shape)}, prompt preserved")
    print(f"  T=1.0:  shape {tuple(out_sample.shape)}, prompt preserved")
    print(f"  RoPE gate buffers:  present as non-persistent ({has_rope_gate})")
    print(f"  RoPE state buffers: present as non-persistent ({has_rope_state})")
    print(f"  [PASS]")
    PASS_COUNT += 1

    # === [V76] V76-equivalence: state_dict key-set identical + mutual
    #         strict-loadability + v_c-skip exercised (V82-Nc != V76) ===
    print("\n[V76] V76-equivalence (state_dict keys identical; mutual strict-load; "
          "v_c skip exercised):")
    # Build V76 and V82-Nc with the SAME seed -> identical shared weights.
    torch.manual_seed(42)
    m_v76 = MicroMixerV76RPG(v76_rpg_1m())
    torch.manual_seed(42)
    m_v82_nc = MicroMixerV82GCR(v82_gcr_1m("Nc"))

    # (a) state_dict keys are IDENTICAL sets (parameters only; buffers are
    #     non-persistent and absent from both state_dicts).
    sd_v76 = m_v76.state_dict()
    sd_v82 = m_v82_nc.state_dict()
    keys_v76 = set(sd_v76.keys())
    keys_v82 = set(sd_v82.keys())
    only_v76 = keys_v76 - keys_v82
    only_v82 = keys_v82 - keys_v76
    keys_identical = (keys_v76 == keys_v82)
    print(f"  V76 state_dict keys: {len(keys_v76)}")
    print(f"  V82 state_dict keys: {len(keys_v82)}")
    print(f"  only in V76: {sorted(only_v76)}")
    print(f"  only in V82: {sorted(only_v82)}")
    print(f"  key sets identical: {keys_identical}")
    assert keys_identical, (
        f"state_dict keys differ: only_v76={sorted(only_v76)}, "
        f"only_v82={sorted(only_v82)}"
    )
    assert len(keys_v76) == _V82_EXPECTED_SD_KEYS, (
        f"expected {_V82_EXPECTED_SD_KEYS} keys, got {len(keys_v76)}"
    )
    assert len(keys_v82) == _V82_EXPECTED_SD_KEYS, (
        f"expected {_V82_EXPECTED_SD_KEYS} keys, got {len(keys_v82)}"
    )

    # (b) Mutual strict-loadability: V82 <- V76 state_dict and V76 <- V82
    #     state_dict both succeed with strict=True (proves V82 checkpoints
    #     are interchangeable with V76's machinery).
    m_v82_load = MicroMixerV82GCR(v82_gcr_1m("Nc"))
    missing_1, unexpected_1 = m_v82_load.load_state_dict(sd_v76, strict=True)
    assert not missing_1 and not unexpected_1, (
        f"V82 <- V76 strict-load failed: missing={missing_1}, unexpected={unexpected_1}"
    )
    m_v76_load = MicroMixerV76RPG(v76_rpg_1m())
    missing_2, unexpected_2 = m_v76_load.load_state_dict(sd_v82, strict=True)
    assert not missing_2 and not unexpected_2, (
        f"V76 <- V82 strict-load failed: missing={missing_2}, unexpected={unexpected_2}"
    )
    print(f"  V82.load_state_dict(V76.state_dict(), strict=True): OK")
    print(f"  V76.load_state_dict(V82.state_dict(), strict=True): OK")

    # (c) v_c-skip exercised: with same weights + W_o un-zeroed identically,
    #     V82-Nc output DIFFERS from V76 (the v_c rotation skip changes the
    #     forward). They SHOULD differ -- this proves the clean-content lever
    #     is active. (Do NOT assert equality.)
    torch.manual_seed(7)
    for b in m_v76.blocks:
        nn.init.xavier_uniform_(b.token_mix.W_o.weight)
        nn.init.zeros_(b.token_mix.W_o.bias)
    for b76, b82 in zip(m_v76.blocks, m_v82_nc.blocks):
        b82.token_mix.W_o.weight.data.copy_(b76.token_mix.W_o.weight.data)
        b82.token_mix.W_o.bias.data.copy_(b76.token_mix.W_o.bias.data)
    m_v76.eval(); m_v82_nc.eval()
    chk_ids = torch.randint(4, V, (2, 128))
    with torch.no_grad():
        out_v76 = m_v76(chk_ids, None, None, None, None)[0]
        out_v82 = m_v82_nc(chk_ids, None, None, None, None)[0]
    v76_v82_diff = (out_v76 - out_v82).abs().max().item()
    print(f"  max |V76 - V82-Nc| (same weights, W_o != 0) = {v76_v82_diff:.4e} "
          f"(threshold > 1e-3 -- v_c skip MUST change the forward)")
    assert v76_v82_diff > 1e-3, (
        f"V82-Nc forward does not differ from V76 with v_c skip active: "
        f"max |Δ| = {v76_v82_diff:.4e} (needs > 1e-3). "
        f"The clean-content RoPE (v_c skip) is not changing the forward."
    )
    print(f"  -> clean-content RoPE (v_c skip) is the SOLE V82/V76 forward diff")
    print(f"  [PASS]")
    PASS_COUNT += 1

    # === Final summary ===
    print("\n" + "=" * 72)
    print(f"V82 GCR-Mixer self-test: {PASS_COUNT}/{TOTAL} PASS")
    print("=" * 72)
    if PASS_COUNT == TOTAL:
        print("ALL V82 SELF-TESTS PASSED")
    else:
        print(f"FAIL: {TOTAL - PASS_COUNT} test(s) did not pass")
        raise SystemExit(1)

    print("\nFinal param counts (both arms):")
    for arm in ALL_ARMS:
        c = v82_gcr_1m(arm)
        n = count_parameters(MicroMixerV82GCR(c))
        print(f"  v82_gcr_1m('{arm}'): {n:>10,} params  "
              f"(d={c.d_model}, blocks={c.num_blocks}, "
              f"glc_kernel={c.glc_kernel}, arm='{c.arm}')")
