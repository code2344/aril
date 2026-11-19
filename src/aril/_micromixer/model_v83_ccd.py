# Adapted from MicroMixer-4 (Apache-2.0): package imports were adjusted.
"""MicroMixer-4 V83 CCD-Mixer -- Content-Controlled Dilation (non-LTI token-mix).

V83 attacks the open question V82_README S4 hands to V83+: can the model
obtain position/timescale information WITHOUT scattering byte-identity (the
RoPE memorization tax), i.e. can the chatter<->EM conservation law be broken?
V80_README S2 names the root cause of collapse-proneness: dilated convs are
LINEAR TIME-INVARIANT (LTI) -- a fixed filter can soften a periodic orbit but
never break it; RoPE is currently the ONLY stationarity-breaker, and it taxes
memorization. V83 replaces the fixed-dilation conv with a CONTENT-CONTROLLED
mixture of dilations: the effective filter becomes TIME-VARYING (a function
of the bytes themselves), which is exactly the non-LTI mechanism the LTI
analysis calls for -- and it costs zero RoPE on content channels.

== Mechanism (token-mix forward) ==

    u    = Linear(128 -> 256)(x)
    v, g = u.chunk(2, dim=-1)                 # each 128-d
    [NcNg/NcNgA] v_c = v[:64] clean, v_s = rope_32(v[64:])   # V82-NcNg pattern
    [RPG]        v = rope_64(v), g = rope_64(g)              # V76 pattern
    # --- CCD: shared-weight multi-dilation causal conv, content-gated ---
    for d in (1, 2, 4, 8):
        y_d = F.conv1d(pad_(k-1)*d(v), W_conv, b_conv, dilation=d)  # SAME weights
    w(t) = softmax(Linear_dil(x)_t / tau)     # per-position 4-way gate (content)
    v    = sum_d w_d(t) * y_d(t)              # time-varying effective filter
    out  = W_o(v * g)                         # W_o zero-init (silent at init)

The four branches SHARE the k=129 depthwise kernel (0 extra conv params);
only the small gate (Linear 128->4) and a learnable temperature tau are new.
Dilation is no longer a hard-coded human prior: every byte picks its own
timescale, per block, per position. Because w(t) depends on content, the
composite operator is not LTI -- periodic orbits can be broken, not just
softened.

== Perfect ablation property ==

With ``force_dilation = 0`` the gate becomes one-hot on the dil=1 branch and
V83 is BIT-IDENTICAL to its RoPE-pattern parent (NcNg/NcNgA -> V82-NcNg,
RPG -> V76-RPG) with matched weights. Self-test [CCD] hard-asserts diff ==
0.0 in this mode and diff > 1e-3 with the gate free (mechanism active).

== Arms ==

    "NcNg":  CCD on the V82-NcNg RoPE pattern (v_s rotated, v_c clean,
             g clean). Tests whether CCD supplies anti-collapse on the
             minimal-rotation endpoint (EM champion side).
    "RPG":   CCD on the V76 full-RoPE pattern (v and g rotated). Tests
             whether CCD lifts the chatter champion past d2 >= 0.90 with
             zero cycles.
    "NcNgA": NcNg + Adaptive Orthogonality Angle (idea #9): the block's
             token-mix and channel-mix run in PARALLEL and are blended
             per-position,
                 a = token_mix(norm1(x)); b = channel_mix(norm2(x))
                 theta_t = pi/4 + angle_proj(norm2(x))   (zero-init -> pi/4)
                 x = x + cos(theta_t) * a + sin(theta_t) * b
             abolishing the "two mixes are orthogonal-sequential" prior.
             +903 params vs the other arms (angle_proj 129 x 7).

== Parameter budget ==

    V82-NcNg chassis:            993,254
    CCD gate per block:          Linear(128->4) 516 + log_tau 1 = 517  (x7 = 3,619)
    NcNg / RPG total:            996,873   (< 1M; state_dict keys 171 + 21 = 192)
    NcNgA total:                 997,776   (+ angle_proj 903; keys 206)

== Init behavior ==

W_o zero-init (inherited) makes the token-mix silent at init, so every arm's
forward is bit-identical to its token-mix-zeroed self (T1). The CCD gate is
zero-init -> uniform 1/4 mixture at step 0 (neutral over timescales; the
"no human prior" stance). log_tau = 0 -> tau = 1. Note: log_tau receives
zero gradient until dil_gate weights leave zero (d softmax(z/tau)/d tau = 0
at z = 0) -- expected, activates early in training.

== Pure MLP-Mixer ==

No attention, no recurrence, no SSM. Depthwise causal convs (left-pad only),
SwiGLU channel mixing, RMSNorm pre-norm, tied byte head. Causality is exact:
every branch left-pads (k-1)*d and the gate is pointwise in position.

Self-test: per-arm T1/T2/T3 + causality (3 arms) + cross-arm interface /
scan / generation / CCD-equivalence, runnable via
``python src/model_v83_ccd.py``.
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
from .model_v71_rtm import RTMBlock, V71Config, count_parameters
from .model_v72_glumixer import SwiGLUChannelMLP
from .model_v67_contentpool import ReMixerLayer
from .model_v76_rpg import (
    GLCTokenMixRPG,
    MicroMixerV76RPG,
    v76_rpg_1m,
)
from .model_v82_gcr import (
    GLCTokenMixGCR,
    MicroMixerV82GCR,
    V82Config,
    v82_gcr_1m,
)


# ============================================================================
# Config
# ============================================================================

_V83_ARMS: tuple[str, ...] = ("NcNg", "RPG", "NcNgA")

# Arms whose block structure is the V71 sequential RTMBlock (NcNgA uses the
# parallel AOARTMBlock instead).
_V83_SEQ_ARMS: tuple[str, ...] = ("NcNg", "RPG")

_V83_NUM_BLOCKS: int = 7
_V83_GLC_KERNEL: int = 129
_V83_DILATIONS: tuple[int, ...] = (1, 2, 4, 8)

# Exact param counts (hard-asserted in self-test).
#   base chassis:                993,254
#   CCD per block:               dil_gate (128*4+4) + log_tau 1 = 517  (x7 = 3,619)
#   NcNgA adds angle_proj:       128+1 = 129 per block                  (x7 =   903)
_V83_EXPECTED_PARAMS: dict[str, int] = {
    "NcNg": 996_873,
    "RPG": 996_873,
    "NcNgA": 997_776,
}

# state_dict key counts (base 171 + 21 CCD keys; NcNgA + 14 angle_proj keys).
_V83_EXPECTED_SD_KEYS: dict[str, int] = {
    "NcNg": 192,
    "RPG": 192,
    "NcNgA": 206,
}

# Per-block token-mix params: V76's 66,176 + 517 CCD.
_V83_TMIX_PARAMS: int = 66_693


@dataclass
class V83Config(V71Config):
    """Configuration for MicroMixerV83CCD.

    Subclasses V71Config (interface parity with the V71->V82 family) and adds:

      arm:           "NcNg"  -- CCD on V82-NcNg RoPE pattern (v_c clean,
                                v_s 32-pair RoPE, g clean).
                     "RPG"   -- CCD on V76 full-RoPE pattern (v, g rotated).
                     "NcNgA" -- NcNg + Adaptive Orthogonality Angle block
                                (parallel token/channel mix, per-position
                                blend angle).
      ccd_dilations: candidate dilation set for the content-controlled
                     mixture. Default (1, 2, 4, 8); dil=1 MUST be present
                     for the [CCD] bit-equivalence ablation to be meaningful.
    """
    arm: str = "NcNg"
    ccd_dilations: tuple = (1, 2, 4, 8)


# ============================================================================
# GLCTokenMixCCD -- GLCTokenMixGCR + content-controlled dilation
# ============================================================================

class GLCTokenMixCCD(GLCTokenMixGCR):
    """GLC Token-Mix with Content-Controlled Dilation.

    Subclasses V82's GLCTokenMixGCR, inheriting proj_in (128->256), the
    CausalDepthwiseConv1d(128, k=129) whose WEIGHTS are reused at every
    candidate dilation, W_o (zero-init), the 64-pair gate RoPE tables
    (rope_cos/rope_sin) and the 32-pair state tables (rope_cos_state /
    rope_sin_state). Adds:

      dil_gate: Linear(d_model, len(dilations)) -- per-position softmax
                gate over dilation branches. ZERO-INIT (weight AND bias)
                -> uniform mixture at step 0.
      log_tau:  scalar parameter; tau = exp(log_tau) = 1 at init.

    Forward:

        u = proj_in(x); v, g = u.chunk(2, -1)
        [NcNg/NcNgA] v_c clean; v_s = rope_32(v_s); v = cat(v_c, v_s)
        [RPG]        v = rope_64(v); g = rope_64(g)
        xt = v.transpose(1, 2)
        y_d = F.conv1d(F.pad(xt, ((k-1)*d, 0)), W, b, dilation=d, groups=d_model)
        gate = softmax(dil_gate(x) / exp(log_tau))        # (B, S, K)
        v = sum_d gate_d * y_d                            # time-varying filter
        out = W_o(v * g)                                  # W_o zero-init

    ``force_dilation`` (int | None, default None): when set to an index i,
    the gate is replaced by a one-hot on branch i. With force_dilation=0
    (dil=1) and matched weights the module is BIT-IDENTICAL to its
    RoPE-pattern parent (V82-NcNg / V76-RPG) -- the perfect-ablation mode.

    Params added per block: 128*K + K (dil_gate) + 1 (log_tau) = 517 at K=4.
    Conv weights are SHARED across branches: 0 extra conv params.
    """

    def __init__(self, d_model: int, kernel_size: int, max_seq_len: int = 1024,
                 arm: str = "NcNg", dilations: tuple = _V83_DILATIONS):
        if arm not in _V83_ARMS:
            raise ValueError(
                f"GLCTokenMixCCD arm must be one of {list(_V83_ARMS)}; got '{arm}'"
            )
        if len(dilations) < 2 or dilations[0] != 1:
            raise ValueError(
                f"GLCTokenMixCCD requires dilations starting at 1 with >=2 "
                f"candidates; got {dilations}"
            )
        # GCR init builds proj_in/conv/W_o (house init), 64-pair gate tables
        # and 32-pair state tables. Its own arm validation only accepts V82
        # arms and its forward is never used here, so register as "NcNg" and
        # set the V83 arm below.
        super().__init__(d_model, kernel_size, max_seq_len, arm="NcNg")
        self.arm = arm
        self.dilations = tuple(int(d) for d in dilations)
        self.n_dil = len(self.dilations)

        self.dil_gate = nn.Linear(d_model, self.n_dil, bias=True)
        nn.init.zeros_(self.dil_gate.weight)
        nn.init.zeros_(self.dil_gate.bias)
        self.log_tau = nn.Parameter(torch.zeros(1))

        # Ablation hook: index -> one-hot gate on that branch (see class doc).
        self.force_dilation: int | None = None
        # === UNIFORM-1/4 CONTROL HOOK (paper1 mechanism-attribution control) ===
        # When True the gate is frozen at the uniform mixture ones(K)/K at every
        # position regardless of input. This is the LTI, receptive-field-preserved
        # control: all K dilations stay active (RF preserved at (k-1)*dil_max*N)
        # but the gate is input-INDEPENDENT (no content-dependent selection).
        # Analogous to force_dilation but uniform instead of one-hot. The gate
        # params (dil_gate, log_tau) are bypassed -> receive zero gradient ->
        # stay at zero-init, so a free gate with these weights is bit-identical
        # to forced uniform (softmax(0/1) = ones/K). Default False leaves all
        # existing V83/V87 behavior (and the self-test) untouched.
        self.force_uniform: bool = False

    def _ccd_conv(self, v: torch.Tensor, x_gate: torch.Tensor) -> torch.Tensor:
        """Content-controlled mixture of shared-weight dilated causal convs.

        v:      (B, S, D) value tensor (post RoPE pattern).
        x_gate: (B, S, D) gate-conditioning input (block input, post-norm1).
        Returns (B, S, D). Exact causal: every branch left-pads (k-1)*d;
        the gate is pointwise in position.
        """
        B, S, D = v.shape
        xt = v.transpose(1, 2)                              # (B, D, S)
        weight = self.conv.conv.weight                      # (D, 1, k) shared
        bias = self.conv.conv.bias
        groups = self.conv.conv.groups                      # D (depthwise)
        k = weight.shape[-1]
        branches = []
        for d in self.dilations:
            y = F.conv1d(
                F.pad(xt, ((k - 1) * d, 0)), weight, bias,
                stride=1, padding=0, dilation=d, groups=groups,
            )
            branches.append(y.transpose(1, 2))              # (B, S, D)
        Y = torch.stack(branches, dim=2)                    # (B, S, K, D)

        if self.force_dilation is not None:
            gate = torch.zeros(B, S, self.n_dil, device=v.device, dtype=v.dtype)
            gate[..., self.force_dilation] = 1.0
        elif self.force_uniform:
            # UNIFORM-1/4 CONTROL: input-independent uniform gate (LTI, RF preserved).
            gate = torch.full(
                (B, S, self.n_dil), 1.0 / self.n_dil,
                device=v.device, dtype=v.dtype,
            )
        else:
            tau = torch.exp(self.log_tau)
            gate = torch.softmax(self.dil_gate(x_gate) / tau, dim=-1)  # (B,S,K)

        return (Y * gate.unsqueeze(-1)).sum(dim=2)          # (B, S, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.proj_in(x)                                 # (B, S, 256)
        v, g = u.chunk(2, dim=-1)                           # each (B, S, 128)

        if self.arm == "RPG":
            # V76 pattern: full 64-pair rotation on v AND g.
            v = self._rope(v)
            g = self._rope(g)
        else:
            # V82-NcNg pattern: content half clean, state half 32-pair RoPE,
            # gate NOT rotated.
            v_c = v[..., :self.d_split]
            v_s = self._rope_state(v[..., self.d_split:])
            v = torch.cat([v_c, v_s], dim=-1)

        v = self._ccd_conv(v, x)
        return self.W_o(v * g)


# ============================================================================
# AOARTMBlock -- RTMBlock + Adaptive Orthogonality Angle (arm NcNgA)
# ============================================================================

class AOARTMBlock(RTMBlock):
    """RTMBlock where token-mix and channel-mix run in PARALLEL and are
    blended per-position by a content-dependent angle:

        a       = token_mix(norm1(x))
        h2      = norm2(x)
        b       = channel_mix(h2)
        theta_t = pi/4 + angle_proj(h2)          # (B, S, 1); zero-init -> pi/4
        x       = x + cos(theta_t) * a + sin(theta_t) * b
        x       = remixer(x, y_norm, freeze_mask)

    Abolishes the "token-mix THEN channel-mix, orthogonally sequenced" prior:
    every byte chooses its own point on the arc between the two mixes.

    Instances are created by CLASS-PROMOTION of the V76-chassis RTMBlocks in
    MicroMixerV83CCD.__init__ (``b.__class__ = AOARTMBlock`` + attach
    angle_proj), so the NcNgA arm shares bit-identical chassis weights with
    the NcNg arm at the same seed -- the angle is the SOLE structural
    difference between those arms. No custom __init__ is defined on purpose.
    """

    def forward(
        self,
        x: torch.Tensor,
        y_norm: torch.Tensor,
        freeze_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a = self.token_mix(self.norm1(x))
        h2 = self.norm2(x)
        b = self.channel_mix(h2)
        theta = math.pi / 4.0 + self.angle_proj(h2)         # (B, S, 1)
        x = x + torch.cos(theta) * a + torch.sin(theta) * b
        return self.remixer(x, y_norm, freeze_mask)


# ============================================================================
# V83 CCD-Mixer model
# ============================================================================

class MicroMixerV83CCD(MicroMixerV82GCR):
    """V83 CCD-Mixer: V76/V82 chassis with each block's token-mix replaced by
    GLCTokenMixCCD (content-controlled dilation); arm NcNgA additionally
    promotes blocks to AOARTMBlock (parallel angle-blended mixes).

    Construction: MicroMixerV76RPG.__init__ is called DIRECTLY (bypassing
    MicroMixerV82GCR.__init__, whose arm validation only knows V82 arms),
    building the full V76 chassis (V72-GLU backbone + V71 RTMBlock stack +
    SwiGLU h=170 + ReMixerLayer sidecar + GLCTokenMixRPG swap). The RPG
    token-mixes are then discarded and replaced by GLCTokenMixCCD. For NcNgA,
    blocks are class-promoted to AOARTMBlock BEFORE the token-mix swap, so
    chassis weights stay seed-aligned with the other arms.

    Interface contract (identical to V71->V82):
      - forward(input_ids, q_mask, a_mask, all_mask, targets)
        -> (logits, loss|None, aux); masks accepted and IGNORED.
      - generate(input_ids, q_mask, a_mask, all_mask, *, max_new_tokens,
        temperature, top_k, top_p, repetition_penalty, no_repeat_ngram_size,
        eos_token_id) -> full sequence.
      - model.config, model.out_norm, model.embed are public attributes.
      - Tied output head: logits = out_norm(h) @ embed.weight.T.
      - Model is copy.deepcopy-able and state_dict strict-loadable.
      - V83-NcNg strict=False-loads a V82-NcNg checkpoint (missing keys are
        exactly the 21 CCD gate keys); V83-RPG likewise from V76.
    """

    def __init__(self, config: V83Config):
        if config.arm not in _V83_ARMS:
            raise ValueError(
                f"unknown V83 arm '{config.arm}'; must be one of "
                f"{list(_V83_ARMS)}"
            )
        if tuple(config.ccd_dilations)[0] != 1:
            raise ValueError(
                f"ccd_dilations must start at 1 (ablation anchor); got "
                f"{config.ccd_dilations}"
            )
        # Build the V76 chassis directly (V82's __init__ rejects V83 arms).
        MicroMixerV76RPG.__init__(self, config)

        if config.arm == "NcNgA":
            # Class-promote blocks BEFORE any further swaps so chassis
            # weights (norms, SwiGLU, remixer) stay seed-aligned with NcNg.
            for b in self.blocks:
                b.__class__ = AOARTMBlock
                b.angle_proj = nn.Linear(config.d_model, 1, bias=True)
                nn.init.zeros_(b.angle_proj.weight)
                nn.init.zeros_(b.angle_proj.bias)

        for b in self.blocks:
            b.token_mix = GLCTokenMixCCD(
                config.d_model, config.glc_kernel, config.max_seq_len,
                config.arm, tuple(config.ccd_dilations),
            )


# ============================================================================
# Presets
# ============================================================================

def v83_ccd_1m(arm: str = "NcNg") -> V83Config:
    """~997K CCD-Mixer preset for the given arm.

    Exact param counts (hard-asserted in self-test):
      "NcNg" / "RPG": 996,873  (993,254 chassis + 3,619 CCD gates)
      "NcNgA":        997,776  (+ 903 angle_proj)
    All arms < 1,000,000.
    """
    if arm not in _V83_ARMS:
        raise ValueError(
            f"unknown V83 arm '{arm}'; must be one of {list(_V83_ARMS)}"
        )
    return V83Config(
        tmix="glc",
        d_model=128,
        num_blocks=_V83_NUM_BLOCKS,
        glc_kernel=_V83_GLC_KERNEL,
        max_seq_len=1024,
        arm=arm,
    )


# ============================================================================
# Self-test helpers
# ============================================================================

def _unzero_w_o(model: MicroMixerV83CCD, seed: int = 7) -> None:
    """Un-zero all zero-init W_o at a fixed seed (exercises the CCD paths)."""
    torch.manual_seed(seed)
    for block in model.blocks:
        tm = block.token_mix
        nn.init.xavier_uniform_(tm.W_o.weight)
        nn.init.zeros_(tm.W_o.bias)


def _monkeypatch_token_mix_zero(model: MicroMixerV83CCD) -> list:
    """Monkeypatch each block's GLCTokenMixCCD forward to return zeros."""
    patched = []
    for i, block in enumerate(model.blocks):
        tm = block.token_mix
        orig = tm.forward

        def _zero(self, x):
            return torch.zeros_like(x)
        tm.forward = types.MethodType(_zero, tm)
        patched.append((i, orig))
    return patched


def _restore_token_mix(model: MicroMixerV83CCD, patched: list) -> None:
    for i, orig in patched:
        model.blocks[i].token_mix.forward = orig


def _set_force_dilation(model: MicroMixerV83CCD, idx: int | None) -> None:
    for b in model.blocks:
        b.token_mix.force_dilation = idx


# ============================================================================
# Self-test
# ============================================================================

def _run_arm_tests(arm: str, V: int) -> int:
    """Run T1/T2/T3 + causality for a single V83 arm. Returns pass count."""
    passes = 0
    cfg = v83_ccd_1m(arm)
    exp_params = _V83_EXPECTED_PARAMS[arm]
    is_aoa = (arm == "NcNgA")

    print(f"\n{'='*72}")
    print(f"  ARM: {arm}  (blocks={_V83_NUM_BLOCKS}, glc_kernel={_V83_GLC_KERNEL}, "
          f"dilations={_V83_DILATIONS}, SwiGLU h=170, AOA={is_aoa})")
    print(f"{'='*72}")

    # === T1: Param count + zero-init + bit-identical init + finite grads ===
    print(f"\n[T1-{arm}] Param count ({exp_params:,}) + zero-init + "
          f"bit-identical init + finite grads:")

    m = MicroMixerV83CCD(cfg)
    n = count_parameters(m)
    ok_params = (n == exp_params) and (n < 1_000_000)
    print(f"  params: {n:>10,}  (expected {exp_params:>10,})  "
          f"[{'OK' if ok_params else 'FAIL'}]")
    assert ok_params, f"[{arm}] param count mismatch: {n} != {exp_params}"

    # Per-block token-mix accounting.
    tm0 = m.blocks[0].token_mix
    tm_params = sum(p.numel() for p in tm0.parameters())
    print(f"  token-mix/block: {tm_params:,}  (expected {_V83_TMIX_PARAMS:,})  "
          f"[{'OK' if tm_params == _V83_TMIX_PARAMS else 'FAIL'}]")
    assert tm_params == _V83_TMIX_PARAMS, (
        f"[{arm}] token-mix params {tm_params} != {_V83_TMIX_PARAMS}"
    )

    # Zero-init conditions: W_o=0, kappa=0, pool_up=0, dil_gate=0, log_tau=0
    # (+ angle_proj=0 for NcNgA).
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
    all_gate_zero = all(
        b.token_mix.dil_gate.weight.abs().max().item() == 0.0
        and b.token_mix.dil_gate.bias.abs().max().item() == 0.0
        and b.token_mix.log_tau.abs().max().item() == 0.0
        for b in m.blocks
    )
    print(f"  W_o=0: {all_wo_zero}; kappa=0: {all_kappa_zero}; "
          f"pool_up=0: {all_pool_up_zero}; dil_gate=0 & log_tau=0: {all_gate_zero}")
    assert all_wo_zero, f"[{arm}] W_o not zero at init"
    assert all_kappa_zero, f"[{arm}] kappa not zero at init"
    assert all_pool_up_zero, f"[{arm}] pool_up not zero at init"
    assert all_gate_zero, f"[{arm}] CCD gate not zero-init"
    if is_aoa:
        all_angle_zero = all(
            b.angle_proj.weight.abs().max().item() == 0.0
            and b.angle_proj.bias.abs().max().item() == 0.0
            for b in m.blocks
        )
        print(f"  angle_proj=0: {all_angle_zero} (theta = pi/4 at init)")
        assert all_angle_zero, f"[{arm}] angle_proj not zero-init"

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

    # === T2: Grad-flow (conv / proj_in / W_o / dil_gate [+ angle_proj]) ===
    print(f"\n[T2-{arm}] Grad-flow (conv / proj_in / W_o / dil_gate"
          f"{'+ angle_proj' if is_aoa else ''}, W_o un-zeroed):")
    m2 = MicroMixerV83CCD(cfg)
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
    gate_grads = [
        b.token_mix.dil_gate.weight.grad.abs().max().item() for b in m2.blocks
    ]
    max_conv = max(conv_grads)
    max_proj = max(proj_in_grads)
    max_wo = max(wo_grads)
    max_gate = max(gate_grads)
    print(f"  conv grads:     {[f'{g:.2e}' for g in conv_grads]}")
    print(f"  proj_in grads:  {[f'{g:.2e}' for g in proj_in_grads]}")
    print(f"  W_o grads:      {[f'{g:.2e}' for g in wo_grads]}")
    print(f"  dil_gate grads: {[f'{g:.2e}' for g in gate_grads]}")
    assert max_conv > 0, f"[{arm}] conv grad zero (CCD conv path not wired)"
    assert max_proj > 0, f"[{arm}] proj_in grad zero (path not wired)"
    assert max_wo > 0, f"[{arm}] W_o grad zero at init"
    assert max_gate > 0, f"[{arm}] dil_gate grad zero (content gate not wired)"
    if is_aoa:
        angle_grads = [
            b.angle_proj.weight.grad.abs().max().item() for b in m2.blocks
        ]
        max_angle = max(angle_grads)
        print(f"  angle_proj grads: {[f'{g:.2e}' for g in angle_grads]}")
        assert max_angle > 0, f"[{arm}] angle_proj grad zero (AOA path not wired)"
    print(f"  max conv={max_conv:.2e}  max proj_in={max_proj:.2e}  "
          f"max W_o={max_wo:.2e}  max dil_gate={max_gate:.2e}")
    print(f"  (log_tau grad may be 0 while dil_gate logits are 0 -- expected)")
    print(f"  [T2 PASS]")
    passes += 1

    # === T3: Periodicity-break (report only; V80 lesson: no gate on init) ===
    print(f"\n[T3-{arm}] Periodicity-break (period-20 orbit, "
          f"warmup=7*(k-1)+period [dil-1 branch]; INFORMATIONAL):")
    m3 = MicroMixerV83CCD(cfg)
    _unzero_w_o(m3, seed=7)
    m3.eval()

    period = 20
    # Warmup computed on the dil=1 branch (V82 convention). CCD branches up
    # to dil=8 extend the potential receptive field to 1024/block; the
    # mixture's effective warmup is content-dependent -- one more reason
    # this probe is informational only.
    warmup_start = _V83_NUM_BLOCKS * (_V83_GLC_KERNEL - 1) + period  # 916
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
    print(f"  max |dlogit| at lag {period} (warmed-up window) = {delta:.4f}")
    print(f"  max |dlogit| at lag {period} (all positions)    = {delta_all:.4f}")
    print(f"  (informational only; V80 lesson: init delta does NOT predict "
          f"trained chatter -- no gate)")
    passes += 1  # T3 always "passes" (report only)

    # === Causality at every lag (all branches left-pad (k-1)*d) ===
    print(f"\n[C-{arm}] Causality at every lag (S=256, perturb at 130/200/255):")
    m4 = MicroMixerV83CCD(cfg)
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
          f"[PASS] (branches causal, gate pointwise)")
    passes += 1

    return passes


if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 72)
    print("MicroMixer-4 V83 CCD-Mixer -- self-test")
    print("(Content-Controlled Dilation: non-LTI token-mix, per-position "
          "dilation gate)")
    print("(3 arms: NcNg / RPG / NcNgA)")
    print("=" * 72)

    V = 256
    ALL_ARMS = list(_V83_ARMS)
    PER_ARM_TESTS = 4  # T1, T2, T3, Causality
    CROSS_ARM_TESTS = 4  # Interface, Module scan, Generation+deepcopy, CCD-equiv

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
    sig = inspect.signature(MicroMixerV83CCD.forward)
    params = list(sig.parameters.keys())
    assert params == ["self", "input_ids", "q_mask", "a_mask", "all_mask", "targets"], \
        f"forward signature mismatch: {params}"
    gsig = inspect.signature(MicroMixerV83CCD.generate)
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

    # aux keys (use NcNg model for the probe).
    m_int = MicroMixerV83CCD(v83_ccd_1m("NcNg"))
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
    assert isinstance(m_int.config, V83Config), "config not V83Config"
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

    # deepcopy on a FRESH NcNgA model (the trickiest: class-promoted blocks).
    m_fresh = MicroMixerV83CCD(v83_ccd_1m("NcNgA"))
    m_fresh.eval()
    m_deep = copy.deepcopy(m_fresh)
    m_deep.eval()
    test_ids2 = torch.randint(4, V, (2, 32))
    with torch.no_grad():
        out_orig = m_fresh(test_ids2, None, None, None, None)[0]
        out_deep = m_deep(test_ids2, None, None, None, None)[0]
    deep_diff = (out_orig - out_deep).abs().max().item()
    assert deep_diff == 0.0, f"deepcopy not bit-identical: {deep_diff}"

    # Strict state_dict round-trip on NcNgA (most keys).
    m_strict_src = MicroMixerV83CCD(v83_ccd_1m("NcNgA"))
    m_strict_dst = MicroMixerV83CCD(v83_ccd_1m("NcNgA"))
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

    # state_dict key counts per arm.
    for arm in ALL_ARMS:
        sd_arm = MicroMixerV83CCD(v83_ccd_1m(arm)).state_dict()
        assert len(sd_arm) == _V83_EXPECTED_SD_KEYS[arm], (
            f"[{arm}] expected {_V83_EXPECTED_SD_KEYS[arm]} state_dict keys, "
            f"got {len(sd_arm)}"
        )

    print(f"  forward 5 args OK; aux keys = {sorted(aux_full.keys())}")
    print(f"  generate kwargs defaults OK (V76 parity)")
    print(f"  tied head (no standalone lm_head) OK")
    print(f"  deepcopy(NcNgA, fresh) bit-identical (diff={deep_diff:.0e})")
    print(f"  strict state_dict round-trip bit-identical (diff={load_diff:.0e})")
    print(f"  state_dict keys per arm: "
          f"{ {a: _V83_EXPECTED_SD_KEYS[a] for a in ALL_ARMS} }")
    print(f"  [PASS]")
    PASS_COUNT += 1

    # === [SCAN] Module scan: no attention / recurrence / SSM ===
    print("\n[SCAN] Module scan (no attention/recurrence/SSM; GLCTokenMixCCD present):")
    m_scan = MicroMixerV83CCD(v83_ccd_1m("NcNg"))
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
        m_a = MicroMixerV83CCD(v83_ccd_1m(arm))
        n_tokenmix = sum(
            1 for _, mod in m_a.named_modules() if isinstance(mod, GLCTokenMixCCD)
        )
        n_swiglu = sum(
            1 for _, mod in m_a.named_modules() if isinstance(mod, SwiGLUChannelMLP)
        )
        n_remixer = sum(
            1 for _, mod in m_a.named_modules() if isinstance(mod, ReMixerLayer)
        )
        n_aoa = sum(
            1 for _, mod in m_a.named_modules() if isinstance(mod, AOARTMBlock)
        )
        assert n_tokenmix == _V83_NUM_BLOCKS, (
            f"[{arm}] expected {_V83_NUM_BLOCKS} GLCTokenMixCCD, got {n_tokenmix}"
        )
        assert n_swiglu == _V83_NUM_BLOCKS, (
            f"[{arm}] expected {_V83_NUM_BLOCKS} SwiGLU, got {n_swiglu}"
        )
        assert n_remixer == _V83_NUM_BLOCKS, (
            f"[{arm}] expected {_V83_NUM_BLOCKS} ReMixerLayer, got {n_remixer}"
        )
        exp_aoa = _V83_NUM_BLOCKS if arm == "NcNgA" else 0
        assert n_aoa == exp_aoa, (
            f"[{arm}] expected {exp_aoa} AOARTMBlock, got {n_aoa}"
        )
    print(f"  forbidden types/names: {has_forbidden_type or has_forbidden_name}")
    print(f"  GLCTokenMixCCD / SwiGLU / ReMixerLayer per arm: "
          f"{_V83_NUM_BLOCKS}/{_V83_NUM_BLOCKS}/{_V83_NUM_BLOCKS}; "
          f"AOARTMBlock: 7 for NcNgA, 0 otherwise")
    print(f"  [PASS] no attention/recurrence/SSM")
    PASS_COUNT += 1

    # === [GEN] Generation smoke + RoPE buffer check ===
    print("\n[GEN] Generation smoke + RoPE buffer check:")
    m_gen = MicroMixerV83CCD(v83_ccd_1m("NcNg"))
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

    # === [CCD] Perfect-ablation equivalence + interop ===
    # (a) V83-NcNg <- V82-NcNg state_dict (strict=False): missing keys are
    #     EXACTLY the 21 CCD gate keys; no unexpected keys.
    # (b) With matched (un-zeroed) W_o and force_dilation=0 (one-hot dil=1),
    #     V83-NcNg forward is BIT-IDENTICAL to V82-NcNg forward; with the
    #     gate free (uniform at init) the forwards DIFFER by > 1e-3
    #     (mechanism active).
    # (c) Same for V83-RPG vs V76-RPG.
    print("\n[CCD] Perfect-ablation equivalence (force_dilation=0 => parent; "
          "free gate => mechanism active):")
    m_v82 = MicroMixerV82GCR(v82_gcr_1m("NcNg"))
    m_v83 = MicroMixerV83CCD(v83_ccd_1m("NcNg"))

    sd_v82 = m_v82.state_dict()
    sd_v83 = m_v83.state_dict()
    ccd_suffixes = ("dil_gate.weight", "dil_gate.bias", "log_tau")
    missing_83 = [k for k in sd_v82.keys() if k not in sd_v83]
    extra_83 = [k for k in sd_v83.keys() if k not in sd_v82]
    assert not missing_83, f"V82 keys missing from V83: {missing_83}"
    assert len(extra_83) == 21 and all(
        k.endswith(ccd_suffixes) for k in extra_83
    ), f"unexpected V83-only keys: {sorted(extra_83)}"
    print(f"  V83-NcNg superset of V82-NcNg keys: +21 CCD keys exactly [OK]")

    missing_l, unexpected_l = m_v83.load_state_dict(sd_v82, strict=False)
    assert not unexpected_l, f"unexpected keys loading V82 into V83: {unexpected_l}"
    assert len(missing_l) == 21, f"expected 21 missing keys, got {len(missing_l)}"
    print(f"  V83.load_state_dict(V82 sd, strict=False): missing=21 CCD keys [OK]")

    # Un-zero W_o IDENTICALLY in both models.
    torch.manual_seed(7)
    for b in m_v82.blocks:
        nn.init.xavier_uniform_(b.token_mix.W_o.weight)
        nn.init.zeros_(b.token_mix.W_o.bias)
    for b82, b83 in zip(m_v82.blocks, m_v83.blocks):
        b83.token_mix.W_o.weight.data.copy_(b82.token_mix.W_o.weight.data)
        b83.token_mix.W_o.bias.data.copy_(b82.token_mix.W_o.bias.data)
    m_v82.eval(); m_v83.eval()
    chk_ids = torch.randint(4, V, (2, 128))

    _set_force_dilation(m_v83, 0)
    with torch.no_grad():
        out_v82 = m_v82(chk_ids, None, None, None, None)[0]
        out_v83_f0 = m_v83(chk_ids, None, None, None, None)[0]
    equiv_diff = (out_v82 - out_v83_f0).abs().max().item()
    print(f"  max |V82-NcNg - V83-NcNg(force=0)| = {equiv_diff:.2e} "
          f"(must be 0.0 -- perfect ablation)")
    assert equiv_diff == 0.0, (
        f"V83-NcNg with force_dilation=0 is NOT bit-identical to V82-NcNg: "
        f"{equiv_diff}"
    )

    _set_force_dilation(m_v83, None)
    with torch.no_grad():
        out_v83_free = m_v83(chk_ids, None, None, None, None)[0]
    mech_diff = (out_v82 - out_v83_free).abs().max().item()
    print(f"  max |V82-NcNg - V83-NcNg(free gate)| = {mech_diff:.4e} "
          f"(must be > 1e-3 -- CCD mechanism active)")
    assert mech_diff > 1e-3, (
        f"CCD free-gate forward does not differ from parent: {mech_diff}"
    )

    # (c) RPG arm vs V76.
    m_v76 = MicroMixerV76RPG(v76_rpg_1m())
    m_v83_rpg = MicroMixerV83CCD(v83_ccd_1m("RPG"))
    sd_v76 = m_v76.state_dict()
    missing_r, unexpected_r = m_v83_rpg.load_state_dict(sd_v76, strict=False)
    assert not unexpected_r, f"unexpected keys loading V76 into V83-RPG: {unexpected_r}"
    assert len(missing_r) == 21, f"expected 21 missing keys, got {len(missing_r)}"
    torch.manual_seed(7)
    for b in m_v76.blocks:
        nn.init.xavier_uniform_(b.token_mix.W_o.weight)
        nn.init.zeros_(b.token_mix.W_o.bias)
    for b76, b83 in zip(m_v76.blocks, m_v83_rpg.blocks):
        b83.token_mix.W_o.weight.data.copy_(b76.token_mix.W_o.weight.data)
        b83.token_mix.W_o.bias.data.copy_(b76.token_mix.W_o.bias.data)
    m_v76.eval(); m_v83_rpg.eval()
    _set_force_dilation(m_v83_rpg, 0)
    with torch.no_grad():
        out_v76 = m_v76(chk_ids, None, None, None, None)[0]
        out_v83_rpg_f0 = m_v83_rpg(chk_ids, None, None, None, None)[0]
    equiv_diff_rpg = (out_v76 - out_v83_rpg_f0).abs().max().item()
    print(f"  max |V76-RPG - V83-RPG(force=0)| = {equiv_diff_rpg:.2e} "
          f"(must be 0.0 -- perfect ablation)")
    assert equiv_diff_rpg == 0.0, (
        f"V83-RPG with force_dilation=0 is NOT bit-identical to V76-RPG: "
        f"{equiv_diff_rpg}"
    )
    print(f"  [PASS]")
    PASS_COUNT += 1

    # === Final summary ===
    print("\n" + "=" * 72)
    print(f"V83 CCD-Mixer self-test: {PASS_COUNT}/{TOTAL} PASS")
    print("=" * 72)
    if PASS_COUNT == TOTAL:
        print("ALL V83 SELF-TESTS PASSED")
    else:
        print(f"FAIL: {TOTAL - PASS_COUNT} test(s) did not pass")
        raise SystemExit(1)

    print("\nFinal param counts (all arms):")
    for arm in ALL_ARMS:
        c = v83_ccd_1m(arm)
        n = count_parameters(MicroMixerV83CCD(c))
        print(f"  v83_ccd_1m('{arm}'): {n:>10,} params  "
              f"(d={c.d_model}, blocks={c.num_blocks}, "
              f"glc_kernel={c.glc_kernel}, dilations={tuple(c.ccd_dilations)}, "
              f"arm='{c.arm}')")
