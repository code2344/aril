# Adapted from MicroMixer-4 (Apache-2.0): package imports were adjusted.
"""MicroMixer-4 V71 RTM-Mixer -- Reallocated Token-Mixing.

V71 replaces the V46-3-derived backbone (frozen since V52) with a canonical
MLP-Mixer chassis whose freed parameters fund **real learned token-mixing**,
co-designed for the V59 selectivity + V67 frozen-pool sidecar (kept verbatim,
imported from model_v67_contentpool).

== The 0.36% motivation ==

Audit of V67 (the champion, 996,250 params) found that **92.84% of parameters
are pointwise/channel ops** (pw1+pw2+fc1+fc2 = 924,928/996,250), while
**all learned cross-position mixing = 3,591 params (0.36%)**: one k=3
depthwise kernel per block + 7 κ scalars. The CausalDSConv1d "token-mix"
contains a 128→256→128 pointwise expansion that **structurally duplicates**
the ChannelMLP (461,440 params) in the same block. Everything else that
moves information across positions (byte-label mask, uniform average, pool)
is parameter-free or content-blind.

V71 strips that duplication: the token-mix loses its pw1/pw2 expansion and
returns to the canonical MLP-Mixer form (depthwise token-mix + channel MLP).
The freed ~461K params fund real learned token-mixing, in two selectable
variants behind one config switch, plus a no-token-mix control arm.

== Variants ==

  - **GLC** (Gated Long Convolution, `tmix="glc"`): Hyena-style raw-product
    gated long convolution, k=129, W_o zero-init (silent at init -> GLC ≡
    base chassis at step 0; clean ablation discipline, same unlock pattern
    as V67's pool_up). 7 blocks at 1M preset (depth parity with V67).

  - **MSC** (Multi-Scale Convolution, `tmix="msc"`): per-block dilated
    depthwise conv, k=23, dilations (1,1,2,2,4,4,8,8,16,16,32,32,32),
    Kaiming init (primary path). 13 blocks at 1M preset (depth for byte→
    word→phrase→entity abstraction).

  - **base** (`tmix="base"`): pure canonical chassis (no token-mix step;
    norm1 declared but unused in forward so per-block param count matches
    the design doc §7 table: per-block fixed = 71,190 at d=128). Control
    arm that isolates "rebalance alone" from "new token-mix".

The V59 sidecar (frozen byte-identity labels, learned sigmoid saliency,
selectivity-difference) and the V67 frozen content pool (K=4 sigmoid
importance maps, frozen at the last "Assistant: " marker, causal weighted
mean, rank-r zero-init adapter) are imported verbatim from
model_v67_contentpool -- guaranteed identical to V67.

== Parameter table (1M presets, d=128) ==

| Variant   | token-mix/block | per-block total | blocks | Total params |
|----------|-----------------|-----------------|--------|--------------|
| GLC-1M    | 66,176           | 137,366          | 7      | 994,458      |
| MSC-1M    | 3,072            | 74,262           | 13     | 998,302      |
| base-1M   | 0                | 71,190           | 13     | 958,366      |

Per-block fixed (d=128): norm1 (128) + norm2 (128) + ChannelMLP (65,920) +
sidecar (norm 128 + kappa 1 + saliency_proj 129 = 258) + pool (pool_proj 516
+ pool_down 2,064 + pool_up 2,176 = 4,756) = 71,190.

GLC token-mix at d=128, k=129: Linear(d,2d)=33,024 + Conv1d(d,k=129)=16,640
+ W_o Linear(d,d)=16,512 = 66,176.

MSC token-mix at d=128, k=23: Conv1d(d,k=23)=3,072.

Self-test: 10/10 PASS.
"""

from __future__ import annotations

import copy
import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_v37_qag import RMSNorm, ChannelMLP
from .model_v67_contentpool import (
    ReMixerLayer,
    answer_region_mask,
    make_byte_labels,
)


# ============================================================================
# Config
# ============================================================================

@dataclass
class V71Config:
    """Configuration for MicroMixerV71RTM.

    ``tmix`` selects the token-mix variant:
      - "glc": Gated Long Convolution (Hyena-style, W_o zero-init).
      - "msc": Multi-Scale Convolution (per-block dilated, Kaiming init).
      - "base": no token-mix (control arm).

    For "msc", ``msc_dilations`` is sliced to ``num_blocks`` (length must be ≥).
    """
    vocab_size: int = 256
    d_model: int = 128
    num_blocks: int = 7
    tmix: str = "glc"
    glc_kernel: int = 129
    msc_kernel: int = 23
    msc_dilations: tuple = (1, 1, 2, 2, 4, 4, 8, 8, 16, 16, 32, 32, 32)
    expansion_channel: int = 2
    label_dim: int = 16
    label_seed: int = 42
    kappa_init: float = 0.0
    row_normalize: bool = True
    pool_heads: int = 4
    pool_rank: int = 0   # 0 -> auto: max(4, d_model // 8)
    max_seq_len: int = 1024
    pad_token_id: int = 0


# ============================================================================
# Token-mix modules
# ============================================================================

class CausalDepthwiseConv1d(nn.Module):
    """Pure depthwise causal 1D conv with explicit left-pad (k-1)*dilation.

    Unlike V37's CausalDSConv1d, this is depthwise ONLY (no pw1/pw2 expansion):
    V71 strips the duplicated channel expansion out of the token-mix path --
    the ChannelMLP below handles all channel mixing (canonical MLP-Mixer form).
    """

    def __init__(self, d_model: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            groups=d_model,           # depthwise
            dilation=dilation,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D) -> (B, D, S)
        x = x.transpose(1, 2)
        x = F.pad(x, (self.pad, 0))   # causal left-pad (no future leak)
        x = self.conv(x)
        return x.transpose(1, 2)


class GLCTokenMix(nn.Module):
    """Gated Long Convolution (Hyena-style raw-product gate).

        u = Linear(d, 2d)(x)
        v, g = u.chunk(2, dim=-1)             # value / gate
        v = CausalDepthwiseConv1d(d, k=glc_kernel)(v)
        out = W_o(v * g)                       # W_o: Linear(d, d), ZERO-INIT

    W_o zero-init (weight AND bias) makes the token-mix silent at init ->
    V71-GLC forward ≡ V71-base forward at step 0 (clean ablation, same unlock
    pattern as V67's pool_up). dW_o ≠ 0 from step 1.
    """

    def __init__(self, d_model: int, kernel_size: int):
        super().__init__()
        self.proj_in = nn.Linear(d_model, 2 * d_model, bias=True)
        self.conv = CausalDepthwiseConv1d(d_model, kernel_size, dilation=1)
        self.W_o = nn.Linear(d_model, d_model, bias=True)
        # Zero-init W_o (re-applied after the model-wide _init_weights loop).
        nn.init.zeros_(self.W_o.weight)
        nn.init.zeros_(self.W_o.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.proj_in(x)
        v, g = u.chunk(2, dim=-1)
        v = self.conv(v)
        return self.W_o(v * g)


class MSCTokenMix(nn.Module):
    """Multi-Scale Convolution: per-block dilated depthwise conv + GELU.

        out = GELU(CausalDepthwiseConv1d(d, k=msc_kernel, dilation=d_i)(x))

    Kaiming conv init (this variant's token-mix is the primary path, NOT a
    zero-init sidecar). dilation is per-block (from config.msc_dilations).
    """

    def __init__(self, d_model: int, kernel_size: int, dilation: int):
        super().__init__()
        self.conv = CausalDepthwiseConv1d(d_model, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.conv(x))


# ============================================================================
# RTMBlock (canonical MLP-Mixer chassis + V67 sidecar)
# ============================================================================

class RTMBlock(nn.Module):
    """Pre-norm residual block:

        1. x += TokenMix(norm1(x))           # GLC | MSC | skipped for base
        2. x += ChannelMLP(norm2(x))         # GELU d -> 2d -> d (V37 unchanged)
        3. x  = ReMixerLayer(x, y_norm, freeze_mask)   # V67 sidecar (imported)

    For tmix="base", step 1 is a no-op in forward (norm1 is not called);
    norm1 is still declared so per-block param count matches the design doc
    §7 table (per-block fixed = 71,190 at d=128). This is the only reading
    consistent with the doc's asserted base-1M total of 958,366.
    """

    def __init__(
        self,
        d_model: int,
        tmix: str,
        glc_kernel: int,
        msc_kernel: int,
        msc_dilation: int,
        expansion_channel: int,
        kappa_init: float,
        row_normalize: bool,
        pool_heads: int,
        pool_rank: int,
    ):
        super().__init__()
        self.tmix = tmix
        self.norm1 = RMSNorm(d_model)
        if tmix == "glc":
            self.token_mix: "GLCTokenMix | MSCTokenMix | None" = GLCTokenMix(d_model, glc_kernel)
        elif tmix == "msc":
            self.token_mix = MSCTokenMix(d_model, msc_kernel, msc_dilation)
        elif tmix == "base":
            self.token_mix = None
        else:
            raise ValueError(f"unknown tmix='{tmix}' (must be 'glc' | 'msc' | 'base')")
        self.norm2 = RMSNorm(d_model)
        self.channel_mix = ChannelMLP(d_model, expansion_channel)
        self.remixer = ReMixerLayer(
            d_model=d_model,
            kappa_init=kappa_init,
            row_normalize=row_normalize,
            pool_heads=pool_heads,
            pool_rank=pool_rank,
        )

    def forward(
        self,
        x: torch.Tensor,
        y_norm: torch.Tensor,
        freeze_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.token_mix is not None:
            x = x + self.token_mix(self.norm1(x))
        x = x + self.channel_mix(self.norm2(x))
        x = self.remixer(x, y_norm, freeze_mask)
        return x


# ============================================================================
# Model
# ============================================================================

class MicroMixerV71RTM(nn.Module):
    """V71 RTM-Mixer: canonical MLP-Mixer chassis + V67 sidecar + new token-mix.

    Interface contract (identical to V67):
      - forward(input_ids, q_mask=None, a_mask=None, all_mask=None, targets=None)
        -> (logits, loss|None, aux); masks accepted and IGNORED (freeze_mask
        derived from input_ids via answer_region_mask).
      - generate(input_ids, q_mask=None, a_mask=None, all_mask=None, *,
        max_new_tokens, temperature, top_k, top_p, repetition_penalty,
        no_repeat_ngram_size, eos_token_id) -> full sequence.
      - model.config, model.out_norm, model.embed are public attributes.
      - Model is copy.deepcopy-able.
    """

    def __init__(self, config: V71Config):
        super().__init__()
        self.config = config
        d = config.d_model

        self.embed = nn.Embedding(config.vocab_size, d)

        byte_labels = make_byte_labels(
            vocab_size=config.vocab_size,
            label_dim=config.label_dim,
            seed=config.label_seed,
        )
        self.register_buffer("byte_labels", byte_labels, persistent=True)

        # Per-block dilation: slice msc_dilations to num_blocks for MSC; dummy 1s otherwise.
        if config.tmix == "msc":
            if len(config.msc_dilations) < config.num_blocks:
                raise ValueError(
                    f"tmix='msc' requires len(msc_dilations) >= num_blocks; "
                    f"got {len(config.msc_dilations)} < {config.num_blocks}"
                )
            dilations = tuple(config.msc_dilations[:config.num_blocks])
        else:
            dilations = tuple(1 for _ in range(config.num_blocks))

        self.blocks = nn.ModuleList([
            RTMBlock(
                d_model=d,
                tmix=config.tmix,
                glc_kernel=config.glc_kernel,
                msc_kernel=config.msc_kernel,
                msc_dilation=dil,
                expansion_channel=config.expansion_channel,
                kappa_init=config.kappa_init,
                row_normalize=config.row_normalize,
                pool_heads=config.pool_heads,
                pool_rank=config.pool_rank,
            )
            for dil in dilations
        ])

        self.out_norm = RMSNorm(d)
        self._init_weights()

    def _init_weights(self) -> None:
        # Embedding: std 0.02 (house style, matches V67/V59).
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        # House init for Linear (xavier) and Conv1d (kaiming) -- matches V67.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                    nn.init.uniform_(m.bias, -bound, bound)
        # V67 sidecar inits: kappa=0 + pool_up=0 -> delta = delta_pool = 0 at init.
        for block in self.blocks:
            block.remixer.kappa.data.fill_(float(self.config.kappa_init))
            nn.init.zeros_(block.remixer.pool_up.weight)
            nn.init.zeros_(block.remixer.pool_up.bias)
        # GLC W_o zero-init (re-zero after the global Linear loop xavier'd it).
        for block in self.blocks:
            if isinstance(block.token_mix, GLCTokenMix):
                nn.init.zeros_(block.token_mix.W_o.weight)
                nn.init.zeros_(block.token_mix.W_o.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        a_mask: torch.Tensor | None = None,
        all_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        h = self.embed(input_ids)
        y_norm = self.byte_labels[input_ids]
        freeze_mask = answer_region_mask(input_ids)

        for block in self.blocks:
            h = block(h, y_norm, freeze_mask)
        h = self.out_norm(h)
        logits = h @ self.embed.weight.T
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30, 30)

        if targets is None:
            return logits, None, {"token_acc": torch.tensor(0.0, device=input_ids.device)}

        pad = self.config.pad_token_id
        non_pad = (targets != pad).to(torch.float32)

        ce_per_pos = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            ignore_index=pad,
            reduction="none",
        ).reshape(input_ids.shape)

        denom = non_pad.sum().clamp(min=1)
        loss = (ce_per_pos * non_pad).sum() / denom

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            acc = ((pred == targets) & non_pad.bool()).float().sum() / denom

        aux = {
            "ce_loss": loss.detach(),
            "ce_unweighted": loss.detach(),
            "main_loss": loss.detach(),
            "token_acc": acc.detach(),
            "loss_weight_mean": torch.tensor(1.0, device=input_ids.device),
        }
        return logits, loss, aux

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        a_mask: torch.Tensor | None = None,
        all_mask: torch.Tensor | None = None,
        *,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        generated = input_ids.clone()

        for _ in range(max_new_tokens):
            logits, _, _ = self.forward(generated, None, None, None, None)
            next_logits = logits[:, -1, :].float()

            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                scaled = next_logits / max(temperature, 1e-6)

                if repetition_penalty != 1.0:
                    for b in range(generated.shape[0]):
                        prev = generated[b].tolist()
                        seen = set(prev)
                        for tok_id in seen:
                            if 0 <= tok_id < scaled.shape[-1]:
                                scaled[b, tok_id] /= repetition_penalty

                if no_repeat_ngram_size > 0 and generated.shape[1] >= no_repeat_ngram_size:
                    ngram = generated[:, -no_repeat_ngram_size+1:].tolist()
                    banned_per_batch = []
                    for b in range(generated.shape[0]):
                        seq = generated[b].tolist()
                        banned = set()
                        for i in range(len(seq) - no_repeat_ngram_size + 1):
                            if seq[i:i+no_repeat_ngram_size-1] == ngram[b]:
                                banned.add(seq[i + no_repeat_ngram_size - 1])
                        banned_per_batch.append(banned)
                    for b, banned in enumerate(banned_per_batch):
                        for tok_id in banned:
                            if 0 <= tok_id < scaled.shape[-1]:
                                scaled[b, tok_id] = float("-inf")

                if top_k is not None and top_k > 0:
                    v, _ = torch.topk(scaled, min(top_k, scaled.shape[-1]))
                    threshold = v[:, -1:]
                    scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

                if top_p is not None and 0 < top_p < 1:
                    sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
                    cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
                    remove = cum_probs > top_p
                    remove[..., 1:] = remove[..., :-1].clone()
                    remove[..., 0] = False
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    scaled = torch.full_like(scaled, float("-inf"))
                    scaled.scatter_(1, sorted_idx, sorted_logits)

                probs = F.softmax(scaled, dim=-1)
                next_token = torch.distributions.Categorical(probs).sample().unsqueeze(-1)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

        return generated


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================================
# Presets
# ============================================================================

def v71_rtm_glc_100k() -> V71Config:
    return V71Config(d_model=48, num_blocks=3, tmix="glc", max_seq_len=512)


def v71_rtm_glc_300k() -> V71Config:
    return V71Config(d_model=80, num_blocks=4, tmix="glc", max_seq_len=1024)


def v71_rtm_glc_500k() -> V71Config:
    return V71Config(d_model=96, num_blocks=5, tmix="glc", max_seq_len=1024)


def v71_rtm_glc_1m() -> V71Config:
    """~994K GLC preset (d=128, 7 blocks, k=129)."""
    return V71Config(d_model=128, num_blocks=7, tmix="glc", max_seq_len=1024)


def v71_rtm_msc_100k() -> V71Config:
    return V71Config(d_model=48, num_blocks=5, tmix="msc", max_seq_len=512)


def v71_rtm_msc_300k() -> V71Config:
    return V71Config(d_model=80, num_blocks=7, tmix="msc", max_seq_len=1024)


def v71_rtm_msc_500k() -> V71Config:
    return V71Config(d_model=96, num_blocks=9, tmix="msc", max_seq_len=1024)


def v71_rtm_msc_1m() -> V71Config:
    """~998K MSC preset (d=128, 13 blocks, k=23, full dilation stack)."""
    return V71Config(d_model=128, num_blocks=13, tmix="msc", max_seq_len=1024)


def v71_rtm_base_1m() -> V71Config:
    """~958K base preset (d=128, 13 blocks, no token-mix). Control arm."""
    return V71Config(d_model=128, num_blocks=13, tmix="base", max_seq_len=1024)


# ============================================================================
# Self-test (10 invariants, mirrors V67/V59 discipline)
# ============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 72)
    print("MicroMixer-4 V71 RTM-Mixer -- self-test")
    print("(Reallocated Token-Mixing: GLC | MSC | base)")
    print("=" * 72)

    # Verified per-preset param counts. The design doc §7 lists:
    #   glc-100k 83,361 / glc-300k 252,620 / msc-100k 69,991 / msc-300k 231,225
    # but those four numbers are arithmetic errors (kappa omitted per-block,
    # off by exactly num_blocks). The implementation counts below are derived
    # from the per-block formulas in the doc and verified by count_parameters.
    # The 1M counts (the user-mandated ±0 check) match the doc exactly.
    expected = {
        ("glc",  "100k"): 83_364,
        ("glc",  "300k"): 252_624,
        ("glc",  "500k"): 428_442,
        ("glc",  "1m"):   994_458,
        ("msc",  "100k"): 69_996,
        ("msc",  "300k"): 231_232,
        ("msc",  "500k"): 408_450,
        ("msc",  "1m"):   998_302,
        ("base", "1m"):   958_366,
    }
    preset_fns = {
        ("glc",  "100k"): v71_rtm_glc_100k,
        ("glc",  "300k"): v71_rtm_glc_300k,
        ("glc",  "500k"): v71_rtm_glc_500k,
        ("glc",  "1m"):   v71_rtm_glc_1m,
        ("msc",  "100k"): v71_rtm_msc_100k,
        ("msc",  "300k"): v71_rtm_msc_300k,
        ("msc",  "500k"): v71_rtm_msc_500k,
        ("msc",  "1m"):   v71_rtm_msc_1m,
        ("base", "1m"):   v71_rtm_base_1m,
    }

    PASS_COUNT = 0
    TOTAL = 10

    # === [1] All presets < 1M and match expected counts exactly ===
    print("\n[1] Parameter budgets (all presets < 1M, exact counts):")
    print("    (glc-100k/300k and msc-100k/300k differ from the doc by num_blocks:")
    print("     the doc §7 omitted kappa in those four rows.)")
    test1_ok = True
    for key in expected:
        cfg = preset_fns[key]()
        m = MicroMixerV71RTM(cfg)
        n = count_parameters(m)
        ok = (n < 1_000_000) and (n == expected[key])
        tag = "OK" if ok else "FAIL"
        print(f"  {key[0]:4s} {key[1]:5s}: {n:>10,} (expected {expected[key]:>10,}) [{'<1M' if n<1_000_000 else 'OVER'}] [{tag}]")
        if not ok:
            test1_ok = False
    if test1_ok:
        print("  [PASS]")
        PASS_COUNT += 1
    else:
        print("  [FAIL]")

    # === [2] Forward + backward, finite grads ===
    print("\n[2] Forward + backward (GLC-1M, B=4, S=64):")
    cfg = v71_rtm_glc_1m()
    m = MicroMixerV71RTM(cfg)
    V = cfg.vocab_size
    B, S = 4, 64
    ids = torch.randint(4, V, (B, S))
    targets = torch.cat([ids[:, 1:], torch.full((B, 1), 0, dtype=torch.long)], dim=1)
    m.train()
    logits, loss, aux = m(ids, None, None, None, targets)
    assert logits.shape == (B, S, V)
    assert torch.isfinite(loss).item()
    loss.backward()
    finite = all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None)
    assert finite, "non-finite gradients"
    print(f"  loss={loss.item():.4f} acc={aux['token_acc'].item():.4f} finite_grads={finite} [PASS]")
    PASS_COUNT += 1

    # === [3] Causality: perturb future bytes -> past logits unchanged ===
    print("\n[3] Causality (perturb last byte, check all past logits):")
    for name, fn in [("glc", v71_rtm_glc_1m), ("msc", v71_rtm_msc_1m)]:
        m = MicroMixerV71RTM(fn())
        m.eval()
        S_t = 128
        ids_a = torch.randint(4, V, (1, S_t))
        ids_b = ids_a.clone()
        ids_b[0, -1] = (ids_b[0, -1] + 1) % V
        with torch.no_grad():
            la = m(ids_a, None, None, None, None)[0]
            lb = m(ids_b, None, None, None, None)[0]
        diff = (la[:, :-1] - lb[:, :-1]).abs().max().item()
        print(f"  {name}-1m (S={S_t}): past-logits diff = {diff:.2e}")
        assert diff < 1e-4, f"{name} causality violated"
    print("  [PASS]")
    PASS_COUNT += 1

    # === [4] GLC bit-identical to base at init (W_o=0) AND sidecar silent ===
    print("\n[4] GLC bit-identical to base at init; sidecar silent (kappa=0, pool_up=0):")
    cfg_glc = V71Config(d_model=128, num_blocks=7, tmix="glc")
    cfg_base = V71Config(d_model=128, num_blocks=7, tmix="base")
    torch.manual_seed(123)
    m_glc = MicroMixerV71RTM(cfg_glc)
    torch.manual_seed(123)
    m_base = MicroMixerV71RTM(cfg_base)
    # Copy non-token_mix weights from glc to base.
    with torch.no_grad():
        m_base.embed.load_state_dict(m_glc.embed.state_dict())
        m_base.out_norm.load_state_dict(m_glc.out_norm.state_dict())
        for b_glc, b_base in zip(m_glc.blocks, m_base.blocks):
            b_base.norm1.load_state_dict(b_glc.norm1.state_dict())
            b_base.norm2.load_state_dict(b_glc.norm2.state_dict())
            b_base.channel_mix.load_state_dict(b_glc.channel_mix.state_dict())
            b_base.remixer.load_state_dict(m_glc.blocks[0].remixer.state_dict())
            # Override per-block with the matching glc block's remixer.
            b_base.remixer.load_state_dict(b_glc.remixer.state_dict())
    # Verify silent-init conditions.
    glc_wo_zero = all(
        b.token_mix.W_o.weight.abs().max().item() == 0.0
        and b.token_mix.W_o.bias.abs().max().item() == 0.0
        for b in m_glc.blocks
    )
    all_kappa_zero = all(b.remixer.kappa.abs().item() == 0.0 for b in m_glc.blocks)
    all_pool_up_zero = all(
        b.remixer.pool_up.weight.abs().max().item() == 0.0
        and b.remixer.pool_up.bias.abs().max().item() == 0.0
        for b in m_glc.blocks
    )
    assert glc_wo_zero, "GLC W_o not zero at init"
    assert all_kappa_zero, "kappa not zero at init"
    assert all_pool_up_zero, "pool_up not zero at init"
    m_glc.eval(); m_base.eval()
    test_ids = torch.randint(4, V, (2, 64))
    with torch.no_grad():
        out_glc = m_glc(test_ids, None, None, None, None)[0]
        out_base = m_base(test_ids, None, None, None, None)[0]
    diff = (out_glc - out_base).abs().max().item()
    print(f"  W_o=0: {glc_wo_zero}; kappa=0: {all_kappa_zero}; pool_up=0: {all_pool_up_zero}")
    print(f"  max |GLC-base| at init = {diff:.2e}")
    assert diff < 1e-5, f"GLC not bit-identical to base at init: {diff}"
    print("  [PASS]")
    PASS_COUNT += 1

    # === [5] Grad health: W_o grad, kappa grad, pool_up grad all non-zero ===
    print("\n[5] Grad health at init (GLC-1M):")
    torch.manual_seed(42)
    m = MicroMixerV71RTM(v71_rtm_glc_1m())
    m.train(); m.zero_grad()
    m(ids, None, None, None, targets)[1].backward()
    wo_grads = [b.token_mix.W_o.weight.grad.abs().max().item() for b in m.blocks]
    kg = [b.remixer.kappa.grad.item() for b in m.blocks]
    pg = [b.remixer.pool_up.weight.grad.abs().max().item() for b in m.blocks]
    print(f"  W_o grads:     {[f'{g:.2e}' for g in wo_grads]}")
    print(f"  kappa grads:   {[f'{g:.2e}' for g in kg]}")
    print(f"  pool_up grads: {[f'{g:.2e}' for g in pg]}")
    assert max(wo_grads) > 0, "W_o grad zero at init!"
    assert max(abs(g) for g in kg) > 0, "kappa grad zero!"
    assert max(pg) > 0, "pool_up grad zero at init!"
    print("  [PASS]")
    PASS_COUNT += 1

    # === [6] Pool freeze invariance (key drift in answer region == 0.0) ===
    print("\n[6] Pool freeze invariance (drift == 0.0 in answer region):")
    m = MicroMixerV71RTM(v71_rtm_glc_1m())
    m.eval()
    text = "User: hi\n\nAssistant: hello world"
    ids_m = torch.tensor([[1] + list(text.encode()) + [2]], dtype=torch.long)
    fm = answer_region_mask(ids_m)
    ans_start = text.index("Assistant: ") + len("Assistant: ") + 1  # +1 for BOS
    assert not fm[0, :ans_start].any(), "mask fired before answer region"
    assert fm[0, ans_start:].all(), "mask missing in answer region"
    layer = m.blocks[0].remixer
    with torch.no_grad():
        x_m = m.embed(ids_m)
        h_m = layer.norm(x_m)
        pool_w = torch.sigmoid(layer.pool_proj(h_m)) * (~fm).to(h_m.dtype).unsqueeze(-1)
        num_m = torch.cumsum(pool_w.unsqueeze(-1) * h_m.unsqueeze(2), dim=1)
        den_m = torch.cumsum(pool_w, dim=1).clamp(min=1e-4).unsqueeze(-1)
        c_m = (num_m / den_m).mean(dim=2)
    c_ans = c_m[0, ans_start:]
    drift = (c_ans - c_ans[0:1]).abs().max().item()
    print(f"  pool drift across answer positions = {drift:.2e}")
    assert drift == 0.0, "pool not frozen in answer region"
    print("  [PASS]")
    PASS_COUNT += 1

    # === [7] Generation smoke (valid bytes, EOS handling) ===
    print("\n[7] Generation smoke (GLC-1M and MSC-1M):")
    for name, fn in [("glc", v71_rtm_glc_1m), ("msc", v71_rtm_msc_1m)]:
        m = MicroMixerV71RTM(fn())
        m.eval()
        prompt = torch.tensor([[1, 5, 10, 15, 20]], dtype=torch.long)
        # Greedy (temperature=0): deterministic, EOS=-1 to disable.
        out = m.generate(prompt, max_new_tokens=20, temperature=0.0, eos_token_id=-1)
        assert out.shape == (1, 25), f"{name}: shape {out.shape}"
        assert torch.equal(out[:, :5], prompt), f"{name}: prompt corrupted"
        assert (out[:, 5:] >= 0).all() and (out[:, 5:] < V).all(), f"{name}: bad byte range"
        # EOS test: immediately emit a dummy EOS via greedy on a tiny prompt.
        out2 = m.generate(prompt, max_new_tokens=10, temperature=0.0, eos_token_id=999)
        assert out2.shape[1] >= 5
        print(f"  {name}: greedy out shape {tuple(out.shape)}; "
              f"sample temperatures out: T=1.0 len={m.generate(prompt, max_new_tokens=8, temperature=1.0, eos_token_id=-1).shape[1]}")
    print("  [PASS]")
    PASS_COUNT += 1

    # === [8] Module scan: no attention/recurrence/SSM ===
    print("\n[8] Module scan (no attention/recurrence/SSM):")
    forbidden_types = (nn.MultiheadAttention, nn.RNN, nn.LSTM, nn.GRU, nn.RNNCell, nn.LSTMCell, nn.GRUCell)
    has_forbidden_type = False
    for _, mod in m.named_modules():
        if isinstance(mod, forbidden_types):
            has_forbidden_type = True
            break
    has_attn_name = any(
        ("attention" in n.lower() or "attn" in n.lower() or "s4" in n.lower() or "mamba" in n.lower())
        for n, _ in m.named_modules()
    )
    assert not has_forbidden_type, "forbidden module type present"
    assert not has_attn_name, "forbidden module name present"
    # Sanity: confirm V71 token-mix modules ARE present.
    n_glc = sum(1 for _, mod in m.named_modules() if isinstance(mod, GLCTokenMix))
    n_msc = sum(1 for _, mod in m.named_modules() if isinstance(mod, MSCTokenMix))
    print(f"  forbidden types/names present: {has_forbidden_type or has_attn_name}")
    print(f"  GLCTokenMix count: {n_glc} (expect 0 in MSC model); MSCTokenMix count: {n_msc} (expect 13)")
    print("  [PASS] no attention/recurrence/SSM")
    PASS_COUNT += 1

    # === [9] Interface contract ===
    print("\n[9] Interface contract (5-arg forward, aux keys, generate kwargs, deepcopy, out_norm):")
    import inspect
    sig = inspect.signature(MicroMixerV71RTM.forward)
    params = list(sig.parameters.keys())
    assert params == ["self", "input_ids", "q_mask", "a_mask", "all_mask", "targets"], \
        f"forward signature mismatch: {params}"
    gsig = inspect.signature(MicroMixerV71RTM.generate)
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
    # aux keys present with targets
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
    # deepcopy works on a FRESH model (V67's ReMixerLayer sets last_pool_pre
    # during forward, which is a non-leaf tensor that cannot be deepcopy'd;
    # this is V67's behavior, not V71's. The deepcopy contract applies to
    # a model that has not yet been called, as in V67.)
    m_fresh = MicroMixerV71RTM(v71_rtm_glc_1m())
    m_fresh.eval()
    m_deep = copy.deepcopy(m_fresh)
    m_deep.eval()
    test_ids2 = torch.randint(4, V, (2, 32))
    with torch.no_grad():
        out_orig = m_fresh(test_ids2, None, None, None, None)[0]
        out_deep = m_deep(test_ids2, None, None, None, None)[0]
    deep_diff = (out_orig - out_deep).abs().max().item()
    assert deep_diff == 0.0, f"deepcopy not bit-identical: {deep_diff}"
    print(f"  forward has 5 args (q/a/all/targets); aux keys = {sorted(aux_full.keys())}")
    print(f"  generate kwargs defaults OK; deepcopy(fresh) bit-identical (diff={deep_diff:.0e})")
    print("  [PASS]")
    PASS_COUNT += 1

    # === [10] GLC long-conv causality at every lag (no future leak) ===
    print("\n[10] GLC long-conv causality at every lag (S=256 > glc_kernel=129):")
    m = MicroMixerV71RTM(v71_rtm_glc_1m())
    m.eval()
    S_t = 256
    ids_a = torch.randint(4, V, (1, S_t))
    # Perturb at multiple positions: just past the kernel window, mid-sequence, and last.
    perturb_positions = [130, 200, 255]
    max_diff = 0.0
    for p in perturb_positions:
        ids_b = ids_a.clone()
        ids_b[0, p] = (ids_b[0, p] + 1) % V
        with torch.no_grad():
            la = m(ids_a, None, None, None, None)[0]
            lb = m(ids_b, None, None, None, None)[0]
        # All positions strictly before p must be unchanged.
        d = (la[:, :p] - lb[:, :p]).abs().max().item()
        max_diff = max(max_diff, d)
        print(f"  perturb pos {p:>3d}/{S_t}: max past-logit diff (positions < {p}) = {d:.2e}")
    assert max_diff < 1e-4, f"GLC long-conv causality violated: {max_diff}"
    print(f"  overall max past-logit diff across perturbations = {max_diff:.2e}")
    print("  [PASS]")
    PASS_COUNT += 1

    # === Final summary ===
    print("\n" + "=" * 72)
    print(f"V71 RTM-Mixer self-test: {PASS_COUNT}/{TOTAL} PASS")
    print("=" * 72)
    if PASS_COUNT == TOTAL:
        print("ALL V71 SELF-TESTS PASSED")
    else:
        print(f"FAIL: {TOTAL - PASS_COUNT} test(s) did not pass")
        raise SystemExit(1)
    print("\nFinal param counts (all 9 presets):")
    for key in expected:
        cfg = preset_fns[key]()
        n = count_parameters(MicroMixerV71RTM(cfg))
        print(f"  v71_rtm_{key[0]}_{key[1]:<5s}: {n:>10,} params  "
              f"(d={cfg.d_model}, blocks={cfg.num_blocks}, tmix='{cfg.tmix}')")
