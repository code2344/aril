# Adapted from MicroMixer-4 (Apache-2.0): package imports were adjusted.
"""MicroMixer-4 V46-3 Pure-MLP-Mixer -- Plain causal MLP-Mixer language model.

This is the **baseline** architecture for the V46 FMSP study: a plain causal
MLP-Mixer LM with NO role separation (no Q/A/C streams, no hypernetwork, no
question encoder) and NO flow-matching machinery. It exists to isolate the
contribution of FMSP fine-tuning from the contribution of the base
architecture.

== Why this model exists ==

Every prior MicroMixer version (V9-V45) uses some form of role separation:
  - V9-V33 (QAA/QAH/QAU): three streams Q/A/ALL-Context.
  - V34-V37 (QAG): two streams + a hypernetwork that generates answer weights.
  - V38-V45 (LMFlow/SFLM/TFREE): flow matching on hyperspheres.

None of these is a "pure" MLP-Mixer LM -- they all have architectural
machinery beyond the basic token-mix + channel-mix MLP-Mixer block. V46-3
creates that missing baseline so FMSP's effect can be measured on a clean
architecture where the ONLY variable is the fine-tuning technique.

== Architecture ==

  Input bytes (B, S)
    -> nn.Embedding(256 -> d_model)
    -> N x StreamBlock (causal depthwise-separable token-mix + GELU channel-mix)
    -> RMSNorm
    -> tied output head (h @ embed.weight.T)
    -> byte logits (256)
    -> CE on next-token targets

Each StreamBlock is reused unchanged from V37's Q-encoder primitive:
  - norm1 -> CausalDSConv1d (left-padded depthwise + pointwise expand/project)
  - norm2 -> ChannelMLP (Linear -> GELU -> Linear)
  - Pre-norm residual additions.

The token mixer is CAUSAL (left-padded conv), so position t only sees
positions <= t. This makes the model a proper autoregressive LM despite
using MLP-Mixer blocks (no attention, no recurrence, no SSM).

== Philosophy compliance ==

  1. MLP-Mixer backbone (depthwise token-mix + MLP channel-mix). No attention.
  2. Strictly under 1M parameters for every preset.
  3. No attention, no recurrence, no SSM.

== Generation ==

Standard autoregressive: at each step, forward the full sequence, take the
last position's logits, apply temperature/top-k/top-p, sample, append.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_v37_qag import (
    RMSNorm,
    CausalDSConv1d,
    ChannelMLP,
    StreamBlock,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class V46PureConfig:
    """Configuration for MicroMixerV46PureMixer."""
    vocab_size: int = 256
    d_model: int = 128
    num_blocks: int = 7
    kernel_size: int = 3
    dilations: tuple = (1, 2, 4, 8, 16, 32, 64)
    expansion_token: int = 2
    expansion_channel: int = 2
    max_seq_len: int = 1024
    pad_token_id: int = 0


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MicroMixerV46PureMixer(nn.Module):
    """Pure causal MLP-Mixer language model (no role separation).

    Single-stream byte-level LM. The forward pass accepts q_mask / a_mask
    / all_mask for interface compatibility with the shared Trainer, but
    ignores them -- all non-pad positions contribute to the loss.
    """

    def __init__(self, config: V46PureConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        self.embed = nn.Embedding(config.vocab_size, d)

        assert len(config.dilations) == config.num_blocks, (
            f"dilations must have one entry per block: "
            f"got {len(config.dilations)} for {config.num_blocks} blocks"
        )
        self.blocks = nn.ModuleList([
            StreamBlock(
                d_model=d,
                kernel_size=config.kernel_size,
                dilation=dil,
                expansion_token=config.expansion_token,
                expansion_channel=config.expansion_channel,
            )
            for dil in config.dilations
        ])

        self.out_norm = RMSNorm(d)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
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

    def forward(
        self,
        input_ids: torch.Tensor,
        q_mask: torch.Tensor | None = None,
        a_mask: torch.Tensor | None = None,
        all_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """Forward pass.

        Args:
            input_ids: (B, S) byte ids.
            q_mask / a_mask / all_mask: ignored (interface compat).
            targets: (B, S) next-token ids (already shifted by caller).
                If given, CE is computed on non-pad positions.

        Returns:
            (logits (B, S, vocab), loss | None, aux dict)
        """
        h = self.embed(input_ids)
        for block in self.blocks:
            h = block(h)
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
        """Autoregressive generation (next-token sampling)."""
        device = input_ids.device
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


# ---------------------------------------------------------------------------
# Parameter count + presets
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _receptive_field(dilations: tuple, kernel_size: int) -> int:
    """Causal half-receptive-field (in bytes, looking backward)."""
    return sum((kernel_size - 1) * d for d in dilations)


def v46_pure_100k() -> V46PureConfig:
    """~70K preset. d=48, 3 blocks. RF +-6 bytes."""
    return V46PureConfig(
        d_model=48, num_blocks=3,
        dilations=(1, 2, 4),
        max_seq_len=512,
    )


def v46_pure_300k() -> V46PureConfig:
    """~282K preset. d=80, 4 blocks. RF +-14 bytes."""
    return V46PureConfig(
        d_model=80, num_blocks=4,
        dilations=(1, 2, 8, 32),
        max_seq_len=1024,
    )


def v46_pure_500k() -> V46PureConfig:
    """~500K preset. d=96, 5 blocks. RF +-30 bytes."""
    return V46PureConfig(
        d_model=96, num_blocks=5,
        dilations=(1, 2, 8, 16, 32),
        max_seq_len=1024,
    )


def v46_pure_1m() -> V46PureConfig:
    """~961K preset. d=128, 7 blocks. RF +-126 bytes."""
    return V46PureConfig(
        d_model=128, num_blocks=7,
        dilations=(1, 2, 4, 8, 16, 32, 64),
        max_seq_len=1024,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)
    print("=" * 72)
    print("MicroMixer-4 V46-3 Pure-MLP-Mixer -- self-test")
    print("(Plain causal MLP-Mixer, no role separation)")
    print("=" * 72)

    # ---- 1. Param budgets -----------------------------------------------
    print("\n[1] Parameter budgets (must all be < 1,000,000):")
    presets = [
        ("100k", v46_pure_100k),
        ("300k", v46_pure_300k),
        ("500k", v46_pure_500k),
        ("1m",   v46_pure_1m),
    ]
    for name, fn in presets:
        cfg = fn()
        m = MicroMixerV46PureMixer(cfg)
        n = count_parameters(m)
        rf = _receptive_field(cfg.dilations, cfg.kernel_size)
        status = "OK" if n < 1_000_000 else "FAIL"
        print(f"  {name:5s}: {n:>10,} params  d={cfg.d_model:<4d} "
              f"blocks={cfg.num_blocks} RF=+-{rf:<5d} bytes [{status}]")
        assert n < 1_000_000, f"Preset {name} exceeds 1M: {n:,}"
    print("  [OK] all 4 presets strictly under 1M")

    # ---- 2. Forward + backward sanity -----------------------------------
    print("\n[2] Forward + backward sanity (1M preset, B=4, S=64):")
    cfg = v46_pure_1m()
    m = MicroMixerV46PureMixer(cfg)
    V = cfg.vocab_size

    B, S = 4, 64
    ids = torch.randint(4, V, (B, S))
    targets = torch.cat([ids[:, 1:], torch.full((B, 1), cfg.pad_token_id, dtype=torch.long)], dim=1)

    m.train()
    logits, loss, aux = m(ids, None, None, None, targets)
    assert logits.shape == (B, S, V), f"logits shape {logits.shape}"
    assert torch.isfinite(loss).item(), f"loss not finite: {loss}"
    loss.backward()

    finite_grads = True
    for name, p in m.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            finite_grads = False
            print(f"  NON-FINITE GRAD: {name}")
    assert finite_grads, "non-finite gradients detected"
    print(f"  loss={loss.item():.4f}  token_acc={aux['token_acc'].item():.4f}")
    print(f"  logits range=[{logits.min().item():.3f}, {logits.max().item():.3f}]")
    print("  [OK]")

    # ---- 3. Causality: position t does not depend on t+1 ----------------
    print("\n[3] Causality test (perturbing future does not change past logits):")
    m.eval()
    ids_a = torch.randint(4, V, (1, S))
    ids_b = ids_a.clone()
    ids_b[0, S - 1] = (ids_b[0, S - 1] + 1) % V  # change last position

    with torch.no_grad():
        logits_a = m(ids_a, None, None, None, None)[0]
        logits_b = m(ids_b, None, None, None, None)[0]

    # All positions except the last should be identical.
    max_past_diff = (logits_a[:, :-1] - logits_b[:, :-1]).abs().max().item()
    print(f"  max |logits_a[:S-1] - logits_b[:S-1]| = {max_past_diff:.2e} (should be ~0)")
    assert max_past_diff < 1e-4, "Causality violation: past logits changed when future was perturbed"
    print("  [OK] causal: past positions do not depend on future")

    # ---- 4. Generation --------------------------------------------------
    print("\n[4] Generation (1M preset, 20 tokens):")
    m.eval()
    prompt = torch.tensor([[1, 5, 10, 15, 20]], dtype=torch.long)  # BOS + bytes
    out = m.generate(prompt, max_new_tokens=20, temperature=0.0)
    assert out.shape == (1, 5 + 20), f"generate shape {out.shape}"
    assert torch.equal(out[:, :5], prompt), "generate modified prompt!"
    print(f"  generated shape: {tuple(out.shape)}")
    print(f"  prompt preserved: True")
    print("  [OK]")

    # ---- 5. Interface compatibility (q_mask/a_mask accepted) ------------
    print("\n[5] Interface compat (q_mask/a_mask/all_mask accepted, ignored):")
    q_mask = torch.ones(B, S, dtype=torch.bool)
    a_mask = torch.zeros(B, S, dtype=torch.bool); a_mask[:, S//2:] = True
    all_mask = torch.ones(B, S, dtype=torch.bool)
    logits2, loss2, _ = m(ids, q_mask, a_mask, all_mask, targets)
    max_diff = (logits - logits2).abs().max().item()
    print(f"  max |logits(no mask) - logits(with mask)| = {max_diff:.2e} (should be 0)")
    assert max_diff < 1e-6, "masks affected output!"
    print("  [OK] masks correctly ignored")

    # ---- 6. Philosophy compliance ---------------------------------------
    print("\n[6] Philosophy compliance (3 rules):")
    has_attn = any("attention" in n.lower() or "attn" in n.lower()
                   for n, _ in m.named_modules())
    has_rnn = any(isinstance(mod, (nn.RNN, nn.LSTM, nn.GRU))
                  for _, mod in m.named_modules())
    has_mha = any(isinstance(mod, nn.MultiheadAttention)
                  for _, mod in m.named_modules())
    assert not has_attn, "model contains attention module!"
    assert not has_rnn, "model contains RNN module!"
    assert not has_mha, "model contains MultiheadAttention!"
    print("  Rule 1 (MLP-Mixer, no attention): PASS")
    print("  Rule 2 (< 1M params, all presets): PASS (verified in [1])")
    print("  Rule 3 (no attention/recurrence/SSM): PASS")

    # ---- 7. No role separation (no Q/A/C stream modules) ----------------
    print("\n[7] No role separation (no question_encoder, answer_stream, etc.):")
    module_names = [n for n, _ in m.named_modules()]
    forbidden = ["question_encoder", "answer_stream", "all_stream", "context_stream",
                 "q_summary", "basis", "comp_mlp", "pos_proj", "film", "time_mlp"]
    found_forbidden = [f for f in forbidden if any(f in n for n in module_names)]
    assert not found_forbidden, f"Found forbidden role-separation modules: {found_forbidden}"
    print(f"  Module list: {[n for n in module_names if n]}")
    print("  [OK] no role separation -- pure MLP-Mixer")

    print("\n" + "=" * 72)
    print("ALL V46-3 PURE-MLP-MIXER SELF-TESTS PASSED")
    print("=" * 72)
    for name, fn in presets:
        cfg = fn()
        m = MicroMixerV46PureMixer(cfg)
        print(f"  {name} preset: {count_parameters(m):,} params")
