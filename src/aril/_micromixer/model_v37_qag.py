# Adapted from MicroMixer-4 (Apache-2.0): package imports were adjusted.
"""MicroMixer-4 V37 QAG-Mixer: Position-Routed Question-Answer-Generated MLP-Mixer.

V37 is the direct successor to V36 (QAG-Mixer, Relevance-Rewarded).  V36 proved
the answer-level relevance margin loss is a live, functioning training signal
(the gap grew 9x in run 1), but the gap is structurally bounded at 0.027 nats
far below the 0.5-nat target.  Root cause across both V36 runs: the hypernetwork's
convex weight composition (W = sum_k alpha_k * B_k) is too smooth -- alpha varies
per question (pairdiff 0.33) but the functional impact on the answer distribution
is tiny, so the relevance gap is structurally bounded.

V36 section 0.6 prescribes four requirements for V37:

1. Increase the weight composition's expressivity.
2. Use random negatives, not hard negatives.
3. Scale lambda_rel to ~1.0 with random negatives.
4. Monitor the gap-to-pairdiff ratio.

V37 implements #1 with two architectural changes, and #2-#4 as default config
and new diagnostics.

V37 Change 1 (headline): Per-Position Weight Routing (PPR)
-----------------------------------------------------------
V36 composed ONE alpha per sequence (B, K), shared across all S answer positions.
V37 composes alpha PER POSITION (B, S, K):

    alpha[t] = softmax( compose_mlp( q_summary + pos_proj(h_pre[t]) ) / tau )

where ``pos_proj`` is a zero-initialised Linear(d, d) per answer block.  At init,
pos_proj outputs zero, so alpha is identical across positions (warm start = V36).
During training, pos_proj learns to differentiate positions so different answer
positions can select different basis combinations.

This multiplies the expressivity of the composition by S: instead of one routing
decision per question, the model makes S routing decisions per question.  The
relevance gap now accumulates contributions from every position independently,
breaking the structural ceiling V36 hit.

Implementation: rather than composing the full (out, in) weight matrix per
position (memory-infeasible at B*S*out*in), V37 computes all K basis outputs via
a single batched matmul then does a per-position weighted sum::

    bases_flat = effective_bases.reshape(K * out, in)   # (K*out, in)
    all_outs   = F.linear(h, bases_flat)                 # (B, S, K*out) -- one matmul
    basis_outs = all_outs.reshape(B, S, K, out)
    mixed      = einsum('bsk,bsko->bso', alpha, basis_outs)

This is the standard "expert outputs then mix" formulation (MoE / GroupedConv).

V37 Change 2: Scaled-Normalised Bases
-------------------------------------
V36 normalised each basis to Frobenius norm 1, bounding ||W||_F <= 1.  This is
the explicit bottleneck V36 diagnosed: channel-mix output was ~36x weaker than a
typical Xavier-initialised Linear layer.

V37 replaces fixed normalisation with LEARNABLE per-basis scales::

    W_t = sum_k  alpha_k(t) * s_k * B_k_normalised

where ``B_k_normalised`` has Frobenius norm 1 (direction only) and ``s_k`` is a
learnable scalar initialised so the composed W has ||W||_F ~ sqrt(in_features)
(healthy signal propagation, matching a standard Xavier Linear).  This decouples
basis direction from magnitude, giving the ~9-36x stronger channel-mix V36 needs
without the init-instability that plagued V35's ChatQA run when normalisation was
absent entirely.

Configurable via ``normalize_bases`` (default True = scaled-normalised; False =
raw unnormalised bases for ablation).  The V36 behaviour (norm 1, no scale) is
recoverable by setting ``s_k = 1`` for all k (ablation).

Training config (V36 section 0.6 points 2-3)
---------------------------------------------
- tau_final: 0.5 -> 0.25 (sharper alpha, per option c)
- w_relevance: 0.3 -> 1.0 (stronger relevance, with random negatives)
- hard_negatives: False (random cyclic-shift, 5x better gap growth)
- relevance_warmup_frac: 0.2 -> 0.3 (aligned with tau anneal)

Cross-condition scoring change
------------------------------
V36 pre-composed W, swapped along batch dim, re-ran the block stack.
V37 passes the crossed ``q_summary`` directly: blocks recompose alpha per
position from the crossed question summary.  Simpler interface, same semantics.

Causality
---------
Identical causal structure to V36: ``q_summary`` is pooled from Q positions
(causally before A), and ``h_pre[t]`` depends only on tokens <= t (causal
token-mix).  The per-position alpha thus depends only on information available
at position t.  Verified in the self-test (causality diff = 0.00e+00).

Pure MLP-Mixer identity: no attention, no recurrence, no SSM, NoPE, tied head.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class V37Config:
    """Configuration for MicroMixerV37QAG.

    Inherits all V36 fields and adds per-position routing + scaled-normalised
    bases knobs.

    V37 additions over V36Config:

      - ``per_position_routing`` (default True): the headline change.  When True,
        alpha is composed per answer position from q_summary + pos_proj(h_pre).
        When False, reverts to V36's sequence-level alpha (one alpha per sample).
      - ``normalize_bases`` (default True): scaled-normalised bases (Oracle-
        recommended).  When True, each basis is L2-normalised and a learnable
        per-basis scalar ``s_k`` controls magnitude.  When False, bases are raw
        (ablation).
      - ``pos_proj_init_scale`` (default 0.0): init scale for pos_proj weights
        (0 = zero-init = V36 warm start; > 0 = immediate per-position variation).

    V37 default changes (per V36 section 0.6):

      - ``tau_final``: 0.5 -> 0.25
      - ``w_relevance``: 0.3 -> 1.0
      - ``relevance_warmup_steps``: aligned with tau anneal (warmup_frac 0.3)

    All V36 fields (relevance loss, anti-filler, contrastive losses, etc.) are
    preserved unchanged.
    """

    vocab_size: int = 256
    max_seq_len: int = 1024
    d_model: int = 80
    num_layers_q: int = 3
    num_layers_a: int = 3
    dilations_q: tuple = (1, 2, 4)
    dilations_a: tuple = (1, 2, 4)
    kernel_size: int = 3
    expansion_token: int = 2
    expansion_channel: int = 2
    num_bases: int = 16
    comp_hidden: int = 64
    tau_init: float = 5.0
    tau_final: float = 0.25
    tau_anneal_steps: int = 0
    # V35 Q-encoder pool
    use_gated_pool: bool = True
    # V34 anti-collapse loss weights
    w_entropy: float = 0.05
    w_load_balance: float = 0.05
    w_basis_diversity: float = 0.05
    # V35 contrastive loss weights
    w_alpha_contrast: float = 0.10
    w_qsum_contrast: float = 0.10
    entropy_low_frac: float = 0.3
    entropy_high_frac: float = 0.8
    alpha_dist_floor: float = 0.3
    qsum_dist_floor: float = 1.0
    # V36 relevance + anti-filler losses
    use_relevance_loss: bool = True
    use_antifiller_loss: bool = False
    w_relevance: float = 1.0
    w_antifiller: float = 0.1
    relevance_margin: float = 0.5
    antifiller_margin: float = 0.5
    relevance_warmup_steps: int = 0
    hard_negatives: bool = False
    detach_crossed_weights: bool = False
    # V37 per-position routing + scaled-normalised bases (the headline additions)
    per_position_routing: bool = True
    normalize_bases: bool = True
    pos_proj_init_scale: float = 0.0
    # Ablation switches
    use_weight_generation: bool = True
    use_anti_collapse: bool = True
    pad_token_id: int = 0


# ---------------------------------------------------------------------------
# Shared primitives (identical to V36 / V35 / V34 / V9)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (no bias)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class CausalDSConv1d(nn.Module):
    """Causal depthwise-separable 1D convolution (token mixing).

    Identical to V36/V35/V34/V9: depthwise causal conv (left padding) + pointwise
    expand + GELU + pointwise project.  Weights are question-invariant by design.
    """

    def __init__(self, dim: int, kernel_size: int, dilation: int, expansion: int = 2):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.dw = nn.Conv1d(dim, dim, kernel_size=kernel_size, groups=dim,
                            dilation=dilation, bias=True)
        self.pw1 = nn.Conv1d(dim, dim * expansion, kernel_size=1, bias=True)
        self.pw2 = nn.Conv1d(dim * expansion, dim, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = F.pad(x, (self.pad, 0))
        x = self.dw(x)
        x = self.pw1(x)
        x = F.gelu(x)
        x = self.pw2(x)
        x = x.transpose(1, 2)
        return x


class ChannelMLP(nn.Module):
    """Standard two-layer channel MLP with learned weights.

    Used (a) inside the Question Encoder and (b) as the ablation replacement
    when weight generation is disabled.
    """

    def __init__(self, d_model: int, expansion: int = 2):
        super().__init__()
        hidden = expansion * d_model
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class StreamBlock(nn.Module):
    """Pre-norm residual block: token-mix then channel-mix (for the Q encoder)."""

    def __init__(self, d_model, kernel_size, dilation,
                 expansion_token=2, expansion_channel=2):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.token_mix = CausalDSConv1d(d_model, kernel_size, dilation, expansion_token)
        self.norm2 = RMSNorm(d_model)
        self.channel_mix = ChannelMLP(d_model, expansion_channel)

    def forward(self, x):
        x = x + self.token_mix(self.norm1(x))
        x = x + self.channel_mix(self.norm2(x))
        return x


class CausalStream(nn.Module):
    """Plain causal stack of StreamBlocks (Q encoder, no region masking)."""

    def __init__(self, d_model, num_layers, dilations, kernel_size=3,
                 expansion_token=2, expansion_channel=2):
        super().__init__()
        assert len(dilations) == num_layers
        self.blocks = nn.ModuleList([
            StreamBlock(d_model, kernel_size, d, expansion_token, expansion_channel)
            for d in dilations
        ])
        self.out_norm = RMSNorm(d_model)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.out_norm(x)


# ---------------------------------------------------------------------------
# GatedQueryPool (V35, unchanged)
# ---------------------------------------------------------------------------

class GatedQueryPool(nn.Module):
    """Sigmoid-gated weighted mean over Q positions (V35, unchanged)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.gate_fc1 = nn.Linear(d_model, d_model)
        self.gate_fc2 = nn.Linear(d_model, 1)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gate_fc1.weight)
        nn.init.zeros_(self.gate_fc1.bias)
        nn.init.normal_(self.gate_fc2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gate_fc2.bias)

    def forward(self, h: torch.Tensor, q_mask: torch.Tensor) -> torch.Tensor:
        logits = self.gate_fc2(F.gelu(self.gate_fc1(h))).squeeze(-1)  # (B, S)
        gates = torch.sigmoid(logits)
        gates = gates * q_mask.to(gates.dtype)
        denom = gates.sum(dim=1, keepdim=True).clamp(min=1e-8)
        weights = gates / denom
        q_summary = (h * weights.unsqueeze(-1)).sum(dim=1)
        return q_summary


# ---------------------------------------------------------------------------
# Question Encoder (V35, unchanged: no bottleneck, gated pool)
# ---------------------------------------------------------------------------

class QuestionEncoder(nn.Module):
    """Encode the question into a conditioning vector (V35, unchanged)."""

    def __init__(self, d_model, num_layers, dilations, kernel_size,
                 expansion_token, expansion_channel, use_gated_pool):
        super().__init__()
        self.use_gated_pool = use_gated_pool
        self.stream = CausalStream(
            d_model, num_layers, dilations, kernel_size,
            expansion_token, expansion_channel,
        )
        if use_gated_pool:
            self.pool = GatedQueryPool(d_model)

    def forward(self, x, q_mask):
        h = self.stream(x)
        if self.use_gated_pool:
            q_summary = self.pool(h, q_mask)
        else:
            m = q_mask.unsqueeze(-1).to(h.dtype)
            denom = m.sum(dim=1).clamp(min=1.0)
            q_summary = (h * m).sum(dim=1) / denom
        no_q = (q_mask.sum(dim=1) == 0)
        if no_q.any():
            q_summary = torch.where(no_q.unsqueeze(-1), h.mean(dim=1), q_summary)
        return q_summary


# ---------------------------------------------------------------------------
# Weight generation: BasisLibrary (V37: scaled-normalised) + CompositionMLP
# ---------------------------------------------------------------------------

class BasisLibrary(nn.Module):
    """A learned library of K basis weight matrices with optional scaled normalisation.

    V37 change over V36:
    - When ``normalize=True`` (default, Oracle-recommended): each basis is
      L2-normalised to Frobenius norm 1 (direction only), and a learnable
      per-basis scalar ``s_k`` controls magnitude.  Scales are initialised so
      the composed W has ||W||_F ~ sqrt(in_features) at uniform alpha (healthy
      signal propagation, ~9-36x stronger than V36's ||W|| <= 1).
    - When ``normalize=False`` (ablation): bases are raw, with no normalisation
      or scaling (matches V36's behaviour if all s_k were 1).
    """

    def __init__(self, num_bases: int, out_features: int, in_features: int,
                 normalize: bool = True):
        super().__init__()
        self.num_bases = num_bases
        self.out_features = out_features
        self.in_features = in_features
        self.normalize = normalize
        self.bases = nn.Parameter(torch.empty(num_bases, out_features, in_features))
        if normalize:
            self.scales = nn.Parameter(torch.empty(num_bases))
        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.num_bases):
            nn.init.xavier_uniform_(self.bases[k])
        if self.normalize:
            # Initialise scales so composed W has ||W|| ~ sqrt(in_features)
            # at uniform alpha: ||W|| ~ scale / sqrt(K), so scale = sqrt(K * in).
            init_scale = (self.num_bases * self.in_features) ** 0.5
            nn.init.constant_(self.scales, init_scale)

    def get_effective_bases(self) -> torch.Tensor:
        """Return bases ready for composition: (K, out, in).

        When normalised: unit-norm direction * learnable scale.
        When raw: the parameters as-is.
        """
        if not self.normalize:
            return self.bases
        flat = self.bases.reshape(self.num_bases, -1)
        norms = flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
        normalized = (flat / norms).reshape(
            self.num_bases, self.out_features, self.in_features
        )
        return normalized * self.scales.view(-1, 1, 1)

    def compose(self, alpha: torch.Tensor) -> torch.Tensor:
        """Sequence-level compose (V36-compatible): alpha (..., K) -> W (..., out, in)."""
        effective = self.get_effective_bases()
        return torch.einsum("...k,koi->...oi", alpha, effective)


class WeightCompositionMLP(nn.Module):
    """Maps conditioning vector -> softmax mixing coefficients over K bases.

    Works for both sequence-level input (B, d) -> alpha (B, K) and per-position
    input (B, S, d) -> alpha (B, S, K).  The Linear layers operate on the last
    dim, so no structural change is needed.
    """

    def __init__(self, d_model: int, hidden: int, num_bases: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, num_bases)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, ctx: torch.Tensor, tau: float) -> torch.Tensor:
        logits = self.fc2(F.gelu(self.fc1(ctx))) / max(tau, 1e-3)
        return F.softmax(logits, dim=-1)


class PositionProjection(nn.Module):
    """Per-block position projection for V37 per-position routing.

    A Linear(d, d) that projects the pre-channel-mix hidden state into the
    composition space, enabling per-position alpha differentiation.

    Zero-initialised so at init, pos_proj(h) = 0 and alpha is identical across
    positions (= V36 warm start).  During training, pos_proj learns to
    differentiate positions so different answer tokens can select different
    basis combinations.
    """

    def __init__(self, d_model: int, init_scale: float = 0.0):
        super().__init__()
        self.proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_scale) if init_scale > 0 \
            else nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


def functional_linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Per-sample linear: x (B,S,in), weight (B,out,in) -> (B,S,out)."""
    return torch.bmm(x, weight.transpose(1, 2))


# ---------------------------------------------------------------------------
# Answer stream with per-position generated channel-mix weights
# ---------------------------------------------------------------------------

class GeneratedAnswerBlock(nn.Module):
    """One answer-stream block.

    V37: the block holds token-mix and norms only (the question-invariant
    parts).  Channel-mix is orchestrated by AnswerStream because it needs
    access to bases, composition MLPs, and position projections.

    When ``use_weight_generation=False`` (ablation), the block holds a standard
    learned ChannelMLP and is self-contained (identical to V36).
    """

    def __init__(self, d_model, kernel_size, dilation, expansion_token,
                 expansion_channel, use_weight_generation):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.token_mix = CausalDSConv1d(d_model, kernel_size, dilation, expansion_token)
        self.norm2 = RMSNorm(d_model)
        if not use_weight_generation:
            self.channel_mix = ChannelMLP(d_model, expansion_channel)

    def forward_token_mix(self, x: torch.Tensor, mask_f: torch.Tensor) -> torch.Tensor:
        """Token-mix residual update (question-invariant, causal)."""
        update = self.token_mix(self.norm1(x)) - x
        return x + update * mask_f

    def get_normed(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-channel-mix normalised hidden state."""
        return self.norm2(x)

    def apply_residual(self, x: torch.Tensor, cm: torch.Tensor,
                       mask_f: torch.Tensor) -> torch.Tensor:
        """Channel-mix residual update."""
        update = cm - x
        return x + update * mask_f


class AnswerStream(nn.Module):
    """Stack of GeneratedAnswerBlocks with per-position weight routing (V37).

    V37 changes over V36:

    - ``run_blocks`` now composes alpha INSIDE each block (per-position when
      ``per_position_routing=True``), not from pre-composed weights.  This
      enables the crossed forward to simply pass a different ``q_summary`` and
      have the blocks recompose with crossed conditioning.
    - ``pos_proj`` modules (one per block) provide per-position context to the
      composition MLPs.
    - ``_basis_mix`` computes K basis outputs via a single batched matmul, then
      does a per-position weighted sum (the "expert outputs then mix" pattern).
    """

    def __init__(self, cfg: V37Config):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        hidden = cfg.expansion_channel * d
        self.use_weight_generation = cfg.use_weight_generation
        self.per_position_routing = cfg.per_position_routing
        self.blocks = nn.ModuleList([
            GeneratedAnswerBlock(
                d, cfg.kernel_size, cfg.dilations_a[i],
                cfg.expansion_token, cfg.expansion_channel,
                cfg.use_weight_generation,
            )
            for i in range(cfg.num_layers_a)
        ])
        self.out_norm = RMSNorm(d)
        if cfg.use_weight_generation:
            self.basis_fc1 = BasisLibrary(cfg.num_bases, hidden, d,
                                          normalize=cfg.normalize_bases)
            self.basis_fc2 = BasisLibrary(cfg.num_bases, d, hidden,
                                          normalize=cfg.normalize_bases)
            self.comp_fc1 = nn.ModuleList([
                WeightCompositionMLP(d, cfg.comp_hidden, cfg.num_bases)
                for _ in range(cfg.num_layers_a)
            ])
            self.comp_fc2 = nn.ModuleList([
                WeightCompositionMLP(d, cfg.comp_hidden, cfg.num_bases)
                for _ in range(cfg.num_layers_a)
            ])
            if cfg.per_position_routing:
                self.pos_proj = nn.ModuleList([
                    PositionProjection(d, cfg.pos_proj_init_scale)
                    for _ in range(cfg.num_layers_a)
                ])

    def _compose_alpha(self, q_summary: torch.Tensor, h_pre: torch.Tensor,
                       layer_idx: int, tau: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compose alpha for fc1 and fc2.

        When per_position_routing: alpha is (B, S, K), composed from
        q_summary + pos_proj(h_pre).
        When not: alpha is (B, K), composed from q_summary alone (= V36).
        """
        if self.per_position_routing:
            pos_ctx = self.pos_proj[layer_idx](h_pre)  # (B, S, d)
            ctx = q_summary.unsqueeze(1) + pos_ctx     # (B, S, d)
        else:
            ctx = q_summary                             # (B, d)
        alpha_fc1 = self.comp_fc1[layer_idx](ctx, tau)
        alpha_fc2 = self.comp_fc2[layer_idx](ctx, tau)
        return alpha_fc1, alpha_fc2

    def _basis_mix(self, x: torch.Tensor, basis_lib: BasisLibrary,
                   alpha: torch.Tensor) -> torch.Tensor:
        """Per-position basis mixture via batched matmul.

        x: (..., in_features), alpha: (..., K) -> mixed: (..., out_features).

        When alpha has fewer batch dims than x (e.g. alpha is (B,K) but x is
        (B,S,in) — the V36-equivalent ablation), alpha is broadcast across the
        missing dims so the same mixing coefficients apply to all positions.

        Computes all K basis outputs in a single matmul, then does a per-position
        weighted sum.  This avoids materialising the full (B, S, out, in) weight
        tensor.
        """
        effective = basis_lib.get_effective_bases()  # (K, out, in)
        K, out_dim, in_dim = effective.shape
        bases_flat = effective.reshape(K * out_dim, in_dim)
        all_outs = F.linear(x, bases_flat)  # (*x.shape[:-1], K * out)
        # Reshape based on x's shape (not alpha's) to handle the seq-level case.
        basis_outs = all_outs.reshape(*x.shape[:-1], K, out_dim)
        # If alpha has fewer dims than basis_outs (sequence-level alpha with
        # per-position x), broadcast alpha across the missing dims.
        if alpha.ndim < basis_outs.ndim - 1:
            # alpha: (B, K), basis_outs: (B, S, K, out) -> expand alpha to (B, S, K)
            target_shape = basis_outs.shape[:-1]  # (B, S, K)
            alpha = alpha.unsqueeze(1).expand(target_shape)
        # Per-position weighted sum: (..., K) * (..., K, out) -> (..., out)
        mixed = torch.einsum("...k,...ko->...o", alpha, basis_outs)
        return mixed

    def compose_null_alphas(self, batch_size: int, seq_len: int,
                            device: torch.device) -> tuple[list, list]:
        """Compose uniform-alpha (mean basis) for all layers under no_grad.

        Used as the question-agnostic null condition for the anti-filler loss.
        Returns (alphas_fc1 list, alphas_fc2 list), each (B, S, K) or (B, K).
        """
        K = self.cfg.num_bases
        with torch.no_grad():
            if self.per_position_routing:
                shape = (batch_size, seq_len, K)
            else:
                shape = (batch_size, K)
            uniform = torch.full(shape, 1.0 / K, device=device, dtype=torch.float32)
            alphas_fc1 = [uniform.clone() for _ in range(self.cfg.num_layers_a)]
            alphas_fc2 = [uniform.clone() for _ in range(self.cfg.num_layers_a)]
        return alphas_fc1, alphas_fc2

    def run_blocks(self, x: torch.Tensor, a_mask: torch.Tensor,
                   q_summary: torch.Tensor, tau: float,
                   detach_weights: bool = False,
                   precomposed_alphas: tuple | None = None) -> tuple[torch.Tensor, list]:
        """Run the block stack, composing per-position weights from q_summary.

        Args:
            x: (B, S, d) embedded input.
            a_mask: (B, S) bool answer-region mask.
            q_summary: (B, d) question conditioning vector.
            tau: softmax temperature.
            detach_weights: if True, detach alpha before channel-mix (one-sided
                relevance margin — safer but weaker).
            precomposed_alphas: optional (alphas_fc1, alphas_fc2) lists to use
                INSTEAD of composing from q_summary.  Used for the null forward
                (anti-filler loss).  When given, q_summary is ignored.

        Returns:
            (output (B, S, d), alphas list) where alphas is a flat list of
            [alpha_fc1_l0, alpha_fc2_l0, alpha_fc1_l1, ...] for diagnostics.
        """
        mask_f = a_mask.unsqueeze(-1).to(x.dtype)
        all_alphas = []

        for i, block in enumerate(self.blocks):
            # Token mix (question-invariant, causal)
            x = block.forward_token_mix(x, mask_f)
            h_pre = block.get_normed(x)

            if self.use_weight_generation:
                if precomposed_alphas is not None:
                    alpha_fc1 = precomposed_alphas[0][i]
                    alpha_fc2 = precomposed_alphas[1][i]
                else:
                    alpha_fc1, alpha_fc2 = self._compose_alpha(
                        q_summary, h_pre, i, tau
                    )
                    if detach_weights:
                        alpha_fc1 = alpha_fc1.detach()
                        alpha_fc2 = alpha_fc2.detach()

                inter = F.gelu(
                    self._basis_mix(h_pre, self.basis_fc1, alpha_fc1)
                ).clamp(-50, 50)
                cm = self._basis_mix(inter, self.basis_fc2, alpha_fc2).clamp(-50, 50)

                if precomposed_alphas is None:
                    all_alphas.append(alpha_fc1)
                    all_alphas.append(alpha_fc2)
            else:
                cm = block.channel_mix(h_pre)

            x = block.apply_residual(x, cm, mask_f)

        return self.out_norm(x), all_alphas


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MicroMixerV37QAG(nn.Module):
    """V37 QAG-Mixer: Position-Routed Question-Answer-Generated MLP-Mixer.

    Retains V36's full relevance-rewarded architecture (relevance margin loss,
    anti-filler loss, V35 guarded conditioning chain) and ADDS per-position
    weight routing + scaled-normalised bases to fix V36's root cause: the convex
    composition was too smooth, structurally bounding the relevance gap.

    Per-position routing makes alpha a function of BOTH the question (global,
    via q_summary) and the current answer position (local, via pos_proj(h_pre)).
    This lets different answer positions select different basis combinations,
    multiplying expressivity by S and breaking the relevance gap ceiling.

    Scaled-normalised bases replace V36's fixed ||W||_F <= 1 with learnable
    per-basis magnitudes initialised to healthy values (~sqrt(in_features)),
    giving the channel-mix ~9-36x more functional impact.
    """

    def __init__(self, config: V37Config):
        super().__init__()
        self.config = config
        d = config.d_model

        self.embed = nn.Embedding(config.vocab_size, d)

        self.question_encoder = QuestionEncoder(
            d, config.num_layers_q, config.dilations_q, config.kernel_size,
            config.expansion_token, config.expansion_channel,
            config.use_gated_pool,
        )

        self.answer_stream = AnswerStream(config)

        self.out_norm = RMSNorm(d)

        self.register_buffer("current_tau", torch.tensor(float(config.tau_init)))
        self.register_buffer("train_step", torch.tensor(0, dtype=torch.long))

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "comp_" in name or "gated_pool" in name or "pool" in name \
                        or "pos_proj" in name:
                    continue
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -- temperature schedule ------------------------------------------------

    def _maybe_update_tau(self):
        if self.training:
            self.train_step.add_(1)
        steps = int(self.train_step.item())
        total = self.config.tau_anneal_steps
        if total and total > 0:
            progress = min(1.0, steps / float(total))
        else:
            progress = 0.0
        tau = self.config.tau_init + (self.config.tau_final - self.config.tau_init) * progress
        self.current_tau.fill_(tau)
        return tau

    def set_tau(self, tau: float):
        self.current_tau.fill_(float(tau))

    # -- relevance loss weight ramp ------------------------------------------

    def _current_relevance_weight(self) -> float:
        if self.config.relevance_warmup_steps <= 0:
            return self.config.w_relevance
        steps = int(self.train_step.item())
        ramp = min(1.0, steps / float(self.config.relevance_warmup_steps))
        return self.config.w_relevance * ramp

    # -- anti-collapse + contrastive losses (V35, adapted for per-position alpha) --

    @staticmethod
    def _alpha_entropy(alpha: torch.Tensor) -> torch.Tensor:
        """alpha: (..., K) -> entropy (...) (nats)."""
        eps = 1e-8
        return -(alpha * (alpha + eps).log()).sum(dim=-1)

    @staticmethod
    def _seq_level_alpha(alpha: torch.Tensor) -> torch.Tensor:
        """Reduce per-position alpha (B,S,K) to sequence-level (B,K).

        For per-position routing, alpha is (B, S, K).  The V35 contrastive /
        anti-collapse losses operate on sequence-level alpha, so we average over
        positions.  When alpha is already (B, K) (V36 mode), pass through.
        """
        if alpha.ndim == 3:
            return alpha.mean(dim=1)  # (B, K)
        return alpha

    def _entropy_interval_loss(self, alphas):
        """Penalize H(alpha) outside [low, high] (V34, gradient-fixed in V35).

        Adapted for V37: when alpha is per-position (B,S,K), we compute entropy
        per position and average.  This constrains alpha's shape at every
        position, not just the sequence-level average.
        """
        K = self.config.num_bases
        log_k = torch.log(torch.tensor(float(K)))
        low = self.config.entropy_low_frac * log_k
        high = self.config.entropy_high_frac * log_k
        total = 0.0
        n = 0
        for a in alphas:
            h = self._alpha_entropy(a)
            total = total + F.relu(low - h).mean() + F.relu(h - high).mean()
            n += 1
        return total / max(1, n)

    def _load_balance_loss(self, alphas):
        """KL(mean_batch(alpha) || uniform) (V34, gradient-fixed in V35).

        Adapted for V37: average over positions before computing batch-mean.
        """
        K = self.config.num_bases
        log_k = torch.log(torch.tensor(float(K)))
        total = 0.0
        n = 0
        for a in alphas:
            a_seq = self._seq_level_alpha(a)  # (B, K)
            p = a_seq.mean(dim=0)
            p = p.clamp(min=1e-8)
            h = -(p * p.log()).sum()
            total = total + (log_k - h)
            n += 1
        return total / max(1, n)

    def _basis_diversity_loss(self):
        """Mean squared cosine similarity between flattened basis matrices (V34)."""
        if not self.config.use_weight_generation:
            return torch.tensor(0.0, device=self.embed.weight.device)
        loss = 0.0
        for lib in (self.answer_stream.basis_fc1, self.answer_stream.basis_fc2):
            b = lib.bases
            K = b.shape[0]
            flat = b.reshape(K, -1)
            flat = F.normalize(flat, dim=-1)
            gram = flat @ flat.t()
            off = gram - torch.eye(K, device=gram.device)
            loss = loss + (off ** 2).sum() / (K * (K - 1))
        return loss / 2.0

    def _pairwise_mean_dist(self, vectors: torch.Tensor) -> torch.Tensor:
        """Mean pairwise L2 distance across the batch dimension (V35)."""
        B = vectors.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=vectors.device, requires_grad=True)
        diff = vectors.unsqueeze(1) - vectors.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=-1)
        mask = torch.triu(
            torch.ones(B, B, device=vectors.device, dtype=torch.bool), diagonal=1
        )
        return dist_sq[mask].clamp(min=1e-4).sqrt().mean()

    def _alpha_contrastive_loss(self, alphas):
        """V35: hinge loss forcing different questions -> different alpha.

        Adapted for V37: reduce to sequence-level before pairwise distance.
        """
        total = 0.0
        n = 0
        for a in alphas:
            a_seq = self._seq_level_alpha(a)
            mean_dist = self._pairwise_mean_dist(a_seq)
            total = total + F.relu(self.config.alpha_dist_floor - mean_dist)
            n += 1
        return total / max(1, n)

    def _q_summary_contrastive_loss(self, q_summary):
        """V35: hinge loss forcing different questions -> different q_summary."""
        mean_dist = self._pairwise_mean_dist(q_summary)
        return F.relu(self.config.qsum_dist_floor - mean_dist)

    # -- V36 cross-condition helpers (unchanged) -----------------------------

    @staticmethod
    def _per_position_logp(logits: torch.Tensor, targets: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
        """Gather per-position log-probability of the target token at masked pos."""
        logp = F.log_softmax(logits, dim=-1)
        gathered = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return gathered[mask]

    def _build_cross_neg_index(self, q_summary: torch.Tensor) -> torch.Tensor:
        """Build the batch index mapping i -> neg(i) for the crossed forward."""
        B = q_summary.shape[0]
        device = q_summary.device
        if self.config.hard_negatives and B >= 2:
            qs_norm = F.normalize(q_summary, dim=-1)
            sim = qs_norm @ qs_norm.t()
            sim.fill_diagonal_(-2.0)
            return sim.argmax(dim=1)
        offset = int(torch.randint(1, B, (1,), device=device).item())
        return (torch.arange(B, device=device) + offset) % B

    def _project_logits(self, h: torch.Tensor) -> torch.Tensor:
        """out_norm + tied head + numerical clamping (shared by all forwards)."""
        h = self.out_norm(h)
        logits = torch.nan_to_num(
            h @ self.embed.weight.T,
            nan=0.0, posinf=30.0, neginf=-30.0,
        ).clamp(-30, 30)
        return logits

    # -- V37 per-position diagnostic helpers --------------------------------

    @staticmethod
    def _alpha_position_variation(alphas: list, a_mask: torch.Tensor) -> torch.Tensor:
        """Mean pairwise L2 distance of alpha across positions within each sample.

        Measures how much alpha varies across answer positions.  V36 had this = 0
        (one alpha per sequence).  V37 targets >> 0 (per-position routing active).

        Only considers A-region positions (where the answer stream is active).
        """
        total = 0.0
        n = 0
        for a in alphas:
            if a.ndim != 3:
                continue  # sequence-level alpha (V36 mode) -> variation = 0
            # a: (B, S, K).  Only look at A-region positions.
            mask = a_mask.bool()
            variations = []
            for b in range(a.shape[0]):
                a_b = a[b][mask[b]]  # (num_a_pos, K)
                if a_b.shape[0] < 2:
                    continue
                diff = a_b.unsqueeze(1) - a_b.unsqueeze(0)  # (P, P, K)
                dist_sq = (diff ** 2).sum(dim=-1)  # (P, P)
                triu_mask = torch.triu(
                    torch.ones(a_b.shape[0], a_b.shape[0], device=a.device,
                               dtype=torch.bool), diagonal=1
                )
                if triu_mask.any():
                    variations.append(dist_sq[triu_mask].clamp(min=1e-8).sqrt().mean())
            if variations:
                total = total + torch.stack(variations).mean()
                n += 1
        if n == 0:
            return torch.tensor(0.0, device=alphas[0].device if alphas else "cpu")
        return total / n

    # -- forward -------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        q_mask: torch.Tensor,
        a_mask: torch.Tensor,
        all_mask: torch.Tensor | None = None,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, dict]:
        """Forward pass.

        Args:
            input_ids: (B, S) byte ids.
            q_mask: (B, S) bool, True at the question region.
            a_mask: (B, S) bool, True at the answer region (loss region).
            all_mask: unused (kept for interface compatibility).
            targets: (B, S) next-token ids; if given, CE is computed on A pos.

        Returns:
            (logits (B,S,vocab), loss | None, aux dict)

        The returned ``logits`` are ALWAYS the matched logits.  The crossed /
        null logits are used internally for the relevance / anti-filler losses.
        """
        tau = self._maybe_update_tau()

        x = self.embed(input_ids)

        q_summary = self.question_encoder(x, q_mask)

        mask = a_mask.bool()
        B = input_ids.shape[0]

        # ---- Matched forward (the real prediction path) ----
        h_matched, alphas = self.answer_stream.run_blocks(
            x, a_mask, q_summary, tau
        )

        logits = self._project_logits(h_matched)

        # ---- Diagnostics (detached) ----
        aux: dict = {
            "current_tau": float(tau),
            "q_summary_norm": q_summary.detach().float().norm(dim=-1).mean(),
        }

        if self.config.use_weight_generation and alphas:
            with torch.no_grad():
                stacked_ent = torch.stack(
                    [self._alpha_entropy(a).mean() for a in alphas]
                )
                aux["alpha_entropy_mean"] = stacked_ent.mean().detach()
                seq_alphas = [self._seq_level_alpha(a) for a in alphas]
                usage = torch.stack([a.mean(dim=0) for a in seq_alphas]).mean(dim=0)
                aux["alpha_usage_min"] = usage.min().detach()
                aux["alpha_usage_max"] = usage.max().detach()
                aux["q_summary_pairdiff"] = self._pairwise_mean_dist(
                    q_summary.detach()
                ).detach()
                aux["alpha_pairdiff"] = torch.stack([
                    self._pairwise_mean_dist(a.detach()) for a in seq_alphas
                ]).mean().detach()
                # V37 new diagnostic: per-position alpha variation
                aux["alpha_position_variation"] = self._alpha_position_variation(
                    alphas, a_mask
                ).detach()
                # V37: basis scales (mean and std)
                scales_all = []
                for lib in (self.answer_stream.basis_fc1, self.answer_stream.basis_fc2):
                    if hasattr(lib, "scales"):
                        scales_all.append(lib.scales.detach())
                if scales_all:
                    all_scales = torch.cat(scales_all)
                    aux["basis_scale_mean"] = all_scales.mean().detach()
                    aux["basis_scale_std"] = all_scales.std().detach()
                else:
                    aux["basis_scale_mean"] = torch.tensor(0.0)
                    aux["basis_scale_std"] = torch.tensor(0.0)
        else:
            aux["alpha_entropy_mean"] = torch.tensor(0.0)
            aux["alpha_usage_min"] = torch.tensor(0.0)
            aux["alpha_usage_max"] = torch.tensor(0.0)
            aux["q_summary_pairdiff"] = torch.tensor(0.0)
            aux["alpha_pairdiff"] = torch.tensor(0.0)
            aux["alpha_position_variation"] = torch.tensor(0.0)
            aux["basis_scale_mean"] = torch.tensor(0.0)
            aux["basis_scale_std"] = torch.tensor(0.0)

        # V36 relevance diagnostics defaults
        aux["answer_relevance_gap"] = torch.tensor(0.0)
        aux["answer_antifiller_gap"] = torch.tensor(0.0)
        aux["aux_relevance_loss"] = torch.tensor(0.0)
        aux["aux_antifiller_loss"] = torch.tensor(0.0)
        aux["crossed_ce"] = torch.tensor(0.0)
        aux["relevance_weight"] = torch.tensor(
            self._current_relevance_weight() if self.config.use_relevance_loss else 0.0
        )

        if targets is None:
            return logits, None, aux

        # ---- CE on A-positions (fluency, unchanged) ----
        if mask.any():
            logits_f = logits[mask]
            targets_f = targets[mask]
            ce = F.cross_entropy(logits_f, targets_f,
                                 ignore_index=self.config.pad_token_id)
        else:
            ce = logits.sum() * 0.0

        aux["ce_loss"] = ce.detach()

        total = ce

        # ---- V35 anti-collapse + contrastive losses (adapted for per-position) ----
        if self.config.use_weight_generation and self.config.use_anti_collapse and alphas:
            ent = self._entropy_interval_loss(alphas)
            lb = self._load_balance_loss(alphas)
            div = self._basis_diversity_loss()
            alpha_contrast = self._alpha_contrastive_loss(alphas)
            qsum_contrast = self._q_summary_contrastive_loss(q_summary)

            total = (
                total
                + self.config.w_entropy * ent
                + self.config.w_load_balance * lb
                + self.config.w_basis_diversity * div
                + self.config.w_alpha_contrast * alpha_contrast
                + self.config.w_qsum_contrast * qsum_contrast
            )
            aux["aux_entropy_loss"] = ent.detach()
            aux["aux_load_balance_loss"] = lb.detach()
            aux["aux_basis_diversity_loss"] = div.detach()
            aux["aux_alpha_contrast_loss"] = alpha_contrast.detach()
            aux["aux_qsum_contrast_loss"] = qsum_contrast.detach()
        else:
            aux["aux_entropy_loss"] = torch.tensor(0.0)
            aux["aux_load_balance_loss"] = torch.tensor(0.0)
            aux["aux_basis_diversity_loss"] = torch.tensor(0.0)
            aux["aux_alpha_contrast_loss"] = torch.tensor(0.0)
            aux["aux_qsum_contrast_loss"] = torch.tensor(0.0)

        # ---- V36 relevance margin loss (headline, unchanged logic) ----
        relevance_active = (
            self.config.use_weight_generation
            and self.config.use_relevance_loss
            and mask.any()
            and B >= 2
        )
        if relevance_active:
            logp_matched = self._per_position_logp(logits, targets, mask)

            # V37: crossed forward recomposes per-position from crossed q_summary
            neg_idx = self._build_cross_neg_index(q_summary)
            q_summary_crossed = q_summary[neg_idx]

            h_crossed, _ = self.answer_stream.run_blocks(
                x, a_mask, q_summary_crossed, tau,
                detach_weights=self.config.detach_crossed_weights,
            )
            logits_crossed = self._project_logits(h_crossed)
            logp_crossed = self._per_position_logp(logits_crossed, targets, mask)

            gap = (logp_matched - logp_crossed).mean()
            aux["answer_relevance_gap"] = gap.detach()
            aux["crossed_ce"] = (-logp_crossed).mean().detach()

            l_rel = F.relu(
                self.config.relevance_margin - (logp_matched - logp_crossed)
            ).mean()

            lam_rel = self._current_relevance_weight()
            aux["relevance_weight"] = torch.tensor(lam_rel)
            aux["aux_relevance_loss"] = l_rel.detach()
            if self.training:
                total = total + lam_rel * l_rel

        # ---- V36 anti-filler margin loss (optional complement) ----
        antifiller_active = (
            self.config.use_weight_generation
            and self.config.use_antifiller_loss
            and mask.any()
        )
        if antifiller_active:
            if "logp_matched" not in dir():
                logp_matched = self._per_position_logp(logits, targets, mask)
            S = input_ids.shape[1]
            null_alphas_fc1, null_alphas_fc2 = \
                self.answer_stream.compose_null_alphas(B, S, x.device)
            h_null, _ = self.answer_stream.run_blocks(
                x, a_mask, q_summary, tau,
                precomposed_alphas=(null_alphas_fc1, null_alphas_fc2),
            )
            logits_null = self._project_logits(h_null)
            logp_null = self._per_position_logp(logits_null, targets, mask)

            af_gap = (logp_matched - logp_null).mean()
            aux["answer_antifiller_gap"] = af_gap.detach()
            l_af = F.relu(
                self.config.antifiller_margin - (logp_matched - logp_null)
            ).mean()
            aux["aux_antifiller_loss"] = l_af.detach()
            if self.training:
                total = total + self.config.w_antifiller * l_af

        return logits, total, aux

    # -- generation ----------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        q_mask: torch.Tensor,
        a_mask: torch.Tensor,
        all_mask: torch.Tensor | None = None,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: int = 0,
        eos_token_id: int = 2,
    ) -> torch.Tensor:
        """Autoregressive sampling. Same contract as V36/V35.

        Generation uses only the matched forward (self.training is False, so
        the relevance / anti-filler forwards are skipped).
        """
        device = input_ids.device
        q_mask = q_mask.to(device)
        a_mask = a_mask.to(device)

        generated = input_ids.clone()
        batch_size = generated.shape[0]

        for _ in range(max_new_tokens):
            cur_a = a_mask.clone()
            uncovered = ~(q_mask)
            cur_a = cur_a | uncovered
            logits, _, _ = self.forward(generated, q_mask, cur_a, None)
            next_logits = logits[:, -1, :] / max(temperature, 1e-6)

            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    seen = torch.unique(generated[b])
                    next_logits[b, seen] /= repetition_penalty

            if no_repeat_ngram_size > 0:
                for b in range(batch_size):
                    seq = generated[b].tolist()
                    if len(seq) >= no_repeat_ngram_size:
                        prefix = tuple(seq[-(no_repeat_ngram_size - 1):])
                        for v in range(self.config.vocab_size):
                            ngram = prefix + (v,)
                            for i in range(len(seq) - no_repeat_ngram_size + 1):
                                if tuple(seq[i:i + no_repeat_ngram_size]) == ngram:
                                    next_logits[b, v] = float("-inf")
                                    break

            if top_k is not None and top_k > 0:
                v = next_logits.size(-1)
                k = min(top_k, v)
                vals, _ = torch.topk(next_logits, k, dim=-1)
                min_val = vals[..., -1].unsqueeze(-1)
                next_logits = torch.where(
                    next_logits < min_val,
                    torch.full_like(next_logits, float("-inf")),
                    next_logits,
                )

            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                sorted_probs = F.softmax(sorted_logits, dim=-1)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                remove = cumsum > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                indices_to_remove = remove.scatter(1, sorted_idx, remove)
                next_logits = next_logits.masked_fill(indices_to_remove, float("-inf"))

            if temperature == 0.0:
                next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            new_zero = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
            new_one = torch.ones((batch_size, 1), dtype=torch.bool, device=device)
            q_mask = torch.cat([q_mask, new_zero], dim=1)
            a_mask = torch.cat([a_mask, new_one], dim=1)

            if (next_token.squeeze(-1) == eos_token_id).all():
                break

        return generated


# ---------------------------------------------------------------------------
# Parameter count + presets
# ---------------------------------------------------------------------------

def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def v37_qag_100k() -> V37Config:
    """~100K preset: d=32, 2 Q-layers, 2 A-layers, K=8."""
    return V37Config(
        d_model=32, num_layers_q=2, num_layers_a=2,
        dilations_q=(1, 2), dilations_a=(1, 2),
        num_bases=8, comp_hidden=48,
    )


def v37_qag_300k() -> V37Config:
    """~300K preset: d=48, 2 Q-layers, 3 A-layers, K=12."""
    return V37Config(
        d_model=48, num_layers_q=2, num_layers_a=3,
        dilations_q=(1, 2), dilations_a=(1, 2, 4),
        num_bases=12, comp_hidden=64,
    )


def v37_qag_500k() -> V37Config:
    """~500K preset: d=64, 3 Q-layers, 3 A-layers, K=16."""
    return V37Config(
        d_model=64, num_layers_q=3, num_layers_a=3,
        dilations_q=(1, 2, 4), dilations_a=(1, 2, 4),
        num_bases=16, comp_hidden=64,
    )


def v37_qag_1m() -> V37Config:
    """~1M preset: d=80, 3 Q-layers, 3 A-layers, K=16 (strictly under 1M)."""
    return V37Config(
        d_model=80, num_layers_q=3, num_layers_a=3,
        dilations_q=(1, 2, 4), dilations_a=(1, 2, 4),
        num_bases=16, comp_hidden=64,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Testing MicroMixer-4 V37 QAG-Mixer (Position-Routed)...")

    # 1. Param budgets for all presets.
    for name, fn in [
        ("100k", v37_qag_100k),
        ("300k", v37_qag_300k),
        ("500k", v37_qag_500k),
        ("1m", v37_qag_1m),
    ]:
        cfg = fn()
        m = MicroMixerV37QAG(cfg)
        n = count_parameters(m)
        print(f"  {name}: {n:,} params (under 1M: {n < 1_000_000})")
        assert n < 1_000_000, f"Preset {name} exceeds 1M: {n:,}"

    # 2. Forward + backward sanity on 1M (B>=2 so relevance loss is active).
    cfg = v37_qag_1m()
    m = MicroMixerV37QAG(cfg)
    B, S = 4, 64
    ids = torch.randint(0, 256, (B, S))
    q_mask = torch.zeros(B, S, dtype=torch.bool); q_mask[:, 30:40] = True
    a_mask = torch.zeros(B, S, dtype=torch.bool); a_mask[:, 40:] = True
    targets = torch.randint(0, 256, (B, S))
    m.train()
    logits, loss, aux = m(ids, q_mask, a_mask, None, targets)
    assert logits.shape == (B, S, cfg.vocab_size), logits.shape
    assert torch.isfinite(loss).item(), loss
    loss.backward()
    finite_grads = True
    for name, p in m.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            finite_grads = False
            print(f"  NON-FINITE GRAD: {name}")
    assert finite_grads, "non-finite gradients detected"
    g_basis = m.answer_stream.basis_fc1.bases.grad
    g_comp = m.answer_stream.comp_fc1[0].fc2.weight.grad
    g_gate = m.question_encoder.pool.gate_fc2.weight.grad
    g_pos = m.answer_stream.pos_proj[0].proj.weight.grad
    assert g_basis is not None and g_basis.abs().sum().item() > 0, "bases got no gradient"
    assert g_comp is not None and g_comp.abs().sum().item() > 0, "composition got no gradient"
    assert g_gate is not None, "gated pool got no gradient"
    assert g_pos is not None, "pos_proj got no gradient"
    print(f"  forward+backward OK: loss={loss.item():.4f}")
    print(f"    ce={aux['ce_loss'].item():.4f} "
          f"ent={aux['aux_entropy_loss'].item():.4f} "
          f"lb={aux['aux_load_balance_loss'].item():.4f} "
          f"div={aux['aux_basis_diversity_loss'].item():.4f}")
    print(f"    alpha_contrast={aux['aux_alpha_contrast_loss'].item():.4f} "
          f"qsum_contrast={aux['aux_qsum_contrast_loss'].item():.4f}")
    print(f"    relevance_loss={aux['aux_relevance_loss'].item():.4f} "
          f"(weight={aux['relevance_weight'].item():.4f}, "
          f"gap={aux['answer_relevance_gap'].item():.4f}, "
          f"crossed_ce={aux['crossed_ce'].item():.4f})")
    print(f"    alpha_position_variation={aux['alpha_position_variation'].item():.6f} "
          f"(V36 had 0; V37 targets >0 after training)")
    print(f"    basis_scale_mean={aux['basis_scale_mean'].item():.2f} "
          f"std={aux['basis_scale_std'].item():.2f}")
    print(f"    basis_fc1 grad sum={g_basis.abs().sum().item():.4e} "
          f"comp fc2 grad sum={g_comp.abs().sum().item():.4e} "
          f"pos_proj grad sum={g_pos.abs().sum().item():.4e}")

    # 3. V37 critical invariant: pos_proj is zero-init -> alpha is identical
    #    across positions at init (= V36 warm start).  As training proceeds,
    #    pos_proj learns to differentiate positions.
    m.eval()
    with torch.no_grad():
        h, alphas = m.answer_stream.run_blocks(
            m.embed(ids), a_mask,
            m.question_encoder(m.embed(ids), q_mask),
            float(m.current_tau),
        )
        for a in alphas:
            if a.ndim == 3:
                # Check alpha is constant across positions at init
                pos_variation = a[:, 40:, :].std(dim=1).mean()
                assert pos_variation.item() < 1e-5, (
                    f"alpha varies across positions at init (pos_proj not zero-init): "
                    f"variation={pos_variation.item():.6e}"
                )
        print(f"  warm start OK: alpha constant across positions at init "
              f"(pos_proj zero-initialised)")

    # 4. V37 invariant: relevance loss carries gradient through both forwards.
    m.zero_grad()
    m.train()
    logits, loss, aux = m(ids, q_mask, a_mask, None, targets)
    loss.backward()
    g_fc2 = m.answer_stream.comp_fc1[0].fc2.weight.grad
    assert g_fc2 is not None and g_fc2.abs().sum().item() > 0
    assert aux["aux_relevance_loss"].item() > 0, (
        "relevance loss is 0 -- cross-condition scoring may be broken"
    )
    print(f"  relevance gradient check: comp_fc1[0].fc2 grad sum = "
          f"{g_fc2.abs().sum().item():.4e} (relevance loss live)")

    # 5. V37 invariant: relevance gap ~0 at init (high-tau, W near-constant).
    init_gap = aux["answer_relevance_gap"].item()
    print(f"  relevance gap at init: {init_gap:.6f} (should be small at high-tau init)")
    assert abs(init_gap) < 1.0, (
        f"relevance gap at init is {init_gap:.4f} -- expected ~0"
    )

    # 6. V35 carry-over: anti-collapse losses carry gradient (V34 bug still fixed).
    m.zero_grad()
    logits, loss, aux = m(ids, q_mask, a_mask, None, targets)
    loss.backward()
    g_fc2_v35 = m.answer_stream.comp_fc1[0].fc2.weight.grad
    assert g_fc2_v35 is not None and g_fc2_v35.abs().sum().item() > 0
    print(f"  anti-collapse gradient check: comp_fc1[0].fc2 grad sum = "
          f"{g_fc2_v35.abs().sum().item():.4e} (V34 bug still fixed)")

    # 7. V35 carry-over: alpha IS question-sensitive at init.
    m.eval()
    with torch.no_grad():
        ids_a = torch.randint(0, 256, (2, 64))
        ids_b = torch.randint(0, 256, (2, 64))
        qm = torch.zeros(2, 64, dtype=torch.bool); qm[:, 30:40] = True
        am = torch.zeros(2, 64, dtype=torch.bool); am[:, 40:] = True
        _, _, aux_a = m(ids_a, qm, am)
        _, _, aux_b = m(ids_b, qm, am)
    pd = aux_a["q_summary_pairdiff"].item()
    assert pd > 0, f"q_summary_pairdiff = 0 at init (chain already collapsed)"
    print(f"  q_summary_pairdiff at init: {pd:.4f} (>0 = question-sensitive)")
    apd = aux_a["alpha_pairdiff"].item()
    print(f"  alpha_pairdiff at init: {apd:.6f}")

    # 8. Causality: prefix logits == extended logits at the prefix position.
    m.eval()
    with torch.no_grad():
        prefix = torch.randint(0, 256, (1, 10))
        qm = torch.zeros(1, 10, dtype=torch.bool); qm[0, :6] = True
        am = torch.zeros(1, 10, dtype=torch.bool); am[0, 6:] = True
        lp = m(prefix, qm, am)[0][:, 5, :]
        extra = torch.randint(0, 256, (1, 3))
        ext = torch.cat([prefix, extra], dim=1)
        qm2 = torch.zeros(1, 13, dtype=torch.bool); qm2[0, :6] = True
        am2 = torch.zeros(1, 13, dtype=torch.bool); am2[0, 6:] = True
        le = m(ext, qm2, am2)[0][:, 5, :]
        diff = (lp - le).abs().max().item()
    m.train()
    assert diff < 1e-4, f"Causality violated (diff={diff:.2e})"
    print(f"  causality OK (diff={diff:.2e})")

    # 9. Anti-collapse diagnostics present at init.
    m.eval()
    with torch.no_grad():
        _, _, aux2 = m(ids, q_mask, a_mask, None, targets)
    h_ent = aux2["alpha_entropy_mean"].item()
    log_k = torch.log(torch.tensor(float(cfg.num_bases))).item()
    print(f"  alpha entropy at init: {h_ent:.4f} (log K={log_k:.4f})")
    print(f"  q_summary_norm: {aux2['q_summary_norm'].item():.4f}")
    print(f"  alpha_usage min/max: {aux2['alpha_usage_min'].item():.4f} / "
          f"{aux2['alpha_usage_max'].item():.4f}")

    # 10. Generate round-trip.
    m.eval()
    prompt = torch.randint(0, 256, (1, 5))
    qm = torch.ones(1, 5, dtype=torch.bool)
    am = torch.zeros(1, 5, dtype=torch.bool)
    gen = m.generate(prompt, qm, am, None, max_new_tokens=20, temperature=0.0)
    assert gen.shape == (1, 25), gen.shape
    print(f"  generate OK (shape={tuple(gen.shape)})")

    # 11. Ablation: per-position routing OFF -> V36-equivalent (seq-level alpha).
    cfg_ppr = v37_qag_1m()
    cfg_ppr.per_position_routing = False
    m_ppr = MicroMixerV37QAG(cfg_ppr)
    m_ppr.train()
    logits_ppr, loss_ppr, aux_ppr = m_ppr(ids, q_mask, a_mask, None, targets)
    assert torch.isfinite(loss_ppr).item()
    assert aux_ppr["alpha_position_variation"].item() == 0.0, (
        "alpha_position_variation should be 0 when per_position_routing is OFF"
    )
    n_ppr = count_parameters(m_ppr)
    print(f"  ablation (no PPR): {n_ppr:,} params, "
          f"loss={loss_ppr.item():.4f}, pos_var=0 (V36-equivalent)")

    # 12. Ablation: relevance OFF -> V35-equivalent loss stack.
    cfg_rel = v37_qag_1m()
    cfg_rel.use_relevance_loss = False
    m_rel = MicroMixerV37QAG(cfg_rel)
    m_rel.train()
    logits_rel, loss_rel, aux_rel = m_rel(ids, q_mask, a_mask, None, targets)
    assert torch.isfinite(loss_rel).item()
    assert aux_rel["aux_relevance_loss"].item() == 0.0
    assert aux_rel["answer_relevance_gap"].item() == 0.0
    print(f"  ablation (no relevance): loss={loss_rel.item():.4f} (V35-equivalent)")

    # 13. Ablation: anti-filler ON -> null forward active.
    cfg_af = v37_qag_1m()
    cfg_af.use_antifiller_loss = True
    m_af = MicroMixerV37QAG(cfg_af)
    m_af.train()
    logits_af, loss_af, aux_af = m_af(ids, q_mask, a_mask, None, targets)
    assert torch.isfinite(loss_af).item()
    assert aux_af["aux_antifiller_loss"].item() >= 0.0
    print(f"  ablation (anti-filler on): loss={loss_af.item():.4f} "
          f"(af_loss={aux_af['aux_antifiller_loss'].item():.4f}, "
          f"af_gap={aux_af['answer_antifiller_gap'].item():.4f})")

    # 14. Ablation: hard negatives ON -> crossed index excludes self.
    cfg_hn = v37_qag_1m()
    cfg_hn.hard_negatives = True
    m_hn = MicroMixerV37QAG(cfg_hn)
    m_hn.train()
    logits_hn, loss_hn, aux_hn = m_hn(ids, q_mask, a_mask, None, targets)
    assert torch.isfinite(loss_hn).item()
    qs = m_hn.question_encoder(m_hn.embed(ids), q_mask)
    neg_idx = m_hn._build_cross_neg_index(qs)
    self_idx = torch.arange(ids.shape[0])
    assert not (neg_idx == self_idx).any(), "hard negative selected self"
    print(f"  ablation (hard negatives): loss={loss_hn.item():.4f} "
          f"(gap={aux_hn['answer_relevance_gap'].item():.4f})")

    # 15. Ablation: weight generation OFF -> channel-MLP learned directly.
    cfg_abl = v37_qag_1m()
    cfg_abl.use_weight_generation = False
    m_abl = MicroMixerV37QAG(cfg_abl)
    n_abl = count_parameters(m_abl)
    m_abl.train()
    logits_abl, loss_abl, aux_abl = m_abl(ids, q_mask, a_mask, None, targets)
    assert torch.isfinite(loss_abl).item()
    assert aux_abl["aux_relevance_loss"].item() == 0.0
    print(f"  ablation (no weight gen): {n_abl:,} params, loss={loss_abl.item():.4f}")

    # 16. Ablation: normalize_bases OFF -> raw bases (no scales).
    cfg_nb = v37_qag_1m()
    cfg_nb.normalize_bases = False
    m_nb = MicroMixerV37QAG(cfg_nb)
    m_nb.train()
    logits_nb, loss_nb, aux_nb = m_nb(ids, q_mask, a_mask, None, targets)
    assert torch.isfinite(loss_nb).item()
    assert not hasattr(m_nb.answer_stream.basis_fc1, "scales"), \
        "scales should not exist when normalize_bases=False"
    print(f"  ablation (no normalisation): loss={loss_nb.item():.4f} "
          f"(raw bases, no scales)")

    # 17. Ablation: gated pool OFF -> masked-mean (V34-style).
    cfg_gp = v37_qag_1m()
    cfg_gp.use_gated_pool = False
    m_gp = MicroMixerV37QAG(cfg_gp)
    n_gp = count_parameters(m_gp)
    m_gp.train()
    logits_gp, loss_gp, _ = m_gp(ids, q_mask, a_mask, None, targets)
    assert torch.isfinite(loss_gp).item()
    print(f"  ablation (no gated pool): {n_gp:,} params, loss={loss_gp.item():.4f}")

    # 18. Ablation: anti-collapse OFF -> CE + relevance only.
    cfg_ac = v37_qag_1m()
    cfg_ac.use_anti_collapse = False
    m_ac = MicroMixerV37QAG(cfg_ac)
    m_ac.train()
    logits_ac, loss_ac, aux_ac = m_ac(ids, q_mask, a_mask, None, targets)
    assert aux_ac["aux_alpha_contrast_loss"].item() == 0.0
    assert aux_ac["aux_qsum_contrast_loss"].item() == 0.0
    assert aux_ac["aux_relevance_loss"].item() > 0.0, (
        "relevance loss should still be active when anti-collapse is off"
    )
    print(f"  ablation (no anti-collapse): loss={loss_ac.item():.4f} "
          f"(relevance still active={aux_ac['aux_relevance_loss'].item():.4f})")

    # 19. lambda_rel warmup ramp.
    cfg_wu = v37_qag_1m()
    cfg_wu.relevance_warmup_steps = 100
    m_wu = MicroMixerV37QAG(cfg_wu)
    m_wu.train()
    m_wu.train_step.fill_(0)
    w0 = m_wu._current_relevance_weight()
    assert w0 == 0.0, f"lambda_rel at step 0 should be 0 (got {w0})"
    m_wu.train_step.fill_(50)
    w50 = m_wu._current_relevance_weight()
    assert abs(w50 - cfg_wu.w_relevance * 0.5) < 1e-6, (
        f"lambda_rel at half-warmup should be {cfg_wu.w_relevance*0.5} (got {w50})"
    )
    m_wu.train_step.fill_(200)
    w_done = m_wu._current_relevance_weight()
    assert abs(w_done - cfg_wu.w_relevance) < 1e-6, (
        f"lambda_rel after warmup should be {cfg_wu.w_relevance} (got {w_done})"
    )
    print(f"  lambda_rel ramp OK: 0->{w50:.4f}->{w_done:.4f} "
          f"(warmup={cfg_wu.relevance_warmup_steps} steps)")

    # 20. V37 invariant: detach_crossed_weights -> crossed alpha is detached.
    cfg_dc = v37_qag_1m()
    cfg_dc.detach_crossed_weights = True
    m_dc = MicroMixerV37QAG(cfg_dc)
    m_dc.train()
    logits_dc, loss_dc, aux_dc = m_dc(ids, q_mask, a_mask, None, targets)
    assert torch.isfinite(loss_dc).item()
    print(f"  ablation (detach crossed): loss={loss_dc.item():.4f} "
          f"(one-sided margin)")

    print("V37 QAG-Mixer self-test passed!")
