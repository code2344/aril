"""ARIL's byte predictor for the pretrained MicroMixer-4 1M checkpoint."""

from functools import cache

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from ._micromixer.model_v87_final import MicroMixerV87Final, v87_final_1m
from .predictor import logits_to_cdf

REPO = "llaa33219/MicroMixer-4-1M-TinyStories"
REVISION = "20a5e09a37ac941f4e87c2bccee95b9c33df714c"
CONTEXT_SIZE = 256


@cache
def load_model() -> MicroMixerV87Final:
    checkpoint = hf_hub_download(REPO, "epoch_2.safetensors", revision=REVISION)
    model = MicroMixerV87Final(v87_final_1m())
    model.load_state_dict(load_file(checkpoint), strict=True)
    return model.eval()


@torch.inference_mode()
def byte_cdf(prefix: bytes | bytearray) -> list[int]:
    # BOS is present only until it falls out of the sliding context.
    context = list(prefix[-CONTEXT_SIZE:])
    if len(context) < CONTEXT_SIZE:
        context.insert(0, 1)
    logits, _, _ = load_model()(torch.tensor([context], dtype=torch.long))
    return logits_to_cdf(logits[0, -1])
