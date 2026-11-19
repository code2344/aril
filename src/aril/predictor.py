from functools import cache
from itertools import accumulate
from math import floor

import torch

REPO = "VeroLM/NoTokenLM-MicroGen"
REVISION = "c06f7ad1133b981c0c93634e141e530380cc44ab"
ALPHABET_SIZE = 256
FREQUENCY_TOTAL = 1 << 16

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


@cache
def microgen_model():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        REPO,
        revision=REVISION,
        subfolder="3.7",
        trust_remote_code=True,
    ).cpu().eval()


@torch.inference_mode()
def byte_cdf(prefix: bytes | bytearray) -> list[int]:
    if not prefix:
        # Both sides use a uniform table for the first byte.
        return list(range(0, FREQUENCY_TOTAL + 1, 256))

    model = microgen_model()
    window = prefix[-model.config.max_len:]
    input_ids = torch.tensor([list(window)], dtype=torch.long)
    logits = model(input_ids=input_ids).logits[0, -1]
    return logits_to_cdf(logits)


def logits_to_cdf(logits: torch.Tensor) -> list[int]:
    probabilities = logits.double().softmax(dim=-1).tolist()

    # Reserve one count per byte so every possible input remains encodable.
    budget = FREQUENCY_TOTAL - ALPHABET_SIZE
    scaled = [p * budget for p in probabilities]
    counts = [floor(value) for value in scaled]
    leftover = budget - sum(counts)

    # Assign spare counts by fractional remainder, breaking ties by byte ID.
    order = sorted(
        range(ALPHABET_SIZE),
        key=lambda i: (-(scaled[i] - counts[i]), i),
    )
    for i in order[:leftover]:
        counts[i] += 1

    return list(accumulate((count + 1 for count in counts), initial=0))
