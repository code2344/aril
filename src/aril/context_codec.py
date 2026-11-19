"""Online, integer-only bit prediction for file-aware archives."""

from array import array
from bisect import bisect_left
from collections.abc import Callable, Iterable
from pathlib import PurePosixPath

from .range_codec import decode_symbols, encode_symbols

TOTAL = 256
TABLE_SIZE = 1 << 16
MASK = TABLE_SIZE - 1
WINDOW_SIZE = 1 << 18
WINDOW_MASK = WINDOW_SIZE - 1
CODE_SUFFIXES = frozenset({
    ".c", ".cpp", ".css", ".go", ".h", ".html", ".java", ".js", ".json",
    ".py", ".rs", ".sh", ".toml", ".ts", ".tsx", ".xml", ".yaml", ".yml",
})
TEXT_SUFFIXES = frozenset({".md", ".rst", ".txt"})


def log2_fixed(value: int) -> int:
    exponent = value.bit_length() - 1
    base = 1 << exponent
    return exponent * 32 + (value - base) * 32 // base


LOG_ODDS = tuple(log2_fixed(p) - log2_fixed(TOTAL - p) for p in range(1, TOTAL))


def bits_of(data: bytes) -> Iterable[int]:
    for symbol in data:
        for shift in range(7, -1, -1):
            yield (symbol >> shift) & 1


class ContextMixer:
    def __init__(self, *, file_aware: bool, neural: bool, selective: bool = False) -> None:
        self.file_aware = file_aware
        self.neural = neural
        self.selective = selective
        self.counts = [bytearray([128]) * TABLE_SIZE for _ in range(4)]
        self.match_keys = array("I", [0]) * TABLE_SIZE
        self.match_positions = array("Q", [0]) * TABLE_SIZE
        self.window = bytearray(WINDOW_SIZE)
        self.position = 0
        self.match_position = -1
        self.match_run = 0
        self.weights = [[256] * 5 for _ in range(24)]
        self.history = 0
        self.seen = 0
        self.prefix = 1
        self.kind = 0
        self.column = 0
        self.indent = 0
        self.leading = True
        self.guess = -1

    def start_file(self, path: str) -> None:
        if self.file_aware:
            suffix = PurePosixPath(path).suffix.lower()
            self.kind = 1 if suffix in CODE_SUFFIXES else 2 if suffix in TEXT_SUFFIXES else 0
            self.history = self.seen = self.column = self.indent = 0
            self.leading = True
            self.match_position = -1
            self.match_run = 0

    def frequencies(self) -> tuple[int, int, int]:
        prefix = self.prefix
        last = self.history & 255
        pair = self.history & 65535
        if prefix == 1:
            if self.match_position < 0 and self.seen >= 4:
                slot = ((self.history ^ (self.history >> 16)) * 40503) & MASK
                previous = self.match_positions[slot] - 1
                if (
                    previous >= max(0, self.position - WINDOW_SIZE)
                    and self.match_keys[slot] == self.history
                ):
                    self.match_position = previous
            self.guess = self.window[self.match_position & WINDOW_MASK] if self.match_position >= 0 else -1

        # The fourth expert learns from line position and indentation in each file type.
        location = min(self.column, 31) if self.file_aware else 0
        style = self.kind * 97 + location * 7 + min(self.indent, 15)
        self.slots = (
            prefix,
            (last * 257 + prefix) & MASK,
            (pair * 40503 + prefix) & MASK,
            (style * 257 + last * 17 + prefix) & MASK,
        )
        shift = 8 - prefix.bit_length()
        match = self.guess >= 0 and ((256 + self.guess) >> (shift + 1)) == prefix
        if self.selective and match and self.match_run >= 8:
            # Use the match directly while it continues to explain the decoded bits.
            self.active = None
            self.probability = 254 if (self.guess >> shift) & 1 else 2
            return 0, TOTAL - self.probability, TOTAL

        probabilities = [table[index] for table, index in zip(self.counts, self.slots)]
        confidence = 240 + min(self.match_run, 14)
        probabilities.append((confidence if (self.guess >> shift) & 1 else TOTAL - confidence) if match else 128)

        self.features = tuple(LOG_ODDS[p - 1] for p in probabilities)
        self.active = self.weights[self.kind * 8 + prefix.bit_length() - 1] if self.neural else None
        if self.active is not None:
            logit = sum(w * x for w, x in zip(self.active, self.features)) // (256 * len(self.features))
        else:
            logit = sum(self.features) // len(self.features)
        self.probability = min(255, bisect_left(LOG_ODDS, logit) + 1)
        return 0, TOTAL - self.probability, TOTAL

    def observe(self, bit: int) -> None:
        target = bit * TOTAL
        for table, index in zip(self.counts, self.slots):
            previous = table[index]
            # Bounded estimates leave every symbol encodable.
            error = target - previous
            step = max(1, error // 16) if bit else min(-1, -((-error) // 16))
            table[index] = min(255, max(1, previous + step))

        if self.active is not None:
            error = target - self.probability
            for i, feature in enumerate(self.features):
                delta = error * feature
                step = (abs(delta) + 2048) // 4096
                weight = self.active[i] + (step if delta >= 0 else -step)
                self.active[i] = min(1024, max(-512, weight))

        self.prefix = self.prefix * 2 + bit
        if self.prefix < 256:
            return
        symbol = self.prefix - 256
        if self.seen >= 4:
            slot = ((self.history ^ (self.history >> 16)) * 40503) & MASK
            self.match_keys[slot] = self.history
            self.match_positions[slot] = self.position + 1
        if symbol == self.guess:
            self.match_position += 1
            self.match_run += 1
        else:
            self.match_position = -1
            self.match_run = 0
        self.window[self.position & WINDOW_MASK] = symbol
        self.position += 1
        self.history = ((self.history << 8) | symbol) & 0xFFFFFFFF
        self.seen = min(self.seen + 1, 4)
        self.prefix = 1
        if self.file_aware:
            if symbol == 10:
                self.column = self.indent = 0
                self.leading = True
            else:
                self.column = min(self.column + 1, 31)
                if self.leading and symbol in (9, 32):
                    self.indent = min(self.indent + 1, 15)
                else:
                    self.leading = False

    def replay(self, data: bytes) -> None:
        for bit in bits_of(data):
            self.frequencies()
            self.observe(bit)


def encode_context(
    data: bytes, predictor: ContextMixer, progress: Callable[[int], None] | None,
) -> bytes:
    def report(bits: int) -> None:
        if progress is not None and bits % 8 == 0:
            progress(bits // 8)

    return encode_symbols(bits_of(data), predictor, progress=report if progress else None)


def decode_context(
    payload: bytes, size: int, predictor: ContextMixer, progress: Callable[[int], None] | None,
) -> bytes:
    data = bytearray()
    byte = 1
    for bit in decode_symbols(payload, size * 8, predictor):
        byte = byte * 2 + bit
        if byte >= 256:
            data.append(byte - 256)
            byte = 1
            if progress is not None:
                progress(len(data))
    return bytes(data)
