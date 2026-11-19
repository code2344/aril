"""32-bit arithmetic coding for independently decodable archive blocks."""

from bisect import bisect_right
from collections.abc import Callable, Iterable, Sequence
from typing import Protocol

MASK = (1 << 32) - 1
HALF = 1 << 31
QUARTER = HALF >> 1
THREE_QUARTERS = HALF + QUARTER


class SymbolPredictor(Protocol):
    def frequencies(self) -> Sequence[int]: ...
    def observe(self, symbol: int) -> None: ...


class BlockPredictor:
    def __init__(self, model: str = "microgen") -> None:
        if model == "micromixer":
            from .micromixer import byte_cdf
        else:
            from .predictor import byte_cdf

        self.byte_cdf = byte_cdf
        self.context = bytearray()
        self.matches: dict[bytes, int] = {}

    def frequencies(self) -> list[int]:
        cdf = self.byte_cdf(self.context)
        guess = self.matches.get(bytes(self.context[-4:]))
        if guess is None:
            return cdf
        # Mix the neural distribution with a prediction from a repeated context.
        bonus = 3 * cdf[-1]
        return [value + (bonus if index > guess else 0) for index, value in enumerate(cdf)]

    def observe(self, symbol: int) -> None:
        if len(self.context) >= 4:
            self.matches[bytes(self.context[-4:])] = symbol
        self.context.append(symbol)


def encode_block(
    data: bytes, model: str = "microgen", *, progress: Callable[[int], None] | None = None
) -> bytes:
    return encode_symbols(data, BlockPredictor(model), progress=progress)


def encode_symbols(
    symbols: Iterable[int], predictor: SymbolPredictor, *,
    progress: Callable[[int], None] | None = None,
) -> bytes:
    low, high = 0, MASK
    pending = 0
    payload = bytearray()
    byte = used = 0

    def write_bit(bit: int) -> None:
        nonlocal byte, used
        byte = (byte << 1) | bit
        used += 1
        if used == 8:
            payload.append(byte)
            byte = used = 0

    def emit(bit: int) -> None:
        nonlocal pending
        write_bit(bit)
        for _ in range(pending):
            write_bit(bit ^ 1)
        pending = 0

    for processed, symbol in enumerate(symbols, start=1):
        cdf = predictor.frequencies()
        width = high - low + 1
        total = cdf[-1]
        high = low + width * cdf[symbol + 1] // total - 1
        low += width * cdf[symbol] // total
        while True:
            if high < HALF:
                emit(0)
            elif low >= HALF:
                emit(1)
                low -= HALF
                high -= HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                pending += 1
                low -= QUARTER
                high -= QUARTER
            else:
                break
            low <<= 1
            high = (high << 1) | 1
        predictor.observe(symbol)
        if progress is not None:
            progress(processed)

    pending += 1
    emit(int(low >= QUARTER))
    if used:
        payload.append(byte << (8 - used))
    return bytes(payload)


def decode_block(
    payload: bytes, size: int, model: str = "microgen", *, progress: Callable[[int], None] | None = None
) -> bytes:
    return bytes(decode_symbols(payload, size, BlockPredictor(model), progress=progress))


def decode_symbols(
    payload: bytes, size: int, predictor: SymbolPredictor, *,
    progress: Callable[[int], None] | None = None,
) -> Iterable[int]:
    bits = (byte >> shift & 1 for byte in payload for shift in range(7, -1, -1))
    code = 0
    for _ in range(32):
        code = (code << 1) | next(bits, 0)

    low, high = 0, MASK
    for processed in range(1, size + 1):
        cdf = predictor.frequencies()
        width = high - low + 1
        total = cdf[-1]
        position = ((code - low + 1) * total - 1) // width
        symbol = bisect_right(cdf, position) - 1
        high = low + width * cdf[symbol + 1] // total - 1
        low += width * cdf[symbol] // total
        while True:
            if high < HALF:
                offset = 0
            elif low >= HALF:
                offset = HALF
            elif low >= QUARTER and high < THREE_QUARTERS:
                offset = QUARTER
            else:
                break
            low = (low - offset) << 1
            high = ((high - offset) << 1) | 1
            code = ((code - offset) << 1) | next(bits, 0)
        predictor.observe(symbol)
        yield symbol
        if progress is not None:
            progress(processed)
