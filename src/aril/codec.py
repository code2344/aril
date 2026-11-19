from bisect import bisect_right
from collections.abc import Callable
from fractions import Fraction


def narrow(low, high, cdf, symbol):
    width = high - low
    total = cdf[-1]
    return (
        low + width * Fraction(cdf[symbol], total),
        low + width * Fraction(cdf[symbol + 1], total),
    )


def shortest_decimal_prefix(low, high):
    """Return the shortest decimal prefix whose range fits in [low, high)."""
    digits = 0
    scale = 1

    while True:
        # Round up to the first decimal grid point at or above low.
        prefix = -(-low.numerator * scale // low.denominator)

        if Fraction(prefix + 1, scale) <= high:
            return prefix, digits

        digits += 1
        scale *= 10


def compress(
    data: bytes, *, progress: Callable[[int, int], None] | None = None
) -> tuple[int, int, int]:
    from .predictor import byte_cdf

    low, high = Fraction(0), Fraction(1)
    context = bytearray()

    if progress is not None:
        progress(0, len(data))
    for processed, symbol in enumerate(data, start=1):
        cdf = byte_cdf(context)
        low, high = narrow(low, high, cdf, symbol)
        context.append(symbol)
        if progress is not None:
            progress(processed, len(data))

    prefix, digits = shortest_decimal_prefix(low, high)
    return prefix, digits, len(data)


def decompress(
    prefix: int, digits: int, byte_count: int, *, progress: Callable[[int, int], None] | None = None
) -> bytes:
    from .predictor import byte_cdf

    code = Fraction(prefix, 10**digits)
    low, high = Fraction(0), Fraction(1)
    decoded = bytearray()

    if progress is not None:
        progress(0, byte_count)
    for processed in range(1, byte_count + 1):
        cdf = byte_cdf(decoded)

        # Map the code into the frequency table's coordinate system.
        position = (code - low) * cdf[-1] / (high - low)
        symbol = bisect_right(cdf, position) - 1

        low, high = narrow(low, high, cdf, symbol)
        decoded.append(symbol)
        if progress is not None:
            progress(processed, byte_count)

    return bytes(decoded)
