"""Reversible replacement of bounded byte spans in a solid group."""

import struct
from collections.abc import Callable

RECORD = struct.Struct(">III")
COUNT = struct.Struct(">I")


def replace_spans(
    data: bytes, spans: list[tuple[int, int]], encode: Callable[[bytes], bytes | None],
) -> bytes | None:
    records = bytearray()
    body = bytearray()
    previous = 0
    for offset, length in spans:
        payload = encode(data[offset:offset + length])
        if not payload or len(payload) >= length:
            continue
        records.extend(RECORD.pack(offset, length, len(payload)))
        body.extend(data[previous:offset])
        body.extend(payload)
        previous = offset + length
    if not records:
        return None
    return COUNT.pack(len(records) // RECORD.size) + records + body + data[previous:]


def restore_spans(payload: bytes, size: int, decode: Callable[[bytes, int], bytes]) -> bytes:
    if len(payload) < COUNT.size:
        raise ValueError("Invalid ARIL transform table")
    (count,) = COUNT.unpack_from(payload)
    start = COUNT.size + count * RECORD.size
    if not count or start > len(payload):
        raise ValueError("Invalid ARIL transform table size")
    restored = bytearray()
    cursor = start
    for offset, length, stored in RECORD.iter_unpack(payload[COUNT.size:start]):
        gap = offset - len(restored)
        end = cursor + gap + stored
        if gap < 0 or not 0 < stored < length <= size - offset or end > len(payload):
            raise ValueError("Invalid ARIL transform span")
        restored.extend(payload[cursor:cursor + gap])
        data = decode(payload[cursor + gap:end], length)
        if len(data) != length:
            raise ValueError("Invalid ARIL reconstructed span size")
        restored.extend(data)
        cursor = end
    if len(restored) + len(payload) - cursor != size:
        raise ValueError("Invalid ARIL transform output size")
    restored.extend(payload[cursor:])
    return bytes(restored)
