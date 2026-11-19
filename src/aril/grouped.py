"""Solid groups with reversible frame deltas and neural candidate selection."""

import lzma
import re
import struct
from collections import Counter
from collections.abc import Callable
from pathlib import PurePosixPath
from typing import BinaryIO
from zlib import crc32

from .context_codec import ContextMixer, decode_context, encode_context
from .jpeg import jpeg_candidate, restore_jpegs
from .preflate import preflate_candidate, restore_preflate
from .transforms import RECORD
from .solid import GROUP_SIZE, MEMORY_LIMIT, SolidReader

HEADER = struct.Struct(">BIII")
SPAN = struct.Struct(">II")
RAW, XZ, FRAME_DELTA, NEURAL, JPEG_XL, PREFLATE, COMPILED = range(1, 8)
GROUP_MIB = (1, 4, 8)
MAX_GROUP_SIZE = max(GROUP_MIB) << 20
TEXT_GROUP_SIZE = 256 << 10
FRAME_NAME = re.compile(r"(.*?)(\d+)(\.[^.]+(?:\.txt)?)$")
Status = Callable[[str, str, int], None]
CODE_FAMILIES = {
    extension: family
    for family, extensions in (
        ("c-family", (".c", ".h", ".cc", ".cpp", ".cxx", ".hh", ".hpp", ".hxx")),
        ("javascript", (".js", ".mjs", ".cjs", ".jsx", ".ts", ".tsx")),
        ("python", (".py", ".pyi", ".pyw")),
        ("assembly", (".asm", ".s", ".inc")),
        ("compiled", (".exe", ".dll", ".so", ".dylib", ".o", ".a", ".wasm")),
        ("jpeg", (".jpg", ".jpeg", ".jpe")),
    )
    for extension in extensions
}


def file_family(path: PurePosixPath) -> str:
    suffix = path.suffix.lower()
    return CODE_FAMILIES.get(suffix, suffix)


def frame_order(name: str) -> tuple[str, int, str]:
    match = FRAME_NAME.fullmatch(name)
    if match is None:
        return name, 0, name
    return match[1] + match[3], int(match[2]), name


def frame_family(name: str) -> str | None:
    path = PurePosixPath(name)
    match = FRAME_NAME.fullmatch(path.name)
    if path.suffix != ".txt" or match is None or "frame" not in path.parent.name.lower():
        return None
    return str(path.parent / (match[1] + match[3]))


def frame_delta(data: bytes, spans: list[tuple[int, int]]) -> bytes:
    output = bytearray(data)
    for (previous, previous_size), (current, size) in zip(spans, spans[1:]):
        if size == previous_size:
            output[current:current + size] = bytes(a ^ b for a, b in zip(
                data[previous:previous + size], data[current:current + size],
            ))
    return struct.pack(">I", len(spans)) + b"".join(SPAN.pack(*span) for span in spans) + output


def undo_frame_delta(payload: bytes, size: int) -> bytes:
    if len(payload) < 4:
        raise ValueError("Invalid ARIL frame table")
    (count,) = struct.unpack_from(">I", payload)
    start = 4 + count * SPAN.size
    if count < 2 or start + size != len(payload):
        raise ValueError("Invalid ARIL frame table size")
    data = bytearray(payload[start:])
    previous = previous_size = 0
    for offset, length in SPAN.iter_unpack(payload[4:start]):
        if offset < previous + previous_size or not 0 < length <= size - offset:
            raise ValueError("Invalid ARIL frame span")
        if length == previous_size:
            data[offset:offset + length] = bytes(a ^ b for a, b in zip(
                data[previous:previous + length], data[offset:offset + length],
            ))
        previous, previous_size = offset, length
    return bytes(data)


class GroupWriter:
    def __init__(
        self, stream: BinaryIO, progress: Callable[[int], None],
        status: Status | None, stats: Counter[str] | None, group_size: int = GROUP_SIZE,
    ) -> None:
        self.stream = stream
        self.progress = progress
        self.status = status
        self.stats = stats
        self.group_size = group_size
        self.buffer = bytearray()
        self.spans: list[tuple[int, int]] = []
        self.route = "solid"
        self.family = ""
        self.name = ""
        self.previous_name = ""
        self.input_size = 0
        self.completed = 0

    def begin(self, name: str, route: str, family: str) -> None:
        if (route, family) != (self.route, self.family):
            self.flush()
        self.route, self.family, self.name = route, family, name

    def write(self, data: bytes, *, input_data: bool = False, transform: bool = False) -> None:
        limit = TEXT_GROUP_SIZE if self.route == "text" else self.group_size
        if self.route == "preflate" and transform and len(data) > limit:
            self.flush()
            limit = len(data)
        if len(self.buffer) + len(data) > limit:
            self.flush()
        if transform:
            self.spans.append((len(self.buffer), len(data)))
        self.buffer.extend(data)
        if input_data:
            self.input_size += len(data)
        self.previous_name = self.name

    def compress_solid(self, data: bytes) -> bytes:
        return lzma.compress(data, filters=[{
            "id": lzma.FILTER_LZMA2, "preset": 4, "dict_size": self.group_size,
        }])

    def compress_compiled(self, data: bytes) -> bytes:
        # BCJ exposes relative-call/jump structure before LZMA2.  The BCJ
        # filter is reversible and the filter id is carried in the XZ stream.
        kind = lzma.FILTER_X86
        if data.startswith(b"\x7fELF"):
            kind = lzma.FILTER_X86
        elif data.startswith(b"MZ"):
            kind = lzma.FILTER_X86
        return lzma.compress(data, filters=[{"id": kind}, {
            "id": lzma.FILTER_LZMA2, "preset": 4, "dict_size": self.group_size,
        }])

    def flush(self) -> None:
        if not self.buffer:
            return
        original = bytes(self.buffer)
        if self.status is not None:
            self.status("Solid LZMA2", self.previous_name, self.input_size)
        compressed = self.compress_solid(original)
        method, payload = (XZ, compressed) if len(compressed) < len(original) else (RAW, original)
        if self.route == "frames" and len(self.spans) >= 2:
            if self.status is not None:
                self.status("Frame delta trial", self.previous_name, self.input_size)
            delta = self.compress_solid(frame_delta(original, self.spans))
            if len(delta) < len(payload):
                method, payload = FRAME_DELTA, delta
        if self.route in ("jpeg", "preflate"):
            kind, label, transform = (
                (JPEG_XL, "JPEG XL", jpeg_candidate) if self.route == "jpeg"
                else (PREFLATE, "Preflate", preflate_candidate)
            )
            if self.status is not None:
                self.status(f"{label} trial", self.previous_name, self.input_size)
            transformed = transform(original, self.spans) if self.spans else None
            if transformed is not None:
                candidate = self.compress_solid(transformed)
                if len(candidate) < len(payload):
                    method, payload = kind, candidate
        if self.route == "compiled":
            if self.status is not None:
                self.status("BCJ + LZMA2 trial", self.previous_name, self.input_size)
            candidate = self.compress_compiled(original)
            if len(candidate) < len(payload):
                method, payload = COMPILED, candidate
        if self.route == "text":
            if self.status is not None:
                self.status("Neural trial", self.previous_name, self.input_size)

            def report(processed: int) -> None:
                self.progress(self.completed + processed * self.input_size // len(original))

            coder = ContextMixer(file_aware=False, neural=True)
            neural = encode_context(original, coder, report if self.status is not None else None)
            if len(neural) < len(payload):
                method, payload = NEURAL, neural
        self.completed += self.input_size
        self.progress(self.completed)

        self.stream.write(HEADER.pack(method, len(original), len(payload), crc32(original)))
        self.stream.write(payload)
        if self.stats is not None and self.input_size:
            label = {RAW: "Grouped raw", XZ: "Solid LZMA2", FRAME_DELTA: "Frame delta",
                     NEURAL: "Neural group", JPEG_XL: "JPEG XL group", PREFLATE: "Preflate group",
                     COMPILED: "BCJ + LZMA2 group"}[method]
            self.stats[label] += self.input_size
        self.buffer.clear()
        self.spans.clear()
        self.input_size = 0

    def finish(self) -> None:
        self.flush()
        self.stream.write(HEADER.pack(0, 0, 0, 0))


class GroupReader(SolidReader):
    def __init__(
        self, stream: BinaryIO, progress: Callable[[int], None] | None = None,
        status: Status | None = None,
    ) -> None:
        super().__init__(stream)
        self.progress = progress
        self.status = status
        self.name = ""

    def _read_group(self) -> None:
        header = self.stream.read(HEADER.size)
        if len(header) != HEADER.size:
            raise ValueError("Truncated ARIL group header")
        method, size, stored, checksum = HEADER.unpack(header)
        if (method, size, stored, checksum) == (0, 0, 0, 0):
            if self.stream.read(1):
                raise ValueError("Trailing data after ARIL groups")
            self.finished = True
            return
        if not (
            method in (RAW, XZ, FRAME_DELTA, NEURAL, JPEG_XL, PREFLATE, COMPILED) and 0 < stored <= size <= MAX_GROUP_SIZE
            and (stored == size if method == RAW else stored < size)
            and (method != NEURAL or size <= TEXT_GROUP_SIZE)
        ):
            raise ValueError("Invalid ARIL group")
        self.payload_start = self.stream.tell()
        self.stored_size = stored
        if self.progress is not None:
            self.progress(self.payload_start)
        if self.status is not None:
            route = {RAW: "Raw", XZ: "Solid LZMA2", FRAME_DELTA: "Frame delta", NEURAL: "Neural", JPEG_XL: "JPEG XL", PREFLATE: "Preflate", COMPILED: "BCJ + LZMA2"}[method]
            self.status(route, self.name or "First group", stored)
        payload = self.stream.read(stored)
        if len(payload) != stored:
            raise ValueError("Truncated ARIL group payload")

        if method == RAW:
            data = payload
        elif method == NEURAL:
            def report(done: int) -> None:
                if self.progress is not None:
                    self.progress(self.payload_start + done * stored // size)

            data = decode_context(
                payload, size, ContextMixer(file_aware=False, neural=True),
                report if self.progress is not None else None,
            )
        else:
            decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=MEMORY_LIMIT)
            # A span describes at least one byte and costs eight bytes in the table.
            table_width = {FRAME_DELTA: SPAN.size, JPEG_XL: RECORD.size, PREFLATE: RECORD.size}.get(method, 0)
            limit = size * (1 + table_width) + (4 if table_width else 0)
            try:
                data = decoder.decompress(payload, max_length=limit + 1)
            except lzma.LZMAError as error:
                raise ValueError("Invalid ARIL group payload") from error
            if len(data) > limit or not decoder.eof or decoder.unused_data:
                raise ValueError("Invalid ARIL group payload length")
            if method == FRAME_DELTA:
                data = undo_frame_delta(data, size)
            elif method == JPEG_XL:
                data = restore_jpegs(data, size)
            elif method == PREFLATE:
                data = restore_preflate(data, size)
        if len(data) != size or crc32(data) != checksum:
            raise ValueError("ARIL group checksum or length mismatch")
        self.buffer = data
        self.position = 0
