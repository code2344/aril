"""Versioned binary archives for regular files and directory trees."""

import os
import shutil
import stat
import struct
import zlib
from collections import Counter
from copy import deepcopy
from collections.abc import Callable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import BinaryIO

from .context_codec import ContextMixer, decode_context, encode_context
from .grouped import GROUP_MIB, MAX_GROUP_SIZE, GroupReader, GroupWriter, Status, file_family, frame_family, frame_order
from .jpeg import require_tools
from .preflate import find_helper, is_loose_object, require_helper
from .range_codec import decode_block, encode_block
from .solid import GROUP_SIZE, SolidReader, SolidWriter

# Each version fixes the model and frequency calculation for neural blocks.
MAGIC = b"ARIL\x01"
ARCHIVE_VERSIONS = {
    "fast": b"ARIL\x03",
    "solid": b"ARIL\x04",
    "grouped": b"ARIL\x0b",
    "context": b"ARIL\x05",
    "context-neural-full": b"ARIL\x06",
    "context-neural": b"ARIL\x09",
    "context-flat": b"ARIL\x07",
    "context-selective": b"ARIL\x08",
    "microgen": MAGIC,
    "micromixer": b"ARIL\x02",
}
CONTEXT_MODES = {
    "context", "context-neural", "context-neural-full", "context-flat", "context-selective",
}
NEURAL_CONTEXT_MODES = {"context-neural", "context-neural-full", "context-selective"}
ARCHIVE_MODELS = {magic: model for model, magic in ARCHIVE_VERSIONS.items()}
ARCHIVE_MODELS[b"ARIL\x0a"] = "grouped"
PREFLATE_MAGIC = b"ARIL\x0c"
ARCHIVE_MODELS[PREFLATE_MAGIC] = "grouped"
BLOCK_SIZE = 64 * 1024
ENTRY = struct.Struct(">BHQ")
BLOCK = struct.Struct(">BIII")
FILE, DIRECTORY, END = 1, 2, 0
RAW, NEURAL, DEFLATE, NEURAL_PREFIX, RAW_BYPASS = range(5)
PREFIX_SIZE = 128
NEURAL_BUDGET = 256
PREFIX_HEADER = struct.Struct(">H")
Progress = Callable[[int, int], None]


def context_model(mode: str) -> ContextMixer | None:
    if mode in CONTEXT_MODES:
        return ContextMixer(
            file_aware=mode != "context-flat",
            neural=mode in NEURAL_CONTEXT_MODES,
            selective=mode == "context-selective",
        )
    return None


def read_exact(stream: BinaryIO | SolidReader, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("Truncated ARIL archive")
    return data


def validate_path(name: str) -> None:
    parts = name.split("/")
    if any(part in ("", ".", "..") or any(c in part for c in "\\:\0") for part in parts):
        raise ValueError(f"Invalid archive path {name!r}")


def source_entries(path: Path, name: str, *, grouped: bool = False) -> Iterator[tuple[Path, str, int, int]]:
    validate_path(name)
    info = path.lstat()
    if stat.S_ISDIR(info.st_mode):
        yield path, name, DIRECTORY, 0
        children = path.iterdir()
        if grouped:
            ordered = sorted(children, key=lambda child: (file_family(child), frame_order(child.name)))
        else:
            ordered = sorted(children)
        for child in ordered:
            yield from source_entries(child, f"{name}/{child.name}", grouped=grouped)
    elif stat.S_ISREG(info.st_mode):
        yield path, name, FILE, info.st_size
    else:
        raise ValueError(f"Unsupported file type: {path}")


def deflate(data: bytes, dictionary: bytes = b"") -> bytes:
    compressor = zlib.compressobj(level=3, wbits=-15, zdict=dictionary)
    return compressor.compress(data) + compressor.flush()


def inflate(payload: bytes, size: int, dictionary: bytes = b"") -> bytes:
    try:
        decoder = zlib.decompressobj(wbits=-15, zdict=dictionary)
        data = decoder.decompress(payload, size + 1)
    except zlib.error as error:
        raise ValueError("Invalid ARIL DEFLATE payload") from error
    if len(data) != size or not decoder.eof or decoder.unused_data:
        raise ValueError("Invalid ARIL DEFLATE payload length")
    return data


def bypass_context(name: str, data: bytes, compressed: bytes) -> str | None:
    # Six-to-one DEFLATE already handles very repetitive blocks cheaply.
    if ".git" in name.split("/"):
        return "Git metadata"
    if b"\0" in data:
        return "Binary data"
    if len(compressed) * 6 <= len(data):
        return "Highly repetitive data"

    # A block may start or end in the middle of a UTF-8 code point.
    text = data.decode("utf-8", errors="replace")
    unusual = sum(
        not (char.isprintable() or char in "\t\n\r") or char == "\ufffd"
        for char in text
    )
    return "Binary data" if unusual * 20 > len(text) else None


def classify_group(path: Path, name: str, size: int, *, jpeg_xl: bool = False, preflate: bool = False) -> tuple[str, str]:
    if size == 0 or (".git" in path.parts and not preflate):
        return "solid", ""
    with path.open("rb") as stream:
        sample = stream.read(4096)
    if preflate and is_loose_object(path, sample, size):
        return "preflate", str(path.parent.parent)
    if ".git" in path.parts:
        return "solid", ""
    # Native and WebAssembly binaries benefit from an executable BCJ pass.
    macho = (b"\xfe\xed\xfa\xce", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xcf\xfa\xed\xfe")
    if (file_family(path) == "compiled" or sample.startswith((b"\x7fELF", b"MZ", b"\x00asm") + macho)) and size <= MAX_GROUP_SIZE:
        return "compiled", ""
    if jpeg_xl and file_family(path) == "jpeg" and size <= GROUP_SIZE and sample.startswith(b"\xff\xd8\xff"):
        return "jpeg", str(path.parent)
    frame = frame_family(name)
    if frame is not None and size <= BLOCK_SIZE and b"\0" not in sample:
        return "frames", frame
    if bypass_context(name, sample, deflate(sample)) is None:
        return "text", str(path.parent / file_family(path))
    return "solid", ""


def encode_context_block(
    data: bytes, name: str, model: str, context: ContextMixer,
    progress: Callable[[int], None] | None,
) -> tuple[int, bytes, ContextMixer, str]:
    if model != "context-neural":
        encoded = encode_context(data, context, progress)
        if len(encoded) < len(data):
            return NEURAL, encoded, context, "Neural payload"
        return RAW, data, context, "Raw payload"

    compressed = deflate(data)
    if len(compressed) < len(data):
        fallback_method, fallback = DEFLATE, compressed
    else:
        fallback_method, fallback = RAW_BYPASS, data
    bypass = bypass_context(name, data, compressed)
    if bypass is not None:
        return fallback_method, fallback, context, bypass

    # Only commit model updates if the decoder will use the same neural payload.
    trial = deepcopy(context)
    encoded = encode_context(data, trial, progress)
    if len(encoded) < len(fallback):
        return NEURAL, encoded, trial, "Neural payload"
    return fallback_method, fallback, context, "Conventional payload smaller or tied"


def encode_fast_block(
    data: bytes, budget: int, progress: Callable[[int], None] | None,
) -> tuple[int, bytes, int]:
    compressed = deflate(data)
    method, payload = (DEFLATE, compressed) if len(compressed) < len(data) else (RAW, data)
    prefix = data[:PREFIX_SIZE]
    # A bounded probe reserves inference for text-like bytes. Try the neural
    # prefix against the complete block, including its length field and suffix.
    if 1 < len(prefix) <= budget and prefix.isascii() and all(
        32 <= value < 127 or value in (9, 10, 13) for value in prefix
    ):
        budget -= len(prefix)
        neural = encode_block(prefix, "microgen", progress=progress)
        if len(data) == len(prefix):
            candidate_method, candidate = NEURAL, neural
        elif len(neural) < PREFIX_SIZE:
            candidate_method = NEURAL_PREFIX
            candidate = PREFIX_HEADER.pack(len(neural)) + neural + deflate(data[PREFIX_SIZE:], prefix)
        else:
            return method, payload, budget
        if len(candidate) < len(payload):
            method, payload = candidate_method, candidate
    return method, payload, budget


def create_archive(
    source: Path, output: Path, *, model: str | None = None, progress: Progress | None = None,
    stats: Counter[str] | None = None, status: Status | None = None,
    group_mib: int | None = None, jpeg_xl: bool = False, preflate: bool | None = None,
) -> None:
    if model is None:
        git_input = ".git" in source.absolute().parts or (source / ".git").is_dir()
        model = "grouped" if preflate is True or (preflate is not False and git_input and find_helper()) else "fast"
    if preflate is None:
        preflate = model == "grouped" and find_helper() is not None
    if group_mib is not None and group_mib not in GROUP_MIB:
        raise ValueError("Unsupported ARIL group size")
    if (group_mib is not None or jpeg_xl or preflate) and model != "grouped":
        raise ValueError("--group-size, --jpeg-xl, and --preflate require --model grouped")
    if jpeg_xl:
        require_tools()
    if preflate:
        require_helper()
    magic = PREFLATE_MAGIC if preflate else ARCHIVE_VERSIONS[model]
    source = source.absolute()
    output = output.absolute()
    if source == output.resolve() or (source.is_dir() and output.resolve().is_relative_to(source.resolve())):
        raise ValueError("Archive output must be outside the input")
    if source.is_file() and output.exists() and source.samefile(output):
        raise ValueError("Archive output must differ from the input file")

    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")

    entries = list(source_entries(source, source.name, grouped=model == "grouped"))
    total = sum(size for _, _, _, size in entries)
    completed = 0
    budget = NEURAL_BUDGET

    def report_block(processed: int) -> None:
        if progress is not None:
            progress(completed + processed, total)

    report_block(0)
    with TemporaryDirectory(dir=output.parent) as staging:
        temporary = Path(staging) / "archive"
        with temporary.open("wb") as archive:
            archive.write(magic)
            if model == "grouped":
                stream = GroupWriter(
                    archive, lambda done: progress(done, total) if progress is not None else None,
                    status, stats, (group_mib or 1) << 20,
                )
            else:
                stream = SolidWriter(archive) if model == "solid" else archive
            context = context_model(model)
            for path, name, kind, size in entries:
                encoded_name = name.encode("utf-8")
                if len(encoded_name) > 65535:
                    raise ValueError(f"Archive path is too long: {name}")
                transform = False
                block_size = BLOCK_SIZE
                if isinstance(stream, GroupWriter):
                    route, family = (
                        classify_group(path, name, size, jpeg_xl=jpeg_xl, preflate=preflate) if kind == FILE else ("solid", "")
                    )
                    stream.begin(name, route, family)
                    transform = route in ("frames", "jpeg", "preflate")
                    if route in ("jpeg", "preflate"):
                        block_size = size
                stream.write(ENTRY.pack(kind, len(encoded_name), size))
                stream.write(encoded_name)
                if kind == FILE:
                    if context is not None:
                        context.start_file(name)
                    with path.open("rb") as contents:
                        remaining = size
                        while remaining:
                            data = read_exact(contents, min(remaining, block_size))
                            callback = report_block if progress is not None else None
                            reason = None
                            if context is not None:
                                method, payload, context, reason = encode_context_block(
                                    data, name, model, context, callback,
                                )
                            elif model in ("solid", "grouped"):
                                method, payload = RAW, data
                            elif model == "fast":
                                method, payload, budget = encode_fast_block(data, budget, callback)
                            else:
                                neural = encode_block(data, model, progress=callback) if len(data) > 1 else data
                                method, payload = (NEURAL, neural) if len(neural) < len(data) else (RAW, data)
                            stream.write(BLOCK.pack(method, len(data), len(payload), zlib.crc32(data)))
                            if isinstance(stream, GroupWriter):
                                stream.write(payload, input_data=True, transform=transform)
                                remaining -= len(data)
                                continue
                            stream.write(payload)
                            if stats is not None:
                                if reason is None:
                                    reason = {
                                        RAW: "Raw payload", DEFLATE: "DEFLATE payload",
                                        NEURAL: "Neural payload", NEURAL_PREFIX: "Neural prefix",
                                    }[method]
                                stats[reason] += len(data)
                            report_block(len(data))
                            remaining -= len(data)
                            completed += len(data)
                        if contents.read(1):
                            raise ValueError(f"Input grew while archiving: {path}")
            stream.write(bytes([END]))
            if isinstance(stream, (SolidWriter, GroupWriter)):
                stream.finish()
        output.hardlink_to(temporary)


def read_entries(
    archive: BinaryIO, destination: Path | None = None, *,
    progress: Callable[[int], None] | None = None, status: Status | None = None,
) -> list[str]:
    model = ARCHIVE_MODELS.get(read_exact(archive, len(MAGIC)))
    if model is None:
        raise ValueError("Unsupported ARIL archive version")
    if model == "grouped":
        stream = GroupReader(archive, progress, status)
    else:
        stream = SolidReader(archive) if model == "solid" else archive

    entries: dict[str, int] = {}
    context = context_model(model) if destination is not None else None
    while (tag := read_exact(stream, 1)[0]) != END:
        kind, name_size, size = ENTRY.unpack(bytes([tag]) + read_exact(stream, ENTRY.size - 1))
        if kind not in (FILE, DIRECTORY) or (kind == DIRECTORY and size):
            raise ValueError("Invalid ARIL entry")
        name = read_exact(stream, name_size).decode("utf-8")
        validate_path(name)
        parent, _, _ = name.rpartition("/")
        if name in entries or (parent and entries.get(parent) != DIRECTORY) or (not parent and entries):
            raise ValueError(f"Unexpected archive entry {name!r}")
        entries[name] = kind
        if isinstance(stream, GroupReader):
            stream.name = name
        if progress is not None:
            progress(stream.tell())

        if kind == DIRECTORY:
            if destination is not None:
                (destination / name).mkdir()
            continue

        if destination is None:
            read_file(stream, size, model)
        else:
            if context is not None:
                context.start_file(name)
            with (destination / name).open("xb") as output:
                read_file(stream, size, model, output, progress=progress, context=context)

    if not entries or stream.read(1):
        raise ValueError("Invalid ARIL archive ending")
    return list(entries)


def read_file(
    stream: BinaryIO | SolidReader, remaining: int, model: str, output: BinaryIO | None = None, *,
    progress: Callable[[int], None] | None = None, context: ContextMixer | None = None,
) -> None:
    block_limit = MAX_GROUP_SIZE if model == "grouped" else BLOCK_SIZE
    while remaining:
        method, size, stored_size, checksum = BLOCK.unpack(read_exact(stream, BLOCK.size))
        valid_method = method == RAW or (model not in ("solid", "grouped") and method == NEURAL) or (
            model == "fast" and method in (DEFLATE, NEURAL_PREFIX)
        ) or (model == "context-neural" and method in (DEFLATE, RAW_BYPASS))
        if not (
            valid_method and 0 < size <= min(block_limit, remaining)
            and (stored_size == size if method in (RAW, RAW_BYPASS) else 0 < stored_size < size)
        ):
            raise ValueError("Invalid ARIL block")
        start = stream.tell() if progress is not None else 0
        payload = read_exact(stream, stored_size)
        if output is not None:
            def report_block(processed: int) -> None:
                # Account for archive bytes proportionally as the block is decoded.
                if progress is not None:
                    progress(start + processed * stored_size // size)

            callback = report_block if progress is not None else None
            if method == NEURAL and context is not None:
                data = decode_context(payload, size, context, callback)
            elif method == NEURAL:
                if model == "fast" and size > PREFIX_SIZE:
                    raise ValueError("Invalid ARIL neural block length")
                data = decode_block(payload, size, "microgen" if model == "fast" else model, progress=callback)
            elif method == NEURAL_PREFIX:
                if size <= PREFIX_SIZE or len(payload) <= PREFIX_HEADER.size:
                    raise ValueError("Invalid ARIL neural prefix")
                (length,) = PREFIX_HEADER.unpack_from(payload)
                end = PREFIX_HEADER.size + length
                if not 0 < length < PREFIX_SIZE or end >= len(payload):
                    raise ValueError("Invalid ARIL neural prefix length")
                prefix = decode_block(payload[PREFIX_HEADER.size:end], PREFIX_SIZE, "microgen", progress=callback)
                data = prefix + inflate(payload[end:], size - PREFIX_SIZE, prefix)
            elif method == DEFLATE:
                data = inflate(payload, size)
            else:
                data = payload
                if context is not None and method == RAW:
                    context.replay(data)
            if zlib.crc32(data) != checksum:
                raise ValueError("ARIL block checksum mismatch")
            output.write(data)
            if progress is not None:
                progress(stream.tell())
        remaining -= size


def list_archive(source: Path) -> list[str]:
    with source.open("rb") as stream:
        return read_entries(stream)


def extract_archive(
    source: Path, output: Path, *, progress: Progress | None = None, status: Status | None = None,
) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Output already exists: {output}")

    with TemporaryDirectory(dir=output.parent) as staging:
        destination = Path(staging)
        with source.open("rb") as stream:
            total = os.fstat(stream.fileno()).st_size

            high_water = 0

            def report(processed: int) -> None:
                nonlocal high_water
                if progress is not None and processed >= high_water:
                    high_water = processed
                    progress(processed, total)

            report(0)
            entries = read_entries(stream, destination, progress=report if progress else None, status=status)
        root = destination / entries[0]
        if root.is_dir():
            output.mkdir()
            try:
                for child in root.iterdir():
                    child.rename(output / child.name)
            except OSError:
                shutil.rmtree(output)
                raise
        else:
            # Linking publishes the complete file without overwriting an existing path.
            output.hardlink_to(root)
        report(total)
