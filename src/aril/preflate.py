"""Preflate integration for loose Git objects."""

import re
import shutil
import subprocess
from pathlib import Path

from .transforms import replace_spans, restore_spans

MAX_OBJECT_SIZE = 8 << 20
SHARD = re.compile(r"[0-9a-f]{2}")
OBJECT = re.compile(r"(?:[0-9a-f]{38}|[0-9a-f]{62})")


def is_loose_object(path: Path, sample: bytes, size: int) -> bool:
    if not (
        2 <= size <= MAX_OBJECT_SIZE and len(sample) >= 2
        and SHARD.fullmatch(path.parent.name) and OBJECT.fullmatch(path.name)
    ):
        return False
    cmf, flags = sample[:2]
    return cmf & 15 == 8 and cmf >> 4 <= 7 and (cmf * 256 + flags) % 31 == 0


def find_helper() -> str | None:
    executable = shutil.which("aril-preflate")
    if executable:
        return executable
    local = Path(__file__).resolve().parents[2] / "tools" / "preflate" / "target" / "release" / "aril-preflate"
    return str(local) if local.is_file() else None


def require_helper() -> str:
    executable = find_helper()
    if executable is not None:
        return executable
    raise ValueError(
        "Preflate requires aril-preflate on PATH or a local release build "
        "(cargo build --release --locked --manifest-path tools/preflate/Cargo.toml)"
    )


def run_helper(executable: str, data: bytes, *args: str) -> bytes:
    try:
        result = subprocess.run(
            [executable, *args], input=data, capture_output=True, check=True, timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("aril-preflate could not process an ARIL loose object") from error
    return result.stdout


def preflate_candidate(data: bytes, spans: list[tuple[int, int]]) -> bytes | None:
    executable = require_helper()

    def encode(original: bytes) -> bytes | None:
        try:
            return run_helper(executable, original, "encode")
        except ValueError:
            return None

    return replace_spans(data, spans, encode)


def restore_preflate(payload: bytes, size: int) -> bytes:
    executable = require_helper()

    def decode(data: bytes, length: int) -> bytes:
        if length > MAX_OBJECT_SIZE:
            raise ValueError("ARIL Preflate object exceeds the size limit")
        return run_helper(executable, data, "decode", str(length))

    return restore_spans(payload, size, decode)
