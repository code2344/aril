"""Byte-exact JPEG recompression for grouped archives."""

import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

from .transforms import replace_spans, restore_spans


def require_tools() -> None:
    if not (shutil.which("cjxl") and shutil.which("djxl")):
        raise ValueError("--jpeg-xl requires the cjxl and djxl tools on PATH")


def run_tool(name: str, source: Path, output: Path) -> None:
    try:
        subprocess.run(
            [name, str(source), str(output), "--num_threads=1", *(("-e", "3") if name == "cjxl" else ())],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError(f"{name} could not process an ARIL JPEG payload") from error


def reconstruct(source: Path, output: Path, size: int) -> bytes:
    run_tool("djxl", source, output)
    if output.stat().st_size != size:
        raise ValueError("Reconstructed JPEG has the wrong length")
    return output.read_bytes()


def jpeg_candidate(data: bytes, spans: list[tuple[int, int]]) -> bytes | None:
    with TemporaryDirectory(prefix="aril-jpeg-") as directory:
        root = Path(directory)
        original, encoded, restored = (root / name for name in ("original.jpg", "image.jxl", "restored.jpg"))

        def encode(image: bytes) -> bytes | None:
            original.write_bytes(image)
            encoded.unlink(missing_ok=True)
            restored.unlink(missing_ok=True)
            try:
                run_tool("cjxl", original, encoded)
                if encoded.stat().st_size < len(image) and reconstruct(encoded, restored, len(image)) == image:
                    return encoded.read_bytes()
            except (OSError, ValueError):
                pass
            return None

        return replace_spans(data, spans, encode)


def restore_jpegs(payload: bytes, size: int) -> bytes:
    with TemporaryDirectory(prefix="aril-jpeg-") as directory:
        root = Path(directory)
        encoded, restored = root / "image.jxl", root / "restored.jpg"

        def decode(image: bytes, length: int) -> bytes:
            encoded.write_bytes(image)
            restored.unlink(missing_ok=True)
            return reconstruct(encoded, restored, length)

        return restore_spans(payload, size, decode)
