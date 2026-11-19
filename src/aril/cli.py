import argparse
import gzip
import sys
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from time import monotonic

from .archive import ARCHIVE_VERSIONS, Progress, create_archive, extract_archive, list_archive
from .codec import compress, decompress
from .grouped import GROUP_MIB, Status


@contextmanager
def progress_bar(description: str) -> Iterator[tuple[Progress | None, Status | None]]:
    if not sys.stderr.isatty():
        yield None, None
        return

    from tqdm import tqdm

    with tqdm(desc=description, unit="B", unit_scale=True, miniters=1, file=sys.stderr) as bar, ExitStack() as stack:
        details = None
        route = path = ""
        start = refreshed = monotonic()
        offset = group_size = 0
        totals: Counter[str] = Counter()
        elapsed: Counter[str] = Counter()

        def update_label(now: float) -> None:
            count = max(0, bar.n - offset)
            seconds = elapsed[route] + now - start
            rate = (totals[route] + count) / seconds if seconds > 0 else 0
            remaining = max(0, group_size - count)
            estimate = f", group ETA {remaining / rate:.0f}s" if rate > 0 and remaining else ""
            speed = f", {rate / 1024:,.1f} KiB/s" if rate > 0 else ""
            if details is not None:
                details.set_description_str(f"{route} · {path}{speed}{estimate}")

        def status(new_route: str, name: str, size: int) -> None:
            nonlocal details, route, path, start, refreshed, offset, group_size
            now = monotonic()
            if route:
                totals[route] += max(0, bar.n - offset)
                elapsed[route] += now - start
            route = new_route
            path = "".join(char if char.isprintable() else "?" for char in name[-64:])
            start = refreshed = now
            offset = bar.n
            group_size = size
            if details is None:
                bar.bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]"
                bar.refresh()
                details = stack.enter_context(tqdm(
                    bar_format="{desc}", position=1, leave=False, file=sys.stderr,
                ))
            update_label(now)

        def report(completed: int, total: int) -> None:
            nonlocal refreshed
            if bar.total is None:
                bar.total = total
                bar.refresh()
            bar.update(max(0, completed - bar.n))
            now = monotonic()
            if route and now - refreshed >= 0.35:
                update_label(now)
                refreshed = now

        yield report, status


def format_code(encoded: tuple[int, int, int]) -> str:
    prefix, digits, byte_count = encoded
    return f"{digits}:{byte_count}:{prefix:b}"


def decode_code(code: str, *, progress: Progress | None = None) -> bytes:
    digits, byte_count, binary = code.strip().split(":")
    if not (
        digits.isascii() and digits.isdecimal()
        and byte_count.isascii() and byte_count.isdecimal()
        and binary and set(binary) <= {"0", "1"}
    ):
        raise ValueError("Expected digits:byte_count:binary_prefix")
    prefix = int(binary, 2)
    digit_count = int(digits)
    if prefix >= 10**digit_count:
        raise ValueError("Prefix does not fit the declared decimal digit count")
    return decompress(prefix, digit_count, int(byte_count), progress=progress)


def interactive() -> None:
    choice = int(input("Compress (1) or Decompress (2): "))
    if choice == 1:
        data = input("Text to compress: ").encode("utf-8")
        with progress_bar("Compressing") as (report, _):
            encoded = compress(data, progress=report)
        prefix, digits, byte_count = encoded

        payload_bytes = (prefix.bit_length() + 7) // 8
        print("Original:", len(data), "bytes")
        print("ARIL payload:", payload_bytes, "bytes")
        print("gzip:", len(gzip.compress(data)), "bytes")

        print(f"Decimal prefix: 0.{prefix:0{digits}d}")
        print(f"Stored tuple: {encoded}")
        print(f"In binary: {bin(prefix)}")

        print(f"ARIL string: {format_code(encoded)}")
    elif choice == 2:
        encoded = input("Input ARIL string (format: digits:byte_count:binary_prefix): ")
        with progress_bar("Decompressing") as (report, _):
            recovered = decode_code(encoded, progress=report)
        print(recovered.decode("utf-8"))
    else:
        raise RuntimeError("Invalid input")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compress and decompress files with ARIL.")
    commands = parser.add_subparsers(dest="command", required=True)

    compress_parser = commands.add_parser("compress", help="Compress a file or folder into a binary archive")
    compress_parser.add_argument(
        "--model", choices=ARCHIVE_VERSIONS,
        help="Compression mode (default: grouped for Git with Preflate, fast otherwise)",
    )
    compress_parser.add_argument(
        "--group-size", type=int, choices=GROUP_MIB, metavar="MiB",
        help="Solid group size for grouped compression (1, 4, or 8 MiB; default: 1)",
    )
    compress_parser.add_argument("--jpeg-xl", action="store_true", help="Try byte-exact JPEG recompression in grouped mode")
    compress_parser.add_argument(
        "--preflate", action=argparse.BooleanOptionalAction,
        help="Try byte-exact recompression of loose Git objects (automatic in grouped mode when available)",
    )
    compress_parser.add_argument("--stats", action="store_true", help="Show input bytes by compression decision")
    decompress_parser = commands.add_parser("decompress", help="Restore a file or folder from an archive")
    for command_parser in (compress_parser, decompress_parser):
        command_parser.add_argument("input", type=Path, help="Input file")
        command_parser.add_argument("-o", "--output", type=Path, required=True, help="Output file")

    list_parser = commands.add_parser("list", help="List paths in a binary archive")
    list_parser.add_argument("input", type=Path, help="Archive file")

    if len(sys.argv) == 1:
        interactive()
        return

    args = parser.parse_args()
    try:
        if args.command == "list":
            for name in list_archive(args.input):
                print(name)
        elif args.command == "compress":
            stats = Counter() if args.stats else None
            with progress_bar("Compressing") as (report, status):
                create_archive(
                    args.input, args.output, model=args.model, progress=report, stats=stats, status=status,
                    group_mib=args.group_size, jpeg_xl=args.jpeg_xl, preflate=args.preflate,
                )
            if stats is not None:
                for reason, size in stats.most_common():
                    print(f"{reason}: {size:,} input bytes", file=sys.stderr)
                print(f"Archive: {args.output.stat().st_size:,} bytes", file=sys.stderr)
        else:
            with args.input.open("rb") as stream:
                binary_archive = stream.read(4) == b"ARIL"
            if binary_archive:
                with progress_bar("Extracting archive") as (report, status):
                    extract_archive(args.input, args.output, progress=report, status=status)
            else:
                with progress_bar("Decompressing") as (report, _):
                    data = decode_code(args.input.read_text(encoding="ascii"), progress=report)
                    with args.output.open("xb") as output:
                        output.write(data)
    except (OSError, ValueError) as error:
        parser.exit(1, f"aril: {error}\n")
