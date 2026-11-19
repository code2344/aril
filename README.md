# ARIL

ARIL is a compression method that takes advantage of context, not just content. 

Compression is inefficient. Encoding a text such as lorum ipsum in ascii produces `01001100 01101111 01110010 01110101 01101101 00100000 01101001 01110000 01110011 01110101 01101101`. Now, this is pretty long. If you were to talk to another person, and say "What word comes after lorum", they'll probably be able to tell you that it's ipsum. So, this program takes advantage of that, and, using a LLM, predicts the next word. This means you can then represent every single phrase, word, and file using a single float between 0 and 1. I explained that really poorly, but (this video)[https://www.youtube.com/watch?v=qbsb9DpdZ6s] explains it better. This program, on 1 MiB of wikipedia, performed better than .zip and .gz. However, it isn't perfect yet, and it still loses to bzip2 and xz. 

## Setup

Requirements: Python 3.13+, [uv](https://docs.astral.sh/uv/), and (optionally) Rust/Cargo for Git Preflate support.

```sh
git clone github.com/code2344/aril
cd aril
uv sync
```

Run ARIL directly from the checkout:

```sh
uv run aril --help
```

To install the command as a user-level tool instead:

```sh
uv tool install .
aril --help
```

## Files and folders

```sh
uv sync
uv run aril compress notes/ -o notes.aril
uv run aril list notes.aril
uv run aril decompress notes.aril -o restored/
```

`-o` is to set output, can be a directory or file name. 

Use `--model microgen` or `--model micromixer` for full neural compression. These modes predict every byte and are intended for small experiments. The archive records how to decode each block, so extraction needs no model flag.

The first operation that uses a predictor downloads its checkpoint from hugging face. MicroMixer downloads about 4 MB of weights.

## Solid compression

```sh
uv run aril compress notes/ --model solid -o notes-solid.aril
uv run aril decompress notes-solid.aril -o restored/
```

Solid mode compresses file contents and archive metadata together in independent 1 MiB groups. Files in the same group can share compression context. It uses LZMA2 and serves as a model-free baseline for ARIL's neural modes.

## Context-mixing experiments

```sh
uv run aril compress notes/ --model context-neural -o notes-context.aril
uv run aril decompress notes-context.aril -o restored/
```

These profiles share a bounded match history across files. `context-flat` models the contents as a single byte stream. `context` adds file type and indentation context. `context-neural` uses the same predictors and adapts their weights online with an integer neural mixer. `context-selective` bypasses that mixer during sustained exact matches. The mixer learns while encoding or decoding and needs no weight download. These modes are experimental and run in Python. `context-neural` uses raw or DEFLATE blocks for Git metadata, binary data, and blocks that DEFLATE shrinks to one-sixth of their original size. For other text, `context-neural` compares the actual neural and conventional payload sizes and keeps the smaller result. Run with `--stats` to see input bytes by compression decision. Use `context-neural-full` to run the mixer on every block.

## Grouped compression

```sh
uv run aril compress project/ --model grouped --stats -o project.aril
uv run aril decompress project.aril -o restored/
```

Grouped mode compares solid LZMA2 with the neural mixer on related text files. C/C++ sources and headers, JavaScript and TypeScript, Python sources and stubs, and assembly sources and includes share file-family groups within each directory. Numbered text files in frame directories also get a previous-frame delta trial. Git metadata and other binary data use conventional compression.

Compiled inputs (ELF, Mach-O, PE, WebAssembly, and common object/library suffixes) also get an automatic reversible BCJ filter trial before LZMA2. The filtered stream is kept only when it is smaller than the ordinary grouped candidate, so unrelated binaries cannot regress.

Use `--group-size 1`, `4`, or `8` to set the solid group size in MiB. The default is 1 MiB. Neural candidate groups stay bounded at 256 KiB.

Use `--jpeg-xl` to try reversible JPEG recompression on JPEG files up to 1 MiB. Compression requires `cjxl` and `djxl` on PATH. ARIL verifies reconstruction of the original JPEG byte-for-byte and keeps the transformed group only if it beats solid compression of the original files. Extraction of a JPEG XL group requires `djxl`.

The terminal shows the active route, the file at the group boundary, and an estimate for the current group as progress becomes available.

## Git loose objects

Build the optional Preflate helper with Cargo:

```sh
cargo build --release --locked --manifest-path tools/preflate/Cargo.toml
uv run aril compress ../repository -o repository.aril --stats
```

ARIL looks for `aril-preflate` on PATH or under `tools/preflate/target/release` in the checkout. With the helper available, it automatically tries Preflate on loose objects in Git repositories and in grouped mode. The helper verifies the original object bytes before ARIL can keep the candidate. Packfiles use the conventional route. Extraction requires the helper only when an archive contains a Preflate group.

Use `--no-preflate` to disable the trial or `--preflate` to require the helper. Loose objects up to 8 MiB can be transformed. An object larger than the configured group size gets its own group.

## Interactive text

Run `uv run aril` for the text prompt. It uses `digits:byte_count:binary_prefix` codes. The `decompress` command can also restore files from those codes.
