"""Independently compressed groups over an ARIL entry stream."""

import lzma
import struct
from typing import BinaryIO
from zlib import crc32

GROUP_SIZE = 1 << 20
GROUP_HEADER = struct.Struct(">III")
MEMORY_LIMIT = 16 << 20


class SolidWriter:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.buffer = bytearray()

    def write(self, data: bytes) -> None:
        self.buffer.extend(data)
        while len(self.buffer) >= GROUP_SIZE:
            self._write_group(bytes(self.buffer[:GROUP_SIZE]))
            del self.buffer[:GROUP_SIZE]

    def _write_group(self, data: bytes) -> None:
        compressed = lzma.compress(data, format=lzma.FORMAT_XZ, preset=4)
        payload = compressed if len(compressed) < len(data) else data
        self.stream.write(GROUP_HEADER.pack(len(data), len(payload), crc32(data)))
        self.stream.write(payload)

    def finish(self) -> None:
        if self.buffer:
            self._write_group(bytes(self.buffer))
        self.stream.write(GROUP_HEADER.pack(0, 0, 0))


class SolidReader:
    def __init__(self, stream: BinaryIO) -> None:
        self.stream = stream
        self.buffer = b""
        self.position = 0
        self.payload_start = stream.tell()
        self.stored_size = 0
        self.finished = False

    def _read_group(self) -> None:
        header = self.stream.read(GROUP_HEADER.size)
        if len(header) != GROUP_HEADER.size:
            raise ValueError("Truncated ARIL solid group")
        size, stored, checksum = GROUP_HEADER.unpack(header)
        if (size, stored, checksum) == (0, 0, 0):
            if self.stream.read(1):
                raise ValueError("Trailing data after ARIL solid groups")
            self.finished = True
            return
        if not 0 < stored <= size <= GROUP_SIZE:
            raise ValueError("Invalid ARIL solid group lengths")

        self.payload_start = self.stream.tell()
        self.stored_size = stored
        payload = self.stream.read(stored)
        if len(payload) != stored:
            raise ValueError("Truncated ARIL solid payload")
        if stored == size:
            self.buffer = payload
        else:
            decoder = lzma.LZMADecompressor(format=lzma.FORMAT_XZ, memlimit=MEMORY_LIMIT)
            try:
                self.buffer = decoder.decompress(payload, max_length=size + 1)
            except lzma.LZMAError as error:
                raise ValueError("Invalid ARIL solid payload") from error
            if len(self.buffer) != size or not decoder.eof or decoder.unused_data:
                raise ValueError("Invalid ARIL solid payload length")
        if crc32(self.buffer) != checksum:
            raise ValueError("ARIL solid group checksum mismatch")
        self.position = 0

    def read(self, size: int) -> bytes:
        result = bytearray()
        while len(result) < size and not self.finished:
            if self.position == len(self.buffer):
                self._read_group()
                if self.finished:
                    break
            end = min(len(self.buffer), self.position + size - len(result))
            result.extend(self.buffer[self.position:end])
            self.position = end
        return bytes(result)

    def tell(self) -> int:
        if self.finished or not self.buffer:
            return self.stream.tell()
        return self.payload_start + self.position * self.stored_size // len(self.buffer)
