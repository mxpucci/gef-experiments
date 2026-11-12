# SElf* integration

This C++17 implementation follows the streaming algorithm in Li et al.,
*Adaptive Encoding Strategies for Lossless Floating-Point Compression*, IEEE
Internet of Things Journal 12(14), July 2025, DOI
[10.1109/JIOT.2025.3554999](https://doi.org/10.1109/JIOT.2025.3554999),
particularly §VII (preceding-window Huffman coding and selective rule updates).

The authors' implementation is Java in
[Spatio-Temporal-Lab/SElfStar](https://github.com/Spatio-Temporal-Lab/SElfStar),
pinned for reference to `457ceb0033e98d516fef42e39d17c9d4b8abbfdb`:

- `compressor/SElfStarCompressor.java`
- `compressor/xor/SElfStarXORCompressor.java`
- `decompressor/SElfStarDecompressor.java`
- `decompressor/xor/SElfStarXORDecompressor.java`

Paths are under `src/main/java/org/urbcomp/startdb/selfstar/`. No upstream license
file was present at this revision; this integration does not assign an
invented license to that source. The external C++ `self` implementation in
[hcmut-huan/Data-Stream-Compression](https://github.com/hcmut-huan/Data-Stream-Compression)
uses float32 and does not provide the required float64 implementation.
Consequently this is a C++ algorithm port, sharing erasure, Huffman and zero-rule
primitives with `benchmark/ElfStar/ElfStar.hpp`; no Java runtime is required.

## Streaming state and access

Use **one `gef_compressors::selfstar::Encoder` per dataset**, and call
`compress_block(values)` once for each consecutive window. The paper's default
window size is 1,000. State carries across calls:

- The first window uses fixed zero rules and the Elf+ prefix coding of beta.
- Each later window uses the preceding window's 17-state Huffman frequencies.
- If the preceding window's compression ratio deteriorated, its observed XOR
  distributions determine new leading/trailing rules. Otherwise rules persist.
- The stored shared leading/trailing window also persists, matching upstream
  `refresh()` semantics. The previous raw value resets at each window.

The convenience `compress(values)` creates a fresh encoder and is a **cold-start
block only**. Repeatedly calling that function for 1,000-value blocks would omit
SElf*'s adaptation; the benchmark must use `Encoder::compress_block`.

Each returned block carries an explicit **decoder-state checkpoint**. This is
necessary for independent indexed block decoding because the original streaming
format depends on previous windows. The checkpoint contains all required
Huffman codes, zero tables and shared-window state; the decoder retains no
original values and needs no preceding blocks. Every checkpoint byte must count
towards `compressed_bits`. This is block decompression, **not native individual
random access**. `decompress(bytes, count)` decodes one complete checkpointed
block. For point/range queries, `decompress_prefix(bytes, total_count,
prefix_count)` restores the checkpoint and decodes only the requested prefix;
it neither visits previous windows nor decodes and discards the block suffix.
It accepts a zero-length prefix, checks the serialized total count, and rejects
prefixes exceeding that count. A partial prefix validates only the checkpoint
and bits it reads; full `decompress` additionally validates the end marker and
all padding/trailing bytes. A prefix equal to the total count also performs
those complete validations. This remains sequential prefix decoding within a
checkpointed block, not constant-time native point access.

The serialized checkpoint also means decoding builds its Huffman tree
from stored codes rather than recomputing preceding-window frequencies; this
is a documented access wrapper, not a byte-identical original container.

## Format and fidelity

A block contains:

1. 32-bit `SES1` magic and 64-bit value count, most significant bit first.
2. One first-window flag and two 7-bit retained XOR window fields (`65` means
   not yet initialized).
3. Leading and trailing rules: 5-bit item count (`0` represents 32 items) and
   6 bits per position, matching the upstream representation.
4. For non-first windows, the preceding window's Huffman codes in the shared
   Elf* code serialization.
5. Zero padding to the next byte, then the native SElf* window payload,
   including its rule-update flag/rules, end marker and byte padding.

The algorithm's internal adaptation trigger uses native payload bits before
its end marker, exactly as upstream does. It excludes access checkpoint bytes;
reported compressed sizes include them. The checkpoint thus does not alter
streaming rule decisions. Four consecutive window payloads match the unmodified
official Java encoder **byte for byte**, exercising first-window coding, table
updates, Huffman transitions and cross-window shared state.

Shared Elf* exactness guards retain values with the algorithm's existing
no-erasure symbol whenever decimal recovery would not be bit exact. NaN bits
are preserved instead of normalized. The explicit count permits the NaN end
marker bit pattern to occur as an ordinary input value, with a separate
validated end marker after `count` values. These changes extend safe handling
of IEEE edge cases; they do not substitute a different compressor. See
`docs/elfstar.md` for the shared erasure details.

Compile in a separate translation unit with
`-fno-fast-math -ffp-contract=off`; the shared header rejects fast-math builds.

## Validation

```sh
cmake -S tests/selfstar -B build-selfstar -DCMAKE_BUILD_TYPE=Release
cmake --build build-selfstar -j2
ctest --test-dir build-selfstar --output-on-failure
```

Release-active tests cover the official Java fixtures, more than 25,000 values
across windows with changing distributions, empty/singleton windows, constant
and periodic data, signed zeros, subnormals, extreme finite values, infinities
and NaN payloads, reverse-order independent block and prefix decoding, all truncated
prefixes of a sample block, forged count and trailing bytes. Local GCC 14 and
Clang AddressSanitizer/UndefinedBehaviorSanitizer runs are part of validation;
remote build verification is recorded by the main integration task.
