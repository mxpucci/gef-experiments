# Elf* integration

`benchmark/ElfStar/ElfStar.hpp` implements the double-precision **Elf*** batch
compressor from Zheng Li et al., *Adaptive Encoding Strategies for Lossless
Floating-Point Compression*, IEEE Internet of Things Journal 12(14), July 2025,
DOI [10.1109/JIOT.2025.3554999](https://ieeexplore.ieee.org/document/10949160/).
The supplied paper was read, particularly Sections III–VII on Huffman states,
adaptive leading/trailing rules, sharing conditions and the streaming variant.

The official implementation is Java:
<https://github.com/Spatio-Temporal-Lab/SElfStar>, pinned reference commit
`457ceb0033e98d516fef42e39d17c9d4b8abbfdb`. The source classes used to check the
port are `ElfStarCompressor`, `ElfStarXORCompressor`, their decoders,
`Elf64Utils`, `PostOfficeSolver`, and `utils/Huffman`. That repository has no
LICENSE file at this commit; no upstream license grant is inferred or
represented as MIT. This is a native C++ implementation, not an official C++
release. Author and source attribution is retained here.

A native third-party candidate was also inspected:
<https://github.com/hcmut-huan/Data-Stream-Compression>. Its `self` compressor
implements the 32-bit variant (nine Huffman symbols); it does not supply the
needed 64-bit Elf* batch implementation. Consequently it is not imported.
There is no Java runtime dependency in this integration.

## Algorithm and format

This is distinct from the repository's older `Elf` baseline. It includes:

- Decimal erasure with the 17 joint Huffman states (16 beta values plus no erasure).
- Frequency-and-height Huffman construction, including Java priority-queue tie
  behavior for reproducible comparison with the authors' encoding.
- The authors' pruned post-office dynamic program selecting independently
  optimized leading- and trailing-zero rules.
- The adaptive sharing cost criterion and XOR encoding cases.

`gef_compressors::elfstar::compress(vector<double>)` returns bytes;
`decompress(bytes, count)` reconstructs the same IEEE-754 bit patterns. All bytes
must be counted. The frame consists of an eight-byte big-endian value count,
followed by the authors' Elf* bitstream: complete Huffman metadata, leading and
trailing rules, data, end marker, and final byte padding. The count adds eight
bytes per independently encoded block and disambiguates actual NaN data from
the authors' canonical-NaN end marker. The decoder checks the count, end marker,
padding, rule indices and Huffman prefix validity. No raw values or hidden
codec state are retained outside this representation.

The adapter uses the enclosing benchmark's block size (normally 1,000 values,
as in the paper). It offers sequential decode.
`decompress_prefix(bytes, total_count, prefix_count)` validates the count and
Huffman/rule metadata and decodes only the requested prefix, with allocation
proportional to that prefix. A zero-length prefix is valid; a prefix exceeding
the total is rejected. Prefix reads do not check unread suffixes or the end
marker; use full `decompress` for complete-frame validation. An external block directory
can support block-decoding point access; this is **not native point access**.
The shared helpers are available under `gef_compressors::elfstar::detail` for
the separately implemented SElf* streaming codec.

## Defined behavior beyond the upstream implementation

The source's decimal-significance loop and shift calculations can overflow,
index outside their tables, or erase a value whose reconstruction is not bit
exact on extreme/high-precision data. The port guards these conditions and
uses Elf*'s existing **no-erasure state**, still followed by the full adaptive
XOR codec. It does not switch to a different compressor. Before accepting an
erasure it checks that the corresponding decimal restoration reproduces the
original bits. This guard is included in compression time. NaN payloads,
infinities, signed zero, and subnormals retain their original bits rather than
canonicalizing NaNs. Empty blocks are supported. Counters use unsigned 64-bit
arithmetic instead of Java's 32-bit distribution counters.

Use strict floating-point compilation: `-fno-fast-math -ffp-contract=off`.
The header rejects `__FAST_MATH__`. It is suitable for the benchmark's separate
strict-IEEE translation unit, so GEF's own fast-math flags do not change the
codec. Include `cmake/ElfStar.cmake` and link `gef_elfstar` if used standalone.
C++17 is required; there are no external build dependencies.

## Verification

```sh
cmake -S tests/elfstar -B build-elfstar -DCMAKE_BUILD_TYPE=Release
cmake --build build-elfstar -j2
ctest --test-dir build-elfstar --output-on-failure
```

`tests/elfstar_roundtrip.cpp` includes a byte-for-byte golden fixture produced
by the **unmodified official Java implementation** at the commit above. Its
696-bit payload matches exactly after the local eight-byte count header.
`tests/elfstar/Golden.java` records the generator. Compile it alongside the
upstream classes with Maven artifacts `it.unimi.dsi:dsiutils:2.7.0` and
`it.unimi.dsi:fastutil:8.5.6` to regenerate it.

Additional tests cover empty/singleton/constant data, signed zeros, all special
floating-point classes and NaN payloads, 100,000 random raw bit patterns,
100,000 decimal values, powers across exponents -320 through 300, all truncated
prefixes of a valid stream, wrong counts and trailing bytes. Checks remain
active in release builds. These tests passed locally with Clang AddressSanitizer
and UndefinedBehaviorSanitizer, and GCC 14 release compilation. Remote validation
is performed through the common integration build.
