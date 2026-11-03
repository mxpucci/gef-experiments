# Kangaroo: source needed before benchmark integration

Deferred at the user’s request on 3 November 2026. The source audit below is
retained for a future integration and does not block the current scope.

Status (2026-11-03): **not implemented or registered as a benchmark**. The paper
is available and has been read. The remaining dependency is the authors' source
artifact, preferably the C++ version explicitly mentioned in the paper.

## Paper and implementation evidence

Shuo Li et al., *Kangaroo: Efficient Lossless Floating-Point Compression via
Dynamic Reference Selection*, SIGMOD 2026, DOI
[10.1145/3802076](https://doi.org/10.1145/3802076).

- Section 6.1.2 identifies **Fast-Kangaroo** as the main experimental variant.
- Section 6.1.4 reports a Java implementation and also a **C++ implementation**
  used for comparison with the open-source C++ ALP implementation.
- Section 6.2 uses window size `W = 32`; section 6.1.4 uses blocks of 1,000 values.
- The paper does not supply a source-code repository or artifact download link.
- The paper-specific repository
  [LeeSure5986/KangarooData](https://github.com/LeeSure5986/KangarooData), inspected
  at commit `19c0ee7a72bb138af5338b1f5e7b1dd6ff4831cf`, contains 23 CSV datasets
  represented through Git LFS, plus `.gitattributes`. It contains no compressor
  implementation, build files, or README linking a code artifact. Its description
  explicitly identifies it as the datasets for this paper; this alone does not
  establish the repository owner's authorship.
- GitHub repository searches for `Kangaroo compression` and
  `"floating-point" "Kangaroo"`, code searches for `KangarooCompressor`,
  `"Kangaroo" "VL-RLE"`, and `"Fast-Kangaroo"`, and web searches for the exact
  paper title did not locate an implementation. Negative search results are not
  proof that no implementation exists elsewhere.

## Why a speculative replacement is not registered

The core ideas are described, but several choices affect exactly the metrics this
repository is intended to compare:

1. **Fast reference search.** Section 4 describes an O(1) lookup indexed by the
   current XOR leading-zero count. It does not give the lookup table's update,
   replacement, and window-expiration procedure. Algorithm 1 describes the
   different, scanning **Compact-Kangaroo** variant. Substituting that algorithm
   would change the main competitor's encoding speed and potentially its size.
2. **Run-length wire encoding.** Section 3.3.5 describes runs and section 5
   reserves prefix `110` for VL-RLE, but the run-length integer code and framing
   are not specified. Choosing an arbitrary varint or gamma code changes the
   reported compression size, especially on repeated-value datasets.
3. **Numeric edge handling.** Section 5 gives the erasure/bit-flip formulas and
   three examples (1.25, 2.87, and 11.44). It does not provide an executable
   definition of decimal-place extraction and exact reconstruction over the
   complete floating-point domain, including negative values, signed zeros,
   subnormals, infinities, and NaN payloads. These should be established from the
   original implementation and checked with bitwise round-trip tests.
4. **Initialization and serialized format.** First-value representation,
   reference-index representation during window warm-up, and stream termination
   are not fully specified. These affect small 1,000-value blocks and metadata
   accounting.

A paper-derived prototype could resolve these choices independently, but it would
not establish reproduction of the reported Fast-Kangaroo competitor. No generic
XOR codec, arbitrary fallback, empty stub, or guessed format is presented as
Kangaroo here.

## Artifact needed and next step

Obtain the authors' C++ source archive or repository URL, including its license,
decoder, and benchmark configuration. The first author contact listed in the
paper is `2410815@stu.neu.edu.cn`; the corresponding author is
`yangxc@mail.neu.edu.cn`. No message has been sent to either author.

Once supplied: pin the artifact, import its build rather than copying its source,
wrap Fast-Kangaroo with `W=32`, include all serialized metadata, and validate exact
round trips and server compilation. If only Java is supplied, a faithful native
port can be verified against upstream encoded fixtures. Random access should be
reported as enclosing-block decoding unless the source reveals an actual native
access API; the paper does not establish one.
