# EAS: implementation pending an author artifact

EAS is **not implemented or registered** in the benchmark yet. The paper is
available and has been read; the missing material is a usable implementation
or a sufficiently complete executable specification. No ALP wrapper, generic
hybrid, or guessed codec has been labelled EAS.

Paper: Dongxuan Chen, Weijie Wang, Jia Peng, Lei Wang, and Xianyou Zhu,
*EAS—An adaptive lossless time series data compression algorithm*, Future
Generation Computer Systems 186, 108742, online 30 July 2026 (January 2027
volume). [Publisher record](https://www.sciencedirect.com/science/article/pii/S0167739X26003766),
[DOI](https://doi.org/10.1016/j.future.2026.108742).

## Source audit

On 3 November 2026, the supplied final PDF, publisher record, GitHub repository
and code searches, and web searches targeting GitHub, Gitee, Zenodo and Mendeley
Data were checked. Queries included the exact title, DOI/PII, author names and
paper contact identifiers. No author-linked EAS source/artifact was found.
GitHub's exact DOI search returned a bibliographic curation record rather than
an implementation. Searches cannot prove that a private or unindexed artifact
does not exist. The PDF's data-availability statement says “Data will be made
available on request”; it does not provide a code URL. The GitHub URLs in its
references are for other codecs or datasets.

The paper uses C++-style pseudocode and discusses scalar, auto-vectorized and
hand-written SIMD versions. This is evidence of an implementation described by
the authors, but does not establish its source language, API or availability.
A repository URL or source archive, with its revision and build instructions,
is needed before importing it. The paper gives author contact addresses
`dxchen@hynu.edu.cn`, `wwj011224@outlook.com`, and corresponding author
`zxy@hynu.edu.cn`. No messages have been sent to the authors.

## Why the paper alone is insufficient for a faithful competitor

These details affect the algorithm, compressed size or measured throughput;
they are not merely choices about a C++ wrapper.

| Area | Specified in the paper | Unresolved implementation details |
| --- | --- | --- |
| Fixed-point conversion | §3.6 Eq. (9) describes flooring `v * 10^e`, masking the lowest `f` integer bits and dividing by `10^e`; §3.3 also presents round-based conversion. | The actual `EAS Encode`/inverse implementation, integer widths, signed conversion and overflow rules, precise operation ordering, and handling of special IEEE values. The terms used for `f` include both a bit mask and a scaling factor. Importing ALP's different mapping would not resolve this. |
| Fast layer | §4.2.1 fixes `e=14`, `f=10`, FFOR, and an outlier threshold of 5%. | Algorithm 2 returns `RAW(input)` above 5%, while nearby prose says to apply integer encoding unconditionally and then transition to the next layer when outliers are excessive. The implemented behavior needs confirmation. |
| Layer selection and candidates | Algorithm 1 calls `select Compression Layer`; Algorithm 3 calls `generate Candidates(features)`. §4.2.2 describes 2–4 integer candidates plus XOR, with an example sample of 32 values. | The rules/thresholds, feature definitions, exact candidate generation, sampling distribution, row-group sizes and early-stop logic are absent. |
| XOR and predictive paths | §3.2 reviews XOR compression; §4.2.2 mentions linear-regression preprocessing; §4.2.3 describes competing predictive, integer and XOR paths. | The actual XOR control format, reference/window policy, predictor construction and storage, and residual representation are not specified. Selecting an existing XOR codec would be a new design choice. |
| Adaptive exceptions | §3.6 and §4.2.2 discuss raw, differential, consecutive/block and extreme-value coding, plus compressed metadata. | Exact classification boundaries, choice of predictor/reference, lossless representation of differences, block and variable-length codes, and the meaning of segmented quantization. Metadata coding differs by layer but has no complete format. |
| Deep search | §3.4 gives `e in [0,21]`, `f in [0,e]`, Metropolis acceptance and exponential cooling; weights 0.7/0.2/0.1 combine size, outlier count and time. | Initial temperature, cooling factor, iteration budget, neighborhood distribution, feature-dependent initialization, stopping rule, score units/normalization and random generator/seed. §4.4 says fixed seeds were used but does not publish them. |
| Stream layout/access | §4.3 lists layer/method, `e,f`, and outlier offsets/counts in block metadata. | Field encodings, length widths, FFOR packing layout, padding, block/vector/row-group boundaries and the index needed for skipping. Without these, compression-ratio and point-access comparisons are not reproducible. |

It is possible to construct a lossless compressor inspired by this architecture
by inventing the missing decisions and storing every non-reconstructible value
as an exception. Successful round trips would establish that new compressor's
correctness, not that it reproduces EAS. Such an implementation is therefore
not an acceptable substitute for this requested baseline.

## Artifact needed to complete integration

Obtain the encoder and decoder used for the paper, including all three layers,
parameter tables, configuration/defaults, dependency versions and at least one
known input/output example. The source should identify the revision and permit
the intended use. Prefer importing the native C/C++/Rust/Go implementation;
if only Java is provided, a documented port can be validated against its output.

Then add the strict IEEE double adapter under `benchmark/EAS`, count every
header, exception and index byte, and test exact reconstruction, each strategy,
threshold transitions and malformed/truncated data. Compile and run those
checks on the requested x86-64 server. For random-access measurements, document
whether the artifact provides actual point access or only independently
addressable vector/block decoding. These implementation and validation steps
remain outstanding; no EAS build or test success is claimed.
