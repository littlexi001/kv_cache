# Experiment design

## Inputs

- Raw corpus: `/home/fdong/data/openweb_every_4096/*.txt`.
- Held-out corpus: `/home/fdong/data/openweb_every_6400/test_openweb256.txt`.
- Vocabulary size: 32,000.
- Model: 12 layers, hidden size 768, 6 query heads, 2 KV heads, head dimension
  128, MLP size 3,072, RoPE base 1,000,000.
- Context length: 2,048 during training.

## Deterministic tokenizer and token stream

1. Train one byte-level BPE tokenizer on a fixed numerically sorted prefix of
   OpenWebText files.
2. Reserve structural retrieval tokens, 1,024 key tokens and 1,024 value tokens.
3. Pack documents with an EOS token into little-endian uint16 binaries.
4. Save SHA-256 hashes and require all servers to use byte-identical artifacts.
5. Select chunks through a deterministic affine permutation of chunk indices, so
   the compared methods see the same examples in the same order without early
   repetition.

## Mixed-example construction

For every global sample index:

1. Read one 2,048-token natural chunk.
2. Deterministically mark 5% of sample indices as retrieval examples.
3. For a retrieval example, insert eight distinct facts before the query region:
   `<fact> <key> K <value> V <sep>`.
4. Insert 32 queries at the end, repeating every one of the eight keys four
   times so the 100M-token pilot contains enough supervised retrieval events:
   `<query> K <answer> V <sep>`.
5. Apply unit next-token loss to all positions and weight the answer targets
   by 16.

## Stages and pass/fail gates

### Stage A: contract and 10M-token smoke test

Pass when all of the following hold:

- tokenizer size is exactly 32,000 and all IDs fit uint16;
- both servers report identical artifact hashes;
- loss and gradient norm remain finite;
- natural validation NLL is below the untrained value;
- the four methods receive identical rank-0 batch hashes.

### Stage B: 100M-token learning pilot

Pass when:

- native 512-token Gold NLL is materially below `ln(1024)`;
- preferably native 512-token retrieval accuracy is at least 25%;
- natural validation PPL does not show divergence.

If the accuracy gate fails, adjust the retrieval curriculum before any 2.5B run.

### Stage C: 2.5B-token comparison

Run only after Stage B passes. Build at least 2.375B natural tokens so the formal
comparison does not repeatedly cycle through a small cache. Start all four
conditions from the same random initialization; do not resume the old
synthetic-only checkpoints.

## Output contract

Each condition writes `config.json`, `train.jsonl`, `eval.jsonl`, checkpoints,
`DONE`, and a launcher log under a new output directory. Existing output
directories are never reused.

## FadeRoPE-band8 prototype

This is a paired prototype of Phase-Confidence RoPE. For frequency band (b),
let its representative angular frequency be the geometric mean
\(\bar\omega_b\), relative distance be \(\Delta\), and the learnable trust radius
for layer \(l\), head \(h\) be

\[
\tau_{l,h}=2\pi\exp(\eta_{l,h}), \qquad \eta_{l,h}=0\text{ at initialization}.
\]

The confidence and score are

\[
c_{l,h,b}(\Delta)=
\frac{1}{1+(|\Delta|\bar\omega_b/\tau_{l,h})^4},
\]

\[
s_{l,h,b}=c_{l,h,b}s^{\mathrm{RoPE}}_{l,h,b}
+(1-c_{l,h,b})s^{\mathrm{NoPE}}_{l,h,b}.
\]

The 64 pairs are grouped as F0--F7, ..., F56--F63. The paired reference uses
the identical eight-band kernel with \(c=1\), so implementation differences do
not confound the comparison.

Prototype pass conditions:

- the band reference matches standard RoPE within numerical tolerance;
- gradients of every \(\eta_{l,h}\) are finite;
- a real forward/backward fits on the assigned GPUs;
- paired runs have identical non-PE initialization and data hashes;
- natural-text validation does not materially regress;
- retrieval Gold NLL or accuracy improves at one or more extrapolation lengths.
