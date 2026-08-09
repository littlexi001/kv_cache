# Visualization results

## 10M-token smoke result

All four conditions completed from identical random initialization and identical
data hashes. Final natural validation NLL was 6.258--6.262 and final training loss
was 6.302--6.312. At 512 tokens, Gold retrieval NLL improved to 9.28--9.35 from
the 32K-vocabulary random reference of about 10.37, while top-1 accuracy remained
zero. The smoke therefore validates the training/data pipeline but is too small
to rank PE methods.

Status: 100M-token learning pilot is the next decision point.

The primary plots after the pilot will be:

1. natural validation NLL versus tokens seen;
2. retrieval Gold NLL versus context length;
3. retrieval accuracy versus context length;
4. short-context PPL delta relative to native RoPE.

No scientific conclusion is recorded until the Stage B learning gate passes.
