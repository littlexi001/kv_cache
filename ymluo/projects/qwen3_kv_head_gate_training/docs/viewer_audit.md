# Viewer Audit

No result viewer has been produced yet.

The first viewer must define:

- PPL as `exp(mean next-token cross entropy)`;
- hard keep ratio as selected KV token-head slots divided by all KV token-head
  slots;
- average heads per token as `hard_keep_ratio * num_key_value_heads`;
- expected KV-cache memory as original memory multiplied by hard keep ratio.

