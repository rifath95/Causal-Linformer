device = "mps"

# training parameters
batch_size = 64
n_steps = 500
dropout = 0.2
lr = 3e-4

# model parameters
block_size = 256
d_hidden = 384
n_layers = 6

n_heads = 6
assert d_hidden % n_heads == 0, "hidden size must be divisble by number of heads to get a integer head size"
d_head = d_hidden//n_heads

attention_type =  "causal_blockwise_linformer"  # "standard" [Codex comment] Keep the existing full causal attention as the default behavior.
local_window = 3  # [Codex comment] Set the number of recent tokens used by the exact local attention branch.
num_global_blocks = 64  # [Codex comment] Divide the configured maximum context into this many fixed global blocks.
share_linformer_projections_across_heads = False  # [Codex comment] Use separate learned compression weights for each attention head by default.
