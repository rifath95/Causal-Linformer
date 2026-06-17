import torch

# Device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

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

attention_type =  "standard"  # Switch between "standard" and "causal_blockwise_linformer"
num_global_blocks = 64  # Number of compressed key/value vectors
assert block_size % num_global_blocks == 0, "block_size must be divisible by num_global_blocks"
causal_block_size = block_size // num_global_blocks # How many key/value pairs are being compressed within each causal block.
local_window = causal_block_size - 1  #  Number of recent tokens used by the exact local attention branch. With this value, there will not be any gaps.
assert local_window > 0, "local_window must be positive"
share_linformer_projections_across_heads = False  # Set True to share projection matrices E and F across heads
