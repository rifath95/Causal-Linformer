import math
import os
import torch

from config import *
from data import val_data
from model import MyGPT


checkpoint_path = "model.pth"
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train.py first.")

checkpoint_state = torch.load(checkpoint_path, map_location="cpu")
checkpoint_uses_linformer = any(key.endswith(".attn.E") for key in checkpoint_state)
checkpoint_attention_type = "causal_blockwise_linformer" if checkpoint_uses_linformer else "standard"
if checkpoint_attention_type != attention_type:
    raise ValueError(f"Checkpoint uses '{checkpoint_attention_type}' attention, but config.py uses '{attention_type}'. Set attention_type correctly or load matching weights.")

model = MyGPT()
model.load_state_dict(checkpoint_state)
model = model.to(device)
model.eval()

total_parameters = sum(parameter.numel() for parameter in model.parameters())
trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

print("Model parameters:")
print(f"  Checkpoint: {checkpoint_path}")
print(f"  Device: {device}")
print(f"  Attention type: {attention_type}")
print(f"  Maximum context length: {block_size}")
print(f"  Hidden size: {d_hidden}")
print(f"  Layers: {n_layers}")
print(f"  Attention heads: {n_heads}")
print(f"  Head size: {d_head}")
print(f"  Dropout configured: {dropout}")
if attention_type == "causal_blockwise_linformer":
    print(f"  Local window: {local_window}")
    print(f"  Global blocks: {num_global_blocks}")
    print(f"  Shared Linformer projections: {share_linformer_projections_across_heads}")
print(f"  Total parameters: {total_parameters:,}")
print(f"  Trainable parameters: {trainable_parameters:,}")

# Evaluate every next-token target in the validation split.
total_target_tokens = len(val_data) - 1
full_pair_count = (total_target_tokens // block_size) * block_size
full_inputs = val_data[:full_pair_count].unfold(0, block_size, block_size)
full_targets = val_data[1:full_pair_count + 1].unfold(0, block_size, block_size)
total_loss = 0.0
evaluated_tokens = 0

with torch.inference_mode():
    for start in range(0, len(full_inputs), batch_size):
        x = full_inputs[start:start + batch_size].to(device)
        y = full_targets[start:start + batch_size].to(device)
        _, loss = model(x, y)
        batch_tokens = y.numel()
        total_loss += loss.item() * batch_tokens
        evaluated_tokens += batch_tokens

    # Include the shorter final chunk instead of dropping it.
    if full_pair_count < total_target_tokens:
        x = val_data[full_pair_count:-1].unsqueeze(0).to(device)
        y = val_data[full_pair_count + 1:].unsqueeze(0).to(device)
        _, loss = model(x, y)
        remainder_tokens = y.numel()
        total_loss += loss.item() * remainder_tokens
        evaluated_tokens += remainder_tokens

validation_loss = total_loss / evaluated_tokens
validation_perplexity = math.exp(validation_loss)

print("Validation results:")
print(f"  Attention type: {attention_type}")
print(f"  Validation tokens evaluated: {evaluated_tokens:,} / {total_target_tokens:,}")
print(f"  Validation loss: {validation_loss:.6f}")
print(f"  Validation perplexity: {validation_perplexity:.6f}")
