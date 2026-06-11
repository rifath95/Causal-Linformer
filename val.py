import math  # [Codex comment] Convert the mean validation loss to perplexity for an additional readable metric.
import os  # [Codex comment] Check that the trained checkpoint exists before attempting validation.
import torch  # [Codex comment] Load trained weights and evaluate validation tensors without gradients.

from config import *  # [Codex comment] Reuse the exact architecture, attention, device, and evaluation settings used by the model.
from data import val_data  # [Codex comment] Evaluate every next-token target in the held-out validation split.
from model import TestGPT  # [Codex comment] Reconstruct the Transformer architecture before loading its trained weights.


checkpoint_path = "model.pth"  # [Codex comment] Use the checkpoint written by train.py unless this path is deliberately changed.
if not os.path.exists(checkpoint_path):  # [Codex comment] Detect a missing training checkpoint before model construction and evaluation.
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train.py first.")  # [Codex comment] Explain how to create the required trained weights.

checkpoint_state = torch.load(checkpoint_path, map_location="cpu")  # [Codex comment] Load weights safely on CPU before moving the complete model to the configured device.
checkpoint_uses_linformer = any(key.endswith(".attn.E") for key in checkpoint_state)  # [Codex comment] Identify Linformer checkpoints by their learned key-compression parameters.
checkpoint_attention_type = "causal_blockwise_linformer" if checkpoint_uses_linformer else "standard"  # [Codex comment] Convert checkpoint parameter structure into the corresponding configured attention name.
if checkpoint_attention_type != attention_type:  # [Codex comment] Prevent loading weights into an incompatible attention architecture.
    raise ValueError(f"Checkpoint uses '{checkpoint_attention_type}' attention, but config.py uses '{attention_type}'. Set attention_type correctly or load matching weights.")  # [Codex comment] Give a concise correction for attention-type checkpoint mismatches.

model = TestGPT()  # [Codex comment] Build the model using the parameters currently selected in config.py.
model.load_state_dict(checkpoint_state)  # [Codex comment] Restore all trained model parameters from the compatible checkpoint.
model = model.to(device)  # [Codex comment] Move the loaded model to the configured CPU, CUDA, or MPS device.
model.eval()  # [Codex comment] Disable dropout so validation loss is deterministic and representative of inference behavior.

total_parameters = sum(parameter.numel() for parameter in model.parameters())  # [Codex comment] Count every learned scalar in the selected model architecture.
trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)  # [Codex comment] Count parameters that were optimized during training.

print("Model parameters:")  # [Codex comment] Introduce the architecture and checkpoint details used for this validation run.
print(f"  Checkpoint: {checkpoint_path}")  # [Codex comment] Show which trained weight file was evaluated.
print(f"  Device: {device}")  # [Codex comment] Show where validation computation is running.
print(f"  Attention type: {attention_type}")  # [Codex comment] Report whether standard or causal blockwise Linformer attention is active.
print(f"  Maximum context length: {block_size}")  # [Codex comment] Report the token length used for each full validation chunk.
print(f"  Hidden size: {d_hidden}")  # [Codex comment] Report the Transformer hidden-state width.
print(f"  Layers: {n_layers}")  # [Codex comment] Report the number of Transformer blocks.
print(f"  Attention heads: {n_heads}")  # [Codex comment] Report the number of attention heads per block.
print(f"  Head size: {d_head}")  # [Codex comment] Report the feature width assigned to each attention head.
print(f"  Dropout configured: {dropout}")  # [Codex comment] Report training dropout even though evaluation mode disables it.
if attention_type == "causal_blockwise_linformer":  # [Codex comment] Print settings that apply only to the Linformer attention implementation.
    print(f"  Local window: {local_window}")  # [Codex comment] Report the number of exact recent tokens visible to each query.
    print(f"  Global blocks: {num_global_blocks}")  # [Codex comment] Report the number of fixed compressed blocks across maximum context.
    print(f"  Shared Linformer projections: {share_linformer_projections_across_heads}")  # [Codex comment] Report whether heads share key and value compression weights.
print(f"  Total parameters: {total_parameters:,}")  # [Codex comment] Print the complete learned parameter count with readable separators.
print(f"  Trainable parameters: {trainable_parameters:,}")  # [Codex comment] Print the number of parameters that participated in optimization.

total_target_tokens = len(val_data) - 1  # [Codex comment] Count every next-token prediction available in the validation split.
full_pair_count = (total_target_tokens // block_size) * block_size  # [Codex comment] Find how many validation targets fit into full maximum-context chunks.
full_inputs = val_data[:full_pair_count].unfold(0, block_size, block_size)  # [Codex comment] Partition full validation inputs into contiguous non-overlapping contexts.
full_targets = val_data[1:full_pair_count + 1].unfold(0, block_size, block_size)  # [Codex comment] Align each full input chunk with its one-token-shifted targets.
total_loss = 0.0  # [Codex comment] Accumulate token-weighted loss across every full and partial validation chunk.
evaluated_tokens = 0  # [Codex comment] Track the exact number of validation targets included in the final mean.

with torch.inference_mode():  # [Codex comment] Disable gradient recording to reduce validation memory and computation.
    for start in range(0, len(full_inputs), batch_size):  # [Codex comment] Evaluate full context chunks in batches using the configured batch size.
        x = full_inputs[start:start + batch_size].to(device)  # [Codex comment] Move one batch of contiguous validation contexts to the model device.
        y = full_targets[start:start + batch_size].to(device)  # [Codex comment] Move the corresponding next-token labels to the model device.
        _, loss = model(x, y)  # [Codex comment] Compute mean cross-entropy for this full-context validation batch.
        batch_tokens = y.numel()  # [Codex comment] Count targets represented by this batch for correct weighted averaging.
        total_loss += loss.item() * batch_tokens  # [Codex comment] Convert batch mean loss into its summed token contribution.
        evaluated_tokens += batch_tokens  # [Codex comment] Add this batch's targets to the total validation coverage.

    if full_pair_count < total_target_tokens:  # [Codex comment] Include the final validation remainder instead of silently dropping it.
        x = val_data[full_pair_count:-1].unsqueeze(0).to(device)  # [Codex comment] Build the last shorter runtime context from remaining validation tokens.
        y = val_data[full_pair_count + 1:].unsqueeze(0).to(device)  # [Codex comment] Align all remaining next-token labels with that shorter context.
        _, loss = model(x, y)  # [Codex comment] Evaluate the remainder using the model's supported variable runtime length.
        remainder_tokens = y.numel()  # [Codex comment] Count the targets in the shorter final validation context.
        total_loss += loss.item() * remainder_tokens  # [Codex comment] Add the remainder's token-weighted loss to the full validation sum.
        evaluated_tokens += remainder_tokens  # [Codex comment] Complete the count of evaluated validation targets.

validation_loss = total_loss / evaluated_tokens  # [Codex comment] Compute mean cross-entropy over the entire validation split.
validation_perplexity = math.exp(validation_loss)  # [Codex comment] Convert cross-entropy to perplexity for easier language-model interpretation.

print("Validation results:")  # [Codex comment] Introduce the final metrics from the loaded trained model.
print(f"  Attention type: {attention_type}")  # [Codex comment] Repeat the attention implementation beside the reported validation metric.
print(f"  Validation tokens evaluated: {evaluated_tokens:,} / {total_target_tokens:,}")  # [Codex comment] Confirm that every available validation target was included.
print(f"  Validation loss: {validation_loss:.6f}")  # [Codex comment] Print the full-split token-weighted mean cross-entropy loss.
print(f"  Validation perplexity: {validation_perplexity:.6f}")  # [Codex comment] Print the equivalent perplexity derived from validation loss.
