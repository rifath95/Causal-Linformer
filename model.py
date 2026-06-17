import torch
import torch.nn as nn
import torch.nn.functional as F

from config import *
from data import vocab_size, get_batch


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(d_hidden, 3*d_hidden, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.mix = nn.Linear(d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each [B,T,d_hidden]
        q = q.view(B, T, n_heads, d_head).transpose(1, 2).contiguous()  # [B,n_head,T,d_head]
        k = k.view(B, T, n_heads, d_head).transpose(1, 2).contiguous()
        v = v.view(B, T, n_heads, d_head).transpose(1, 2).contiguous()

        wei = (q @ k.transpose(-1, -2)) * (d_head**-0.5) # [B,n_head,T_q,d_head] @ [B,n_head,d_head,T_k] = [B,n_head,T_q,T_k]
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1) 
        wei = self.drop(wei)
        attn = wei @ v # [B,n_head,T_q,T_k] @ [B,n_head,T_k,d_head] = [B,n_head,T_q,d_head]

        attn = attn.transpose(1, 2).reshape(B, T, d_hidden)  # [B,T,d_hidden]
        attn = self.mix(attn)
        attn = self.drop(attn)
        return attn


class CausalBlockwiseLinformerAttention(nn.Module):  
    def __init__(self):  
        super().__init__()   
        self.qkv = nn.Linear(d_hidden, 3 * d_hidden, bias=False)  
        projection_shape = (num_global_blocks, causal_block_size) if share_linformer_projections_across_heads else (n_heads, num_global_blocks, causal_block_size)  # [Codex comment] Choose shared or per-head E/F shapes directly from config.py.
        self.E = nn.Parameter(torch.full(projection_shape, 1.0 / causal_block_size))  # [Codex comment] Initialize key compression as average pooling within each block.
        self.F = nn.Parameter(torch.full(projection_shape, 1.0 / causal_block_size))  # [Codex comment] Initialize value compression as average pooling within each block.
        self.mix = nn.Linear(d_hidden, d_hidden)  # [Codex comment] Mix merged attention heads using the configured model width.
        self.drop = nn.Dropout(dropout)  # [Codex comment] Apply config dropout to attention probabilities and projected output.

    def _build_local_mask(self, length, device):  # [Codex comment] Build the dense exact-local causal mask for a runtime prefix.
        positions = torch.arange(length, device=device)  # [Codex comment] Create query and key position indices on the input device.
        distance = positions[:, None] - positions[None, :]  # [Codex comment] Measure how far each key lies behind each query.
        return (distance >= 0) & (distance < local_window)  # [Codex comment] Allow only current and recent keys inside the config local window.

    def _build_global_mask(self, length, num_completed_blocks, device):  # [Codex comment] Build the inclusive causal mask over completed global blocks.
        query_positions = torch.arange(length, device=device)[:, None]  # [Codex comment] Represent every runtime query position as a mask row.
        block_ends = (torch.arange(num_completed_blocks, device=device) + 1) * causal_block_size - 1  # [Codex comment] Compute the final token position of each completed block.
        return block_ends[None, :] <= query_positions  # [Codex comment] Allow a block starting at its final token because its full contents are then causal.

    def _compress_completed_blocks(self, k, v, num_completed_blocks):  # [Codex comment] Compress only blocks fully present in the current runtime prefix.
        B, H, _, Dh = k.shape  # [Codex comment] Read dimensions needed to reshape completed key and value tokens.
        completed_length = num_completed_blocks * causal_block_size  # [Codex comment] Exclude the trailing partial block from global compression.
        k_blocks = k[:, :, :completed_length].reshape(B, H, num_completed_blocks, causal_block_size, Dh)  # [Codex comment] Group completed keys by their fixed global block.
        v_blocks = v[:, :, :completed_length].reshape(B, H, num_completed_blocks, causal_block_size, Dh)  # [Codex comment] Group completed values by their fixed global block.
        if share_linformer_projections_across_heads:  # [Codex comment] Select the einsum form when config shares compression weights across heads.
            k_tilde = torch.einsum("bhkrd,kr->bhkd", k_blocks, self.E[:num_completed_blocks])  # [Codex comment] Produce one learned compressed key per completed block.
            v_tilde = torch.einsum("bhkrd,kr->bhkd", v_blocks, self.F[:num_completed_blocks])  # [Codex comment] Produce one learned compressed value per completed block.
        else:  # [Codex comment] Use independent blockwise compression weights for each head.
            k_tilde = torch.einsum("bhkrd,hkr->bhkd", k_blocks, self.E[:, :num_completed_blocks])  # [Codex comment] Produce per-head compressed keys for completed blocks.
            v_tilde = torch.einsum("bhkrd,hkr->bhkd", v_blocks, self.F[:, :num_completed_blocks])  # [Codex comment] Produce per-head compressed values for completed blocks.
        return k_tilde, v_tilde  # [Codex comment] Return summaries used by the global attention branch.

    def forward(self, x):  # x.shape = [B,T,C]
        B, T, _ = x.shape  
        if T > block_size:  
            raise ValueError("runtime sequence length exceeds block_size")  
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)  
        q = q.view(B, T, n_heads, d_head).transpose(1, 2)  # [B,n_head,T,d_head]
        k = k.view(B, T, n_heads, d_head).transpose(1, 2)  
        v = v.view(B, T, n_heads, d_head).transpose(1, 2) 
        scale = d_head ** -0.5  

        local_scores = (q @ k.transpose(-1, -2)) * scale  # [Codex comment] Compute dense scores for the correctness-first local implementation.
        local_mask = self._build_local_mask(T, x.device)  # [Codex comment] Restrict local scores to causal keys inside the recent window.
        local_scores = local_scores.masked_fill(~local_mask, float("-inf"))  # [Codex comment] Remove future and out-of-window keys before softmax.
        local_weights = self.drop(F.softmax(local_scores, dim=-1))  # [Codex comment] Normalize and regularize exact local attention weights.
        local_out = local_weights @ v  # [Codex comment] Aggregate exact local values for every query.

        num_completed_blocks = min(T // causal_block_size, num_global_blocks)  # [Codex comment] Count only config-sized blocks fully available in this runtime prefix.
        global_out = torch.zeros_like(local_out)  # [Codex comment] Make the global contribution exactly zero when no block is complete.
        if num_completed_blocks > 0:  # [Codex comment] Skip compression and softmax entirely for short prefixes with no global block.
            k_tilde, v_tilde = self._compress_completed_blocks(k, v, num_completed_blocks)  # [Codex comment] Summarize only fully completed key and value blocks.
            global_scores = (q @ k_tilde.transpose(-1, -2)) * scale  # [Codex comment] Score each query against available compressed keys.
            global_mask = self._build_global_mask(T, num_completed_blocks, x.device)  # [Codex comment] Enforce that each query sees only blocks completed by that position.
            global_scores = global_scores.masked_fill(~global_mask, float("-inf"))  # [Codex comment] Mask compressed blocks that are still future information for a query.
            has_global = global_mask.any(dim=-1)  # [Codex comment] Identify query rows with at least one causally available block.
            global_scores[:, :, ~has_global, :] = 0.0  # [Codex comment] Avoid all-negative-infinity softmax rows and their resulting NaNs.
            global_weights = self.drop(F.softmax(global_scores, dim=-1))  # [Codex comment] Normalize and regularize valid compressed attention weights.
            global_out = global_weights @ v_tilde  # [Codex comment] Aggregate globally compressed values for every query.
            global_out = global_out * has_global.view(1, 1, T, 1).to(global_out.dtype)  # [Codex comment] Restore exact zero output for early queries with no completed block.

        attn = local_out + global_out  # [Codex comment] Combine the two independently normalized branches required by the specification.
        attn = attn.transpose(1, 2).contiguous().view(B, T, d_hidden)  # [Codex comment] Merge attention heads back into the configured model width.
        attn = self.mix(attn)  # [Codex comment] Apply the usual output projection after merging heads.
        return self.drop(attn)  # [Codex comment] Apply output dropout and preserve the existing attention-module shape contract.


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.up_proj = nn.Linear(d_hidden, 4*d_hidden)
        self.act = nn.ReLU()
        self.down_proj = nn.Linear(4*d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.down_proj(self.act(self.up_proj(x)))
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre_attn_ln = nn.LayerNorm(d_hidden)
        if attention_type == "standard": 
            self.attn = MultiHeadAttention()  
        elif attention_type == "causal_blockwise_linformer":  
            self.attn = CausalBlockwiseLinformerAttention()  
        else:  
            raise ValueError(f"Unknown attention_type: {attention_type}")  
        self.pre_ffwd_ln = nn.LayerNorm(d_hidden)
        self.ffwd = FeedForward()

    def forward(self, x):
        x = x + self.attn(self.pre_attn_ln(x))
        x = x + self.ffwd(self.pre_ffwd_ln(x))
        return x


class TestGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, d_hidden)
        self.position_embedding_table = nn.Embedding(block_size, d_hidden)
        self.layers = nn.ModuleList([Block() for _ in range(n_layers)])
        self.final_ln = nn.LayerNorm(d_hidden)
        self.lm_head = nn.Linear(d_hidden, vocab_size)

    def forward(self, x, targets=None):  # x.shape = [B,T]
        B, T = x.shape
        token_embedding = self.token_embedding_table(x)  # [B,T,C]
        position_embedding = self.position_embedding_table(
            torch.arange(T, device=x.device))  # [T,C]
        x = token_embedding + position_embedding
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.final_ln(x))
        if targets is None:
            return logits
        else:
            loss = F.cross_entropy(logits.view(
                B*T, vocab_size), targets.view(B*T))
            return logits, loss

    @torch.no_grad()
    def generate(self, context, max_new_tokens):  # context.shape = [1,T]
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            cropped_context = context[:, -block_size:]
            logits = self(cropped_context)  # [1,T,vocab_size]
            last_logits = logits[:, -1]  # [1,vocab_size]
            last_logits = last_logits - last_logits.max()
            probs = F.softmax(last_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # [1,1]
            context = torch.cat((context, next_token), dim=-1)
        if was_training:
            self.train()
        return context
