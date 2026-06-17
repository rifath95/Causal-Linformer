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
        projection_shape = (num_global_blocks, causal_block_size) if share_linformer_projections_across_heads else (n_heads, num_global_blocks, causal_block_size)  #  Choose shared or per-head E/F shapes directly from config.py.
        self.E = nn.Parameter(torch.full(projection_shape, 1.0 / causal_block_size))  # Initialize key and value compression as averaging within each block.
        self.F = nn.Parameter(torch.full(projection_shape, 1.0 / causal_block_size))  
        self.mix = nn.Linear(d_hidden, d_hidden)  
        self.drop = nn.Dropout(dropout)  

    def _build_local_mask(self, length, device):  
        positions = torch.arange(length, device=device)  
        distance = positions[:, None] - positions[None, :]  #  Measure how far each key lies behind each query.
        return (distance >= 0) & (distance < local_window)  # Allow only current and recent keys inside the config local window.

    def _build_global_mask(self, length, num_completed_blocks, device):  # Build the inclusive causal mask over completed global blocks.
        query_positions = torch.arange(length, device=device)[:, None]  # Represent every runtime query position as a mask row.
        block_ends = (torch.arange(num_completed_blocks, device=device) + 1) * causal_block_size - 1  # Compute the final token position of each completed block.
        return block_ends[None, :] <= query_positions  #  Allow a block starting at its final token because its full contents are then causal.

    def _compress_completed_blocks(self, k, v, num_completed_blocks):  #  Compress only blocks fully present in the current runtime prefix.
        B, H, _, C = k.shape  
        completed_length = num_completed_blocks * causal_block_size  
        k_blocks = k[:, :, :completed_length].reshape(B, H, num_completed_blocks, causal_block_size, C)  #  Group completed keys by their fixed global block.
        v_blocks = v[:, :, :completed_length].reshape(B, H, num_completed_blocks, causal_block_size, C)  #  Group completed values by their fixed global block.
        if share_linformer_projections_across_heads:  #  Select the einsum form when config shares compression weights across heads.
            k_tilde = torch.einsum("bhkrd,kr->bhkd", k_blocks, self.E[:num_completed_blocks])  
            v_tilde = torch.einsum("bhkrd,kr->bhkd", v_blocks, self.F[:num_completed_blocks])  
        else:  # Use independent blockwise compression weights for each head.
            k_tilde = torch.einsum("bhkrd,hkr->bhkd", k_blocks, self.E[:, :num_completed_blocks])  
            v_tilde = torch.einsum("bhkrd,hkr->bhkd", v_blocks, self.F[:, :num_completed_blocks])  
        return k_tilde, v_tilde  # [B,n_head,num_completed_blocks,d_head]

    def forward(self, x):  # x.shape = [B,T,C]
        B, T, _ = x.shape  
        if T > block_size:  
            raise ValueError("runtime sequence length exceeds block_size")  
        q, k, v = torch.chunk(self.qkv(x), 3, dim=-1)  
        q = q.view(B, T, n_heads, d_head).transpose(1, 2)  # [B,n_head,T,d_head]
        k = k.view(B, T, n_heads, d_head).transpose(1, 2)  
        v = v.view(B, T, n_heads, d_head).transpose(1, 2) 
        scale = d_head ** -0.5  

        local_scores = (q @ k.transpose(-1, -2)) * scale  #  Compute dense scores for the correctness-first local implementation.
        local_mask = self._build_local_mask(T, x.device)  #  Restrict local scores to causal keys inside the recent window.
        local_scores = local_scores.masked_fill(~local_mask, float("-inf"))  # Remove future and out-of-window keys before softmax.
        local_scores = F.softmax(local_scores, dim=-1)
        local_scores = self.drop(local_scores)  
        local_out = local_scores @ v  # [B,n_head,T,d_head]

        num_completed_blocks = min(T // causal_block_size, num_global_blocks)  
        global_out = torch.zeros_like(local_out)  
        if num_completed_blocks > 0:  # Skip compression and softmax entirely for short prefixes with no global block.
            k_tilde, v_tilde = self._compress_completed_blocks(k, v, num_completed_blocks)  # [B,n_head,num_completed_blocks,d_head]. Summarize only fully completed key and value blocks. 
            global_scores = (q @ k_tilde.transpose(-1, -2)) * scale  # [B,n_head,T,num_completed_blocks]
            
            global_mask = self._build_global_mask(T, num_completed_blocks, x.device)  # Enforce that each query sees only blocks completed by that position.
            global_scores = global_scores.masked_fill(~global_mask, float("-inf"))  #  Mask compressed blocks that are still future information for a query.
            has_global = global_mask.any(dim=-1)  # Identify query rows with at least one causally available block.
            global_scores = global_scores.masked_fill((~has_global).view(1, 1, T, 1), 0.0)  # Avoid all-negative-infinity softmax rows and their resulting NaNs.
            
            global_scores = F.softmax(global_scores, dim=-1)
            global_scores = global_scores.masked_fill((~has_global).view(1, 1, T, 1), 0.0) # Remove the fake softmax probabilities used only to avoid NaNs for queries with no completed global block.
            global_scores = self.drop(global_scores)  
            global_out = global_scores @ v_tilde  # [B,n_head,T,num_completed_blocks] @ [B,n_head,num_completed_blocks,d_head] = [B,n_head,T,d_head]
            
        attn = local_out + global_out  # Combine the two independently normalized branches 
        attn = attn.transpose(1, 2).reshape(B, T, d_hidden)  
        attn = self.mix(attn)  
        attn = self.drop(attn)
        return attn  


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


class MyGPT(nn.Module):
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
