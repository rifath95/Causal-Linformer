# Goal: Implement Causal Blockwise Linformer Attention in My PyTorch Autoregressive Transformer

I have an existing standard autoregressive Transformer implementation in PyTorch. It currently uses standard causal self-attention. I want to extend it by implementing a **causal blockwise Linformer-style attention module**.

The new attention operator should have two branches:

1. **Local exact causal attention** over a small recent window.
2. **Global compressed causal attention** over blockwise summaries of earlier keys and values.

The final attention output at position $i$ should be

$$
o_i = o_i^{\mathrm{local}} + o_i^{\mathrm{global}}.
$$

The goal is to preserve autoregressive causality while reducing the amount of full sequence attention.

---

## 1. Existing baseline

The current model likely computes hidden states

$$
X \in \mathbb{R}^{B \times T \times d_{\mathrm{model}}},
$$

then projects them to queries, keys, and values:

$$
Q = XW_Q,\qquad K = XW_K,\qquad V = XW_V.
$$

After splitting into heads,

$$
Q,K,V \in \mathbb{R}^{B \times H \times T \times d_h},
$$

where:

* $B$ is batch size,
* $T$ is sequence length,
* $H$ is number of heads,
* $d_h$ is per-head dimension.

Standard causal attention computes

$$
A_{ij}
======

\frac{q_i^T k_j}{\sqrt{d_h}},
$$

with causal mask

$$
M_{ij}
======

\begin{cases}
0, & j \le i,\
-\infty, & j > i.
\end{cases}
$$

Then

$$
o_i
===

\sum_{j\le i}\alpha_{ij}v_j,
\qquad
\alpha_{ij}
===========

\frac{\exp(q_i^Tk_j/\sqrt{d_h})}
{\sum_{\ell\le i}\exp(q_i^Tk_\ell/\sqrt{d_h})}.
$$

Equivalently,

$$
O
=

\operatorname{softmax}
\left(
\frac{QK^T}{\sqrt{d_h}} + M
\right)V.
$$

---

## 2. New intended operator

Replace standard full causal attention with

$$
o_i
===

o_i^{\mathrm{local}}
+
o_i^{\mathrm{global}}.
$$

The local part attends exactly to the most recent $w$ tokens:

$$
\mathcal L_i
============

{\max(0,i-w+1),\ldots,i}.
$$

The local output is

$$
o_i^{\mathrm{local}}
====================

\operatorname{softmax}
\left(
\frac{q_iK_{\mathcal L_i}^{T}}{\sqrt{d_h}}
\right)
V_{\mathcal L_i}.
$$

The global part attends to compressed block summaries of keys and values.

---

## 3. Blockwise Linformer compression

Let the sequence length be $T$ and the number of compressed global blocks be $k$.

Assume initially that $T$ is divisible by $k$. Define the block size

$$
b=\frac{T}{k}.
$$

Using zero-indexed positions, define block $a$ as

$$
B_a
===

{ab,ab+1,\ldots,(a+1)b-1},
\qquad
a=0,\ldots,k-1.
$$

For each block, learn sequence-compression weights $E$ and $F$ that summarize keys and values inside the block:

$$
\tilde{k}_a
===========

\sum_{r=0}^{b-1}
E_{a,r}k_{ab+r},
\qquad
\tilde{v}_a
===========

\sum_{r=0}^{b-1}
F_{a,r}v_{ab+r}.
$$

So the compressed keys and values are

$$
\widetilde K
============

{\tilde{k}*0,\ldots,\tilde{k}*{k-1}},
\qquad
\widetilde V
============

{\tilde{v}*0,\ldots,\tilde{v}*{k-1}}.
$$

Equivalently, this is a block-sparse Linformer projection:

$$
\widetilde K = EK,
\qquad
\widetilde V = FV.
$$

Do not construct a full dense $T\times T$ projection matrix. Instead, parameterize the blockwise projection directly.

A good implementation is to store learned projection weights as either per-head:

$$
E,F\in\mathbb{R}^{H\times k\times b},
$$

or shared across heads:

$$
E,F\in\mathbb{R}^{k\times b}.
$$

Prefer per-head weights if easy, but make sharing configurable if convenient.

---

## 4. Causal global rule

A query at position $i$ may only attend to compressed blocks that are fully available by position $i$.

Block $a$ ends at position

$$
(a+1)b-1.
$$

Therefore query $i$ can attend to compressed block $a$ if and only if

$$
(a+1)b-1 \le i.
$$

Define

$$
\mathcal G_i
============

{a : (a+1)b-1 \le i}.
$$

Then the global output is

$$
o_i^{\mathrm{global}}
=====================

\operatorname{softmax}
\left(
\frac{q_i\widetilde K_{\mathcal G_i}^{T}}{\sqrt{d_h}}
\right)
\widetilde V_{\mathcal G_i}.
$$

If $\mathcal G_i$ is empty, set

$$
o_i^{\mathrm{global}}=0.
$$

The final output is

$$
o_i
===

\operatorname{softmax}
\left(
\frac{q_iK_{\mathcal L_i}^{T}}{\sqrt{d_h}}
\right)
V_{\mathcal L_i}
+
\operatorname{softmax}
\left(
\frac{q_i\widetilde K_{\mathcal G_i}^{T}}{\sqrt{d_h}}
\right)
\widetilde V_{\mathcal G_i}.
$$

Important: this global rule is inclusive. At the last token of a block, the query is allowed to attend to that block’s compressed summary. This is still causal, because the block contains no future tokens relative to that query.

---

## 5. Toy example: $T=12$, $k=3$, local window $w=3$

Here $T=12$, $k=3$, so

$$
b=\frac{12}{3}=4.
$$

The compressed blocks are:

$$
\tilde{k}_0 \sim k_0:k_3,
\qquad
\tilde{k}_1 \sim k_4:k_7,
\qquad
\tilde{k}*2 \sim k_8:k*{11}.
$$

In one-indexed notation for readability:

$$
\tilde{k}_1 \sim k_1:k_4,
\qquad
\tilde{k}_2 \sim k_5:k_8,
\qquad
\tilde{k}*3 \sim k_9:k*{12}.
$$

The intended attention pattern is:

| Query    | Local exact keys       | Global compressed keys                |
| -------- | ---------------------- | ------------------------------------- |
| $q_1$    | $k_1$                  | none                                  |
| $q_2$    | $k_1,k_2$              | none                                  |
| $q_3$    | $k_1,k_2,k_3$          | none                                  |
| $q_4$    | $k_2,k_3,k_4$          | $\tilde{k}_1$                         |
| $q_5$    | $k_3,k_4,k_5$          | $\tilde{k}_1$                         |
| $q_6$    | $k_4,k_5,k_6$          | $\tilde{k}_1$                         |
| $q_7$    | $k_5,k_6,k_7$          | $\tilde{k}_1$                         |
| $q_8$    | $k_6,k_7,k_8$          | $\tilde{k}_1,\tilde{k}_2$             |
| $q_9$    | $k_7,k_8,k_9$          | $\tilde{k}_1,\tilde{k}_2$             |
| $q_{10}$ | $k_8,k_9,k_{10}$       | $\tilde{k}_1,\tilde{k}_2$             |
| $q_{11}$ | $k_9,k_{10},k_{11}$    | $\tilde{k}_1,\tilde{k}_2$             |
| $q_{12}$ | $k_{10},k_{11},k_{12}$ | $\tilde{k}_1,\tilde{k}_2,\tilde{k}_3$ |

This table is the intended behavior.

The important point is that $q_4$ still gets information about $k_1$ through $\tilde{k}_1$. Therefore there is no gap caused by the local window.

---

## 6. Implementation requirements

Please inspect my existing Transformer code and implement this cleanly.

Add a new attention class, for example:

```python
class CausalBlockwiseLinformerAttention(nn.Module):
    ...
```

It should be usable as a drop-in replacement for the existing causal self-attention module.

Add config options such as:

```python
attention_type: str = "standard"
local_window: int = 3
num_global_blocks: int = 3
share_linformer_projections_across_heads: bool = False
dropout: float = 0.0
```

The module should:

1. Compute $Q,K,V$ as usual.
2. Compute local causal attention over a sliding window of size `local_window`.
3. Compute compressed global keys and values using learned blockwise $E,F$.
4. Apply a causal global block mask so each query sees only completed blocks.
5. Sum local and global outputs.
6. Apply the usual output projection.
7. Preserve tensor shapes expected by the rest of the Transformer.

The default model behavior should remain standard attention unless `attention_type="causal_blockwise_linformer"` is explicitly selected.

---

## 7. Tensor shape details

Assume that after projection and head splitting,

```python
q, k, v: [B, H, T, Dh]
```

Let:

```python
num_global_blocks = K_blocks
block_size = T // K_blocks
```

For the first implementation, require:

```python
T % K_blocks == 0
```

If it is easy to support padding later, that can be added, but correctness is more important.

View keys and values as blocks:

```python
k_blocks = k.view(B, H, K_blocks, block_size, Dh)
v_blocks = v.view(B, H, K_blocks, block_size, Dh)
```

If using per-head projection weights:

```python
E: [H, K_blocks, block_size]
F: [H, K_blocks, block_size]
```

Then compute compressed keys and values:

```python
k_tilde: [B, H, K_blocks, Dh]
v_tilde: [B, H, K_blocks, Dh]
```

Mathematically:

$$
\tilde{k}_{B,H,a,:}
===================

\sum_{r=0}^{b-1}
E_{H,a,r}k_{B,H,a,r,:}.
$$

Same for values using $F$.

A possible implementation is:

```python
k_tilde = torch.einsum("bhkrd,hkr->bhkd", k_blocks, E)
v_tilde = torch.einsum("bhkrd,hkr->bhkd", v_blocks, F)
```

Here:

* `b` in the einsum string is batch,
* `h` is head,
* `k` is block index,
* `r` is position inside block,
* `d` is head dimension.

If projections are shared across heads, use:

```python
E: [K_blocks, block_size]
F: [K_blocks, block_size]
```

and:

```python
k_tilde = torch.einsum("bhkrd,kr->bhkd", k_blocks, E)
v_tilde = torch.einsum("bhkrd,kr->bhkd", v_blocks, F)
```

---

## 8. Local attention implementation

For simplicity and correctness, the first implementation may compute local attention using a dense $T\times T$ score matrix plus a local causal mask.

The local mask is:

$$
M^{\mathrm{local}}_{ij}
=======================

\begin{cases}
0, & 0 \le i-j < w,\
-\infty, & \text{otherwise}.
\end{cases}
$$

In code:

```python
scores = q @ k.transpose(-2, -1) / math.sqrt(Dh)
scores = scores.masked_fill(~local_mask, float("-inf"))
attn = torch.softmax(scores, dim=-1)
local_out = attn @ v
```

The local mask should allow only positions satisfying:

```python
0 <= i - j < local_window
```

This gives exact local causal attention.

A later optimized implementation can replace this dense masked local attention with a true sliding-window or banded attention implementation.

---

## 9. Global compressed attention implementation

Compute global scores:

```python
global_scores = q @ k_tilde.transpose(-2, -1) / math.sqrt(Dh)
```

Shapes:

```python
q:             [B, H, T, Dh]
k_tilde:       [B, H, K_blocks, Dh]
global_scores: [B, H, T, K_blocks]
```

Build a causal global block mask of shape:

```python
global_mask: [T, K_blocks]
```

For each query position `i` and block index `a`, allow if:

```python
(a + 1) * block_size - 1 <= i
```

Equivalently:

```python
(a + 1) * block_size <= i + 1
```

Then apply:

```python
global_scores = global_scores.masked_fill(~global_mask, float("-inf"))
```

Early positions may have no valid global block. If a row has all entries equal to `-inf`, softmax will produce NaNs.

Handle this carefully.

One clean approach is:

```python
has_global = global_mask.any(dim=-1)  # [T]
```

For rows with no global block, temporarily set scores to zero before softmax, then zero out the resulting output afterward.

For example:

```python
safe_global_scores = global_scores.masked_fill(~global_mask, float("-inf"))

# Avoid NaNs for rows with no valid global blocks.
no_global = ~has_global
safe_global_scores[:, :, no_global, :] = 0.0

global_attn = torch.softmax(safe_global_scores, dim=-1)
global_out = global_attn @ v_tilde

global_out = global_out * has_global.view(1, 1, T, 1).to(global_out.dtype)
```

Please adjust the indexing carefully to match PyTorch broadcasting rules.

Final attention output before merging heads:

```python
out = local_out + global_out
```

Then merge heads and apply the usual output projection.

---

## 10. Complexity

Let:

* $T$ be sequence length,
* $d_h$ be head dimension,
* $w$ be local window size,
* $k$ be number of compressed global blocks.

Ignoring batch and number of heads:

### Local branch

If implemented with true sliding-window attention:

$$
O(Twd_h).
$$

If initially implemented with a dense masked score matrix:

$$
O(T^2d_h).
$$

The dense version is acceptable for the first correctness pass, but the intended mathematical operator has local cost $O(Twd_h)$.

### Compression branch

Each key and value participates in exactly one block summary, so compression costs:

$$
O(Td_h).
$$

### Global branch

Each query attends to at most $k$ compressed blocks, so:

$$
O(Tkd_h).
$$

### Intended total complexity

With optimized local attention:

$$
O(T(w+k)d_h).
$$

For fixed $w$ and fixed $k$, this is linear in $T$.

The intended attention-weight memory is:

$$
O(Tw + Tk),
$$

instead of standard attention memory:

$$
O(T^2).
$$

---

## 11. Initialization of $E,F$

Initialize the block projection weights stably.

Use average pooling initialization:

$$
E_{a,r}=\frac{1}{b},
\qquad
F_{a,r}=\frac{1}{b}.
$$

In code:

```python
nn.init.constant_(self.E, 1.0 / block_size)
nn.init.constant_(self.F, 1.0 / block_size)
```

Optionally add very small noise around this average initialization, but the simple average initialization is preferred for the first implementation.

---

## 12. Tests and sanity checks

Please add small tests or sanity checks.

### Shape test

For random input:

```python
x: [B, T, d_model]
```

the new attention output should have shape:

```python
[B, T, d_model]
```

matching the standard attention module.

### Causality test

Changing future tokens should not affect earlier outputs.

Procedure:

1. Create two inputs `x1` and `x2`.
2. Make them identical up to position `t`.
3. Change only positions after `t`.
4. Run the attention module on both.
5. Check that outputs up to position `t` are identical up to numerical tolerance.

This is essential.

### Global mask behavior test

For $T=12$, $k=3$, and $w=3$, verify that the global block mask behaves as follows in zero-indexed positions:

* positions `0,1,2`: no global blocks,
* position `3`: block `0` allowed,
* positions `4,5,6`: block `0` allowed,
* position `7`: blocks `0,1` allowed,
* positions `8,9,10`: blocks `0,1` allowed,
* position `11`: blocks `0,1,2` allowed.

This corresponds to the one-indexed table:

* $q_1,q_2,q_3$: no global block,
* $q_4$: $\tilde{k}_1$ allowed,
* $q_8$: $\tilde{k}_1,\tilde{k}_2$ allowed,
* $q_{12}$: $\tilde{k}_1,\tilde{k}_2,\tilde{k}_3$ allowed.

### No-NaN test

Ensure outputs contain no NaNs, especially for early positions with no valid global block.

---

## 13. Integration requirements

Please integrate this cleanly with the existing codebase.

Do not rewrite the whole model unnecessarily.

Prefer adding a config option such as:

```python
attention_type = "standard"
```

or:

```python
attention_type = "causal_blockwise_linformer"
```

The default should remain standard attention.

If the code has a Transformer block class, modify it so that it selects the attention implementation based on the config.

Example:

```python
if config.attention_type == "standard":
    self.attn = CausalSelfAttention(config)
elif config.attention_type == "causal_blockwise_linformer":
    self.attn = CausalBlockwiseLinformerAttention(config)
else:
    raise ValueError(f"Unknown attention_type: {config.attention_type}")
```

Preserve the existing training and generation APIs as much as possible.

---

## 14. Autoregressive generation note

For now, it is acceptable if generation recomputes attention over the whole current context.

Do not implement an optimized KV cache unless it is straightforward.

The priority order is:

1. correctness,
2. clean integration,
3. causal behavior,
4. no NaNs,
5. tests,
6. only then optimization.

---

## 15. Deliverables

Please produce:

1. A new attention module implementing this operator.
2. Any config changes needed to select it.
3. Integration into the Transformer block.
4. Tests or minimal sanity checks.
5. Short comments in the code explaining:

   * local attention branch,
   * blockwise global compression,
   * causal global block mask,
   * why early positions need special no-global handling.

The source-of-truth mathematical operator is:

$$
o_i
===

\operatorname{Attn}(q_i,K_{\mathcal L_i},V_{\mathcal L_i})
+
\operatorname{Attn}(q_i,\widetilde K_{\mathcal G_i},\widetilde V_{\mathcal G_i}),
$$

where

$$
\mathcal L_i
============

{\max(0,i-w+1),\ldots,i},
$$

$$
\mathcal G_i
============

{a:(a+1)b-1\le i},
$$

and

$$
\tilde{k}_a
===========

\sum_{r=0}^{b-1}
E_{a,r}k_{ab+r},
\qquad
\tilde{v}_a
===========

\sum_{r=0}^{b-1}
F_{a,r}v_{ab+r}.
$$

Use this as the source of truth.
