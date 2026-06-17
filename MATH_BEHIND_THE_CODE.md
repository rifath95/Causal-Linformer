# Math Behind the Code

This document reduces the code in this project to the mathematical operations it implements. It follows the model in chronological order, records the important tensor shapes, compares the two available attention mechanisms, calculates their parameter counts and complexity, and explains the reasoning behind the causal blockwise Linformer design.

The current configuration is:

| Symbol | Config value | Meaning |
| --- | ---: | --- |
| \(B\) | \(64\) | Training batch size |
| \(T_{\max}\) | \(256\) | Maximum context length |
| \(d_{\mathrm{model}}\) | \(384\) | Hidden dimension |
| \(L\) | \(6\) | Number of Transformer blocks |
| \(H\) | \(6\) | Number of attention heads |
| \(d_h\) | \(64\) | Dimension of each attention head |
| \(w\) | \(3\) | Exact local attention window, derived as \(b-1\) |
| \(K\) | \(64\) | Maximum number of compressed global blocks |
| \(b\) | \(4\) | Number of tokens in each global block |
| \(V\) | \(64\) | Character vocabulary size for the current dataset |

The head dimension is

```math
d_h = \frac{d_{\mathrm{model}}}{H}
    = \frac{384}{6}
    = 64.
```

The fixed Linformer block width is

```math
b = \frac{T_{\max}}{K}
  = \frac{256}{64}
  = 4.
```

In `config.py`, this same value is named `causal_block_size`:

```python
causal_block_size = 4
```

The local attention window is derived from the causal block size:

```math
w = b - 1 = 4 - 1 = 3.
```

The code supports two values of `attention_type`:

1. `standard`
2. `causal_blockwise_linformer`

The current default in `config.py` is `standard`. Switching to `causal_blockwise_linformer` activates the local-plus-compressed-global attention module. The main model class is `MyGPT`; everything outside the attention sublayer is shared by the two attention modes.

## Character Tokenization and Dataset

The model is a character-level language model. Let the complete text contain \(N\) characters. The code builds the sorted set of unique characters:

```math
\mathcal{V} = \{c_0,c_1,\ldots,c_{V-1}\}.
```

Each character is assigned an integer:

```math
\mathrm{encode}(c_i)=i.
```

The complete text becomes a one-dimensional token tensor:

```math
D \in \{0,1,\ldots,V-1\}^{N}.
```

The first \(90\%\) of the tensor is used for training and the final \(10\%\) is used for validation:

```math
D_{\mathrm{train}} = D_{0:\lfloor0.9N\rfloor},
```

```math
D_{\mathrm{val}} = D_{\lfloor0.9N\rfloor:N}.
```

For a sampled start position \(s\), a training input is

```math
X = D_{s:s+T},
```

and its target is the same sequence shifted by one character:

```math
Y = D_{s+1:s+T+1}.
```

Therefore, the model at position \(t\) is trained to predict the character at position \(t+1\).

For a batch:

```math
X,Y \in \{0,1,\ldots,V-1\}^{B\times T}.
```

## Token and Position Embeddings

The token embedding table is

```math
W_{\mathrm{tok}} \in \mathbb{R}^{V\times d_{\mathrm{model}}}.
```

For this dataset:

```math
W_{\mathrm{tok}} \in \mathbb{R}^{64\times384}.
```

Every token id selects one row:

```math
E_{\mathrm{tok}}[b,t,:]
=
W_{\mathrm{tok}}[X[b,t],:].
```

Thus,

```math
E_{\mathrm{tok}} \in \mathbb{R}^{B\times T\times384}.
```

The model also uses learned absolute position embeddings:

```math
W_{\mathrm{pos}} \in \mathbb{R}^{T_{\max}\times d_{\mathrm{model}}}
=
\mathbb{R}^{256\times384}.
```

For a runtime sequence length \(T_{\mathrm{cur}}\), the positions are

```math
0,1,\ldots,T_{\mathrm{cur}}-1,
```

and the initial residual stream is

```math
X^{(0)} = E_{\mathrm{tok}} + E_{\mathrm{pos}}
\in
\mathbb{R}^{B\times T_{\mathrm{cur}}\times384}.
```

Parameter counts:

```math
N(W_{\mathrm{tok}})=64\cdot384=24{,}576,
```

```math
N(W_{\mathrm{pos}})=256\cdot384=98{,}304.
```

## Transformer Block

The model contains \(L=6\) identical-structure Transformer blocks with independent parameters.

If \(X^{(\ell)}\) enters block \(\ell\), the pre-normalized attention residual update is

```math
Y^{(\ell)}
=
X^{(\ell)}
+
\mathrm{Attention}_{\ell}
\left(
\mathrm{LayerNorm}_{\ell,\mathrm{attn}}
(X^{(\ell)})
\right).
```

The feedforward residual update is

```math
X^{(\ell+1)}
=
Y^{(\ell)}
+
\mathrm{FFN}_{\ell}
\left(
\mathrm{LayerNorm}_{\ell,\mathrm{ffn}}
(Y^{(\ell)})
\right).
```

Both residual additions preserve the shape

```math
\mathbb{R}^{B\times T_{\mathrm{cur}}\times384}.
```

## Layer Normalization

For one token vector

```math
x\in\mathbb{R}^{d_{\mathrm{model}}},
```

LayerNorm computes the coordinate mean

```math
\mu(x)
=
\frac{1}{d_{\mathrm{model}}}
\sum_{c=1}^{d_{\mathrm{model}}}x_c,
```

and variance

```math
\sigma^2(x)
=
\frac{1}{d_{\mathrm{model}}}
\sum_{c=1}^{d_{\mathrm{model}}}
(x_c-\mu(x))^2.
```

The normalized output is

```math
\mathrm{LayerNorm}(x)_c
=
\gamma_c
\frac{x_c-\mu(x)}
{\sqrt{\sigma^2(x)+\epsilon}}
+
\beta_c,
```

where

```math
\gamma,\beta\in\mathbb{R}^{384}
```

are learned parameters.

Each LayerNorm therefore has

```math
2d_{\mathrm{model}}=768
```

parameters.

Pre-normalization gives gradients a direct identity route through each residual addition:

```math
X\longmapsto X+f(\mathrm{LayerNorm}(X)).
```

## Shared Query, Key, and Value Projection

Both attention implementations begin in the same way.

Given

```math
X\in\mathbb{R}^{B\times T_{\mathrm{cur}}\times384},
```

one bias-free linear layer computes all queries, keys, and values:

```math
[Q_{\mathrm{flat}};K_{\mathrm{flat}};V_{\mathrm{flat}}]
=
XW_{\mathrm{QKV}},
```

where

```math
W_{\mathrm{QKV}}
\in
\mathbb{R}^{384\times(3\cdot384)}.
```

The result is split and rearranged into heads:

```math
Q,K,V
\in
\mathbb{R}^{B\times H\times T_{\mathrm{cur}}\times d_h}
=
\mathbb{R}^{B\times6\times T_{\mathrm{cur}}\times64}.
```

The QKV projection contains

```math
384\cdot(3\cdot384)
=
442{,}368
```

parameters per Transformer block.

## Standard Causal Multi-Head Attention

The standard implementation lets each query attend exactly to all keys at the same or earlier positions.

### Scaled Dot-Product Scores

For each batch item and head:

```math
S
=
\frac{QK^T}{\sqrt{d_h}}
\in
\mathbb{R}^{B\times H\times T_{\mathrm{cur}}\times T_{\mathrm{cur}}}.
```

At individual positions:

```math
S_{b,h,i,j}
=
\frac{
Q_{b,h,i,:}\cdot K_{b,h,j,:}
}
{\sqrt{64}}.
```

The factor

```math
\frac{1}{\sqrt{d_h}}
=
\frac{1}{8}
```

keeps dot-product magnitudes from growing with the head dimension.

### Causal Mask

The lower-triangular mask allows

```math
j\le i
```

and forbids

```math
j>i.
```

Equivalently,

```math
M_{i,j}
=
\begin{cases}
0, & j\le i,\\
-\infty, & j>i.
\end{cases}
```

The masked probabilities are

```math
A
=
\mathrm{softmax}(S+M)
\in
\mathbb{R}^{B\times H\times T_{\mathrm{cur}}\times T_{\mathrm{cur}}}.
```

Every row sums to one before dropout:

```math
\sum_{j=0}^{T_{\mathrm{cur}}-1}A_{b,h,i,j}=1.
```

### Weighted Value Sum

The head outputs are

```math
O_{\mathrm{heads}}=AV
\in
\mathbb{R}^{B\times H\times T_{\mathrm{cur}}\times64}.
```

The heads are concatenated:

```math
O_{\mathrm{merged}}
\in
\mathbb{R}^{B\times T_{\mathrm{cur}}\times384}.
```

The output mixing layer computes

```math
O
=
O_{\mathrm{merged}}W_O+b_O,
```

where

```math
W_O\in\mathbb{R}^{384\times384},
\qquad
b_O\in\mathbb{R}^{384}.
```

This output layer has

```math
384^2+384
=
147{,}840
```

parameters per block.

### Standard Attention Complexity

Ignoring constants:

- QKV projection: \(O(BT_{\mathrm{cur}}d_{\mathrm{model}}^2)\)
- Score matrix: \(O(BHT_{\mathrm{cur}}^2d_h)\)
- Weighted value sum: \(O(BHT_{\mathrm{cur}}^2d_h)\)
- Attention-weight memory: \(O(BHT_{\mathrm{cur}}^2)\)

The quadratic dependence on sequence length comes from the

```math
T_{\mathrm{cur}}\times T_{\mathrm{cur}}
```

attention matrix.

At \(T_{\mathrm{cur}}=256\), each head has

```math
256^2=65{,}536
```

attention scores.

Across six heads and a training batch of 64:

```math
64\cdot6\cdot256^2
=
25{,}165{,}824
```

scores per layer.

## Causal Blockwise Linformer Attention

The second attention implementation replaces full causal attention with two separately normalized branches:

```math
O_i
=
O_i^{\mathrm{local}}
+
O_i^{\mathrm{global}}.
```

The local branch preserves exact recent-token attention. The global branch provides compressed information from completed fixed-size blocks.

## Exact Local Branch

For query position \(i\), the exact local key set is

```math
\mathcal{L}_i
=
\{
\max(0,i-w+1),\ldots,i
\}.
```

Because `local_window = causal_block_size - 1`, the current value is

```math
w = b - 1 = 3.
```

So:

```math
\mathcal{L}_i
=
\{\max(0,i-2),\ldots,i\}.
```

The local mask permits a query-key pair exactly when

```math
0\le i-j<w.
```

Therefore:

```math
M^{\mathrm{local}}_{i,j}
=
\begin{cases}
0, & 0\le i-j<3,\\
-\infty, & \mathrm{otherwise}.
\end{cases}
```

The local scores and probabilities are

```math
S^{\mathrm{local}}
=
\frac{QK^T}{\sqrt{d_h}}
+
M^{\mathrm{local}},
```

```math
A^{\mathrm{local}}
=
\mathrm{softmax}(S^{\mathrm{local}}).
```

The exact local output is

```math
O^{\mathrm{local}}
=
A^{\mathrm{local}}V.
```

For example:

- Position \(0\) sees exact token \(0\).
- Position \(1\) sees exact tokens \(0,1\).
- Position \(2\) sees exact tokens \(0,1,2\).
- Position \(100\) sees exact tokens \(98,99,100\).

The current code constructs a dense

```math
T_{\mathrm{cur}}\times T_{\mathrm{cur}}
```

score matrix and masks most entries. This is mathematically local attention, but its current implementation still takes

```math
O(BHT_{\mathrm{cur}}^2d_h)
```

time and

```math
O(BHT_{\mathrm{cur}}^2)
```

score memory.

A future banded or sliding-window implementation could compute only the \(w\) permitted keys per query, reducing the local branch to

```math
O(BHT_{\mathrm{cur}}wd_h)
```

time and

```math
O(BHT_{\mathrm{cur}}w)
```

attention-weight memory.

## Fixed Global Blocks

Global blocks are defined from the configured maximum context, not from the changing runtime prefix.

The fixed block width is

```math
b=\frac{T_{\max}}{K}.
```

With the current configuration:

```math
b=\frac{256}{64}=4.
```

The first few maximum-context blocks are:

```math
\mathcal{B}_0=\{0,1,2,3\},
```

```math
\mathcal{B}_1=\{4,5,6,7\},
```

```math
\mathcal{B}_2=\{8,9,10,11\},
```

```math
\mathcal{B}_3=\{12,13,14,15\}.
```

This pattern continues through

```math
\mathcal{B}_{63}=\{252,253,254,255\}.
```

For a changing runtime prefix \(T_{\mathrm{cur}}\), the number of complete blocks is

```math
K_{\mathrm{complete}}
=
\left\lfloor
\frac{T_{\mathrm{cur}}}{4}
\right\rfloor.
```

Since the model never receives more than \(256\) positions:

```math
K_{\mathrm{complete}}
=
\min\left(
\left\lfloor\frac{T_{\mathrm{cur}}}{4}\right\rfloor,
64
\right).
```

Examples:

| Runtime length \(T_{\mathrm{cur}}\) | Completed blocks | Globally compressed tokens | Partial local-only tokens |
| ---: | ---: | --- | --- |
| \(3\) | \(0\) | none | \(0\ldots2\) |
| \(4\) | \(1\) | \(0\ldots3\) | none |
| \(7\) | \(1\) | \(0\ldots3\) | \(4\ldots6\) |
| \(70\) | \(17\) | \(0\ldots67\) | \(68\ldots69\) |
| \(150\) | \(37\) | \(0\ldots147\) | \(148\ldots149\) |
| \(256\) | \(64\) | \(0\ldots255\) | none |

The partial block is never globally compressed. Its tokens participate only through the exact local branch until all four positions in that block exist.

This rule is essential for causality during generation.

With the current \(w=3\) and \(b=4\), the initial positions behave as follows:

| Query position | Exact local positions | Available global blocks |
| ---: | --- | --- |
| \(0\) | \(0\) | none |
| \(1\) | \(0,1\) | none |
| \(2\) | \(0,1,2\) | none |
| \(3\) | \(1,2,3\) | block \(0\), summarizing \(0,1,2,3\) |
| \(4\) | \(2,3,4\) | block \(0\) |

Therefore, the first three queries do not yet have a compressed key, but they can already see their entire causal history through the local branch. At position \(3\), token \(0\) leaves the three-token local window exactly when block \(0\), which contains token \(0\), becomes globally available. Under this configuration, no earlier causal token is left uncovered by both branches.

## Learned Blockwise Compression

After splitting into heads:

```math
K,V
\in
\mathbb{R}^{B\times H\times T_{\mathrm{cur}}\times d_h}.
```

Only the completed prefix is reshaped into blocks:

```math
K_{\mathrm{blocks}},V_{\mathrm{blocks}}
\in
\mathbb{R}^{B\times H\times K_{\mathrm{complete}}\times b\times d_h}.
```

With per-head projections, the learned compression parameters are

```math
E,F
\in
\mathbb{R}^{H\times K\times b}.
```

For the current config:

```math
E,F\in\mathbb{R}^{6\times64\times4}.
```

For head \(h\), block \(a\), and coordinate \(c\), the compressed key is

```math
\widetilde{K}_{B,h,a,c}
=
\sum_{r=0}^{3}
E_{h,a,r}
K_{B,h,4a+r,c}.
```

The compressed value is

```math
\widetilde{V}_{B,h,a,c}
=
\sum_{r=0}^{3}
F_{h,a,r}
V_{B,h,4a+r,c}.
```

Therefore,

```math
\widetilde{K},\widetilde{V}
\in
\mathbb{R}^{B\times H\times K_{\mathrm{complete}}\times d_h}.
```

Each completed group of four key vectors becomes one compressed key vector per head. Each completed group of four value vectors becomes one compressed value vector per head.

The code performs these sums with:

```python
k_tilde = torch.einsum("bhkrd,hkr->bhkd", k_blocks, E)
v_tilde = torch.einsum("bhkrd,hkr->bhkd", v_blocks, F)
```

Here:

- `b` is batch.
- `h` is attention head.
- `k` is global block index.
- `r` is token position inside the block.
- `d` is head coordinate.

## Compression Initialization

Every entry of \(E\) and \(F\) starts as

```math
\frac{1}{b}
=
\frac{1}{4}.
```

At initialization:

```math
\widetilde{K}_{h,a,:}
=
\frac{1}{4}
\sum_{r=0}^{3}K_{h,4a+r,:},
```

```math
\widetilde{V}_{h,a,:}
=
\frac{1}{4}
\sum_{r=0}^{3}V_{h,4a+r,:}.
```

Thus, each summary initially performs average pooling. During training, the entries of \(E\) and \(F\) are updated independently, allowing each head and block to learn which positions deserve more weight.

The weights are not passed through a softmax. Consequently, after training they are general learned linear-combination coefficients: they do not have to be positive or sum to one.

## Shared Versus Per-Head Compression

The current configuration uses

```python
share_linformer_projections_across_heads = False
```

so every head has separate \(E\) and \(F\) weights.

Per-head shapes:

```math
E,F\in\mathbb{R}^{H\times K\times b}.
```

If sharing were enabled:

```math
E,F\in\mathbb{R}^{K\times b}.
```

Sharing reduces parameters, but it forces every head to summarize token positions with the same coefficients. Per-head projections permit one head to emphasize different positions from another head.

## Causal Global Mask

Block \(a\) contains positions

```math
ab,\ldots,(a+1)b-1.
```

Its final position is

```math
\mathrm{end}(a)
=(a+1)b-1.
```

Query position \(i\) may attend to compressed block \(a\) exactly when

```math
(a+1)b-1\le i.
```

The inequality is inclusive. At the final token of a block, the entire block exists, and no token inside that block lies in the query's future.

With \(b=4\):

- Block \(0\) becomes visible at query position \(3\).
- Block \(1\) becomes visible at query position \(7\).
- Block \(2\) becomes visible at query position \(11\).
- Block \(3\) becomes visible at query position \(15\).
- Block \(63\) becomes visible at query position \(255\).

The global mask is

```math
M^{\mathrm{global}}_{i,a}
=
\begin{cases}
0, & (a+1)b-1\le i,\\
-\infty, & \mathrm{otherwise}.
\end{cases}
```

## Global Compressed Attention

The global score tensor is

```math
S^{\mathrm{global}}
=
\frac{Q\widetilde{K}^T}{\sqrt{d_h}}
\in
\mathbb{R}^{B\times H\times T_{\mathrm{cur}}\times K_{\mathrm{complete}}}.
```

After the causal block mask:

```math
A^{\mathrm{global}}
=
\mathrm{softmax}
\left(
S^{\mathrm{global}}+M^{\mathrm{global}}
\right).
```

The global output is

```math
O^{\mathrm{global}}
=
A^{\mathrm{global}}\widetilde{V}
\in
\mathbb{R}^{B\times H\times T_{\mathrm{cur}}\times d_h}.
```

At the final position of a full 256-token context, a query can attend to all 64 compressed keys:

```math
\widetilde{k}_0,\widetilde{k}_1,\ldots,\widetilde{k}_{63}.
```

Together with the three exact local keys, it attends to at most

```math
w+K=3+64=67
```

representations.

Earlier positions attend to fewer than 67 because fewer global blocks are causally available.

## Why Early Global Rows Need Special Handling

Before position \(3\), no global block is complete. Even when block \(0\) exists somewhere in the runtime tensor, query positions \(0,1,2\) are not allowed to use it.

Their global score rows would therefore contain only

```math
-\infty.
```

Softmax of an all-\(-\infty\) row is undefined numerically because it produces a \(0/0\)-type normalization, resulting in `NaN`.

The code handles this in two steps:

1. Temporarily replace all-masked score rows with zeros before softmax.
2. Multiply the resulting global output by a Boolean `has_global` indicator.

Mathematically:

```math
O_i^{\mathrm{global}}=0
\qquad
\mathrm{when\ no\ block\ is\ available\ to\ query\ }i.
```

This preserves the intended operator and prevents NaNs.

## Combining the Two Branches

The two branches use separate softmax normalizations:

```math
O_i^{\mathrm{local}}
=
\mathrm{softmax}
\left(
\frac{q_iK_{\mathcal{L}_i}^T}{\sqrt{d_h}}
\right)
V_{\mathcal{L}_i},
```

```math
O_i^{\mathrm{global}}
=
\mathrm{softmax}
\left(
\frac{q_i\widetilde{K}_{\mathcal{G}_i}^T}{\sqrt{d_h}}
\right)
\widetilde{V}_{\mathcal{G}_i}.
```

They are then added:

```math
O_i
=
O_i^{\mathrm{local}}
+
O_i^{\mathrm{global}}.
```

This is not equivalent to concatenating local and global keys and applying one softmax. Each nonempty branch independently assigns total probability mass one before dropout.

Therefore, the model receives:

- One normalized exact-local summary.
- One normalized compressed-global summary.

Their sum is passed through the output mixing projection.

## Local and Global Information Can Overlap

At a block endpoint, some information can occur in both branches.

For example, at position \(3\):

- Local exact keys are positions \(1,2,3\).
- Global block \(0\) summarizes positions \(0,1,2,3\).

Thus, positions \(1,2,3\) contribute directly through local attention and indirectly through the global block summary.

This overlap is part of the specified operator. The global branch is a learned summary of the entire completed block, while the local branch preserves exact detail for the most recent tokens.

## Linformer Complexity

### Compression

Every completed key and value is used once in its block summary:

```math
O(BHT_{\mathrm{cur}}d_h).
```

### Global Scores

Each query compares against at most \(K\) compressed keys:

```math
O(BHT_{\mathrm{cur}}Kd_h).
```

The global attention-weight tensor has at most

```math
B\times H\times T_{\mathrm{cur}}\times K
```

entries.

At \(T_{\mathrm{cur}}=256\), \(H=6\), \(K=64\), and \(B=64\):

```math
64\cdot6\cdot256\cdot64
=
6{,}291{,}456
```

global scores per layer.

### Intended Optimized Total

With true sliding-window local attention, the attention-specific cost would be

```math
O\left(
BHT_{\mathrm{cur}}(w+K)d_h
\right),
```

and attention-weight memory would be

```math
O\left(
BHT_{\mathrm{cur}}(w+K)
\right).
```

For fixed \(w\) and \(K\), this is linear in \(T_{\mathrm{cur}}\).

### Current Implemented Total

The current local branch is dense and masked. Consequently, the implementation still has a quadratic local score matrix:

```math
O(BHT_{\mathrm{cur}}^2d_h)
```

time and

```math
O(BHT_{\mathrm{cur}}^2)
```

score memory.

The current implementation is therefore a correctness-first implementation of the Linformer operator, not yet a fully linear-complexity implementation.

The global branch is compressed as intended, but the local branch must later be replaced by a banded kernel to realize the full asymptotic benefit.

## Feedforward Network

After attention, each block applies a two-layer position-wise feedforward network.

For each token vector

```math
x\in\mathbb{R}^{384},
```

the first layer expands it to four times the hidden dimension:

```math
u=xW_{\mathrm{up}}+b_{\mathrm{up}}
\in
\mathbb{R}^{1536}.
```

The activation is ReLU:

```math
\mathrm{ReLU}(u)_j
=
\max(0,u_j).
```

The second layer projects back:

```math
\mathrm{FFN}(x)
=
\mathrm{ReLU}(u)W_{\mathrm{down}}
+
b_{\mathrm{down}}
\in
\mathbb{R}^{384}.
```

Shapes:

```math
W_{\mathrm{up}}\in\mathbb{R}^{384\times1536},
\qquad
b_{\mathrm{up}}\in\mathbb{R}^{1536},
```

```math
W_{\mathrm{down}}\in\mathbb{R}^{1536\times384},
\qquad
b_{\mathrm{down}}\in\mathbb{R}^{384}.
```

Parameter count:

```math
384\cdot1536+1536
=
591{,}360,
```

```math
1536\cdot384+384
=
590{,}208.
```

Total:

```math
1{,}181{,}568
```

feedforward parameters per block.

The same feedforward transformation is applied independently to every token position.

## Final LayerNorm and Language-Model Head

After six blocks:

```math
X^{(6)}
\in
\mathbb{R}^{B\times T_{\mathrm{cur}}\times384}.
```

The final LayerNorm produces

```math
\widetilde{X}
=
\mathrm{LayerNorm}(X^{(6)}).
```

The language-model head maps each token vector to vocabulary logits:

```math
Z
=
\widetilde{X}W_{\mathrm{LM}}+b_{\mathrm{LM}},
```

where

```math
W_{\mathrm{LM}}\in\mathbb{R}^{384\times64},
\qquad
b_{\mathrm{LM}}\in\mathbb{R}^{64}.
```

Therefore:

```math
Z\in\mathbb{R}^{B\times T_{\mathrm{cur}}\times64}.
```

The token embedding and language-model head are separate parameters in this implementation. Their weights are not tied.

## Cross-Entropy Training Loss

For target token \(Y_{b,t}\), the predicted probability is

```math
p(v\mid X_{\le t})
=
\frac{\exp(Z_{b,t,v})}
{\sum_{u=0}^{V-1}\exp(Z_{b,t,u})}.
```

The batch cross-entropy is

```math
\mathcal{L}_{\mathrm{train}}
=
-\frac{1}{BT}
\sum_{b=1}^{B}
\sum_{t=1}^{T}
\log
p(Y_{b,t}\mid X_{b,\le t}).
```

`AdamW` updates all trainable parameters using this loss.

The loss plot records one scalar

```math
\mathcal{L}_{\mathrm{train}}^{(s)}
```

for every optimization step \(s\). `train.py` saves each plot in `docs/` with the attention type and a timestamp in the filename, so repeated runs do not overwrite earlier plots.

## Full Validation Loss

The validation script loads `model.pth`, switches the model to evaluation mode, and divides the validation split into contiguous contexts.

Suppose the validation split provides \(N_{\mathrm{val}}-1\) next-token targets. If chunk \(c\) contains \(n_c\) targets and has mean loss \(\mathcal{L}_c\), then the full token-weighted validation loss is

```math
\mathcal{L}_{\mathrm{val}}
=
\frac{
\sum_c n_c\mathcal{L}_c
}{
\sum_c n_c
}.
```

Weighting by \(n_c\) matters because the final chunk may be shorter than 256 tokens.

The reported perplexity is

```math
\mathrm{PPL}
=
\exp(\mathcal{L}_{\mathrm{val}}).
```

Lower validation loss and lower perplexity indicate that the model assigns more probability to the correct next characters on held-out text.

The validation script evaluates every available target token, although contexts are reset at chunk boundaries. Consequently, the first positions of each new validation chunk do not carry hidden context from the preceding chunk.

## Autoregressive Generation

Generation begins with an encoded prompt:

```math
X^{(0)}
\in
\{0,\ldots,V-1\}^{1\times T_0}.
```

At each generation step:

1. Keep only the latest 256 tokens.
2. Run the complete model on that context.
3. Select logits from the final position.
4. Convert logits to probabilities with softmax.
5. Sample one token with `torch.multinomial`.
6. Append the sampled token to the context.

If \(z\in\mathbb{R}^{V}\) is the final-position logit vector:

```math
p(v)=\frac{\exp(z_v)}{\sum_{u=0}^{V-1}\exp(z_u)}.
```

Subtracting

```math
\max_v z_v
```

before softmax does not change the probabilities:

```math
\frac{\exp(z_v-c)}
{\sum_u\exp(z_u-c)}
=
\frac{\exp(z_v)}
{\sum_u\exp(z_u)}.
```

It improves numerical stability.

The implementation does not use a KV cache. It recomputes the whole cropped context for every generated token.

For Linformer generation, the runtime sequence length does not need to be divisible by 64 or by 4. Only complete four-token blocks are globally summarized; the remaining suffix stays local-only.

## Parameter Count: Standard Attention

For one standard attention layer:

```math
N(W_{\mathrm{QKV}})
=
442{,}368,
```

```math
N(W_O,b_O)
=
147{,}840.
```

Therefore:

```math
N(\mathrm{Attention}_{\mathrm{standard}})
=
590{,}208
```

parameters per block.

The complete standard model has:

```math
10{,}788{,}160
```

parameters.

## Additional Linformer Parameters

With per-head projections:

```math
N(E)
=
H\cdot K\cdot b
=
6\cdot64\cdot4
=
1{,}536.
```

Similarly:

```math
N(F)=1{,}536.
```

Additional parameters per block:

```math
N(E)+N(F)
=
3{,}072.
```

Across six blocks:

```math
6\cdot3{,}072
=
18{,}432.
```

Therefore, the complete causal Linformer model has:

```math
10{,}788{,}160+18{,}432
=
10{,}806{,}592
```

parameters.

The percentage increase is

```math
\frac{18{,}432}{10{,}788{,}160}\cdot100
\approx
0.171\%.
```

The causal Linformer adds very few parameters because \(E\) and \(F\) compress positions, not hidden coordinates.

Changing the number of blocks from \(4\) to \(64\) does not change this parameter count because

```math
K b
=
K\frac{T_{\max}}{K}
=
T_{\max}
=
256.
```

Increasing \(K\) changes the shape and behavior of the projection:

```math
6\times4\times64
\longrightarrow
6\times64\times4,
```

but both shapes contain

```math
6\cdot256=1{,}536
```

entries for each of \(E\) and \(F\).

If projections were shared across heads, the additional parameters would instead be

```math
2Kb
=
2\cdot64\cdot4
=
512
```

per block, or

```math
6\cdot512=3{,}072
```

for the full model.

## Why Use a Local Exact Branch?

A single compressed block vector cannot preserve every token-level detail from its four positions. Recent syntax and character patterns often depend on exact nearby tokens.

The local branch guarantees that the latest three keys and values remain individually visible:

```math
\{i-2,i-1,i\}.
```

This gives the model high-resolution short-range information while global summaries carry lower-resolution long-range information.

## Why Use Completed Blocks Only?

Suppose the current partial block contains positions \(128,129,130\), while its final position \(131\) has not arrived yet. Compressing the complete four-position block for query position \(130\) would require the key and value at future position \(131\).

Waiting until the block is complete avoids this leak.

For that block ending at \(131\), its summary is first available to query \(131\):

```math
131\le131.
```

This inclusive boundary is causal because all positions summarized by that block are at or before the query.

## Why Compress Keys and Values Separately?

Keys determine attention compatibility:

```math
q_i\cdot\widetilde{k}_a.
```

Values determine the information returned after the attention weight is chosen:

```math
\alpha_{i,a}\widetilde{v}_a.
```

Using separate \(E\) and \(F\) matrices allows the model to learn one positional combination for deciding relevance and another positional combination for constructing the returned content.

## Why Keep Standard Attention as an Option?

Standard causal attention acts as the baseline:

```math
O_{\mathrm{standard}}
=
\mathrm{softmax}
\left(
\frac{QK^T}{\sqrt{d_h}}+M
\right)V.
```

Keeping both implementations under the same model, data, optimizer, and training loop makes comparisons more controlled. Differences in training loss, validation loss, model size, and generation can be attributed more directly to the attention mechanism.

## Important Practical Interpretation

With the current configuration, saying that a late Linformer query attends to at most 67 "keys" means:

- Three are exact token-level key vectors.
- Sixty-four are compressed key vectors, each summarizing four original key vectors.

The 64 global keys therefore represent information from as many as 256 source positions. They are not equivalent to 64 ordinary tokens.

The compression is lossy: each four-token block is reduced to one vector per head. Compared with the previous four-block configuration, this preserves finer global resolution and makes global attention available sooner, but it also increases the number of global comparisons per query.

## Summary of the Two Attention Operators

Standard causal attention:

```math
O_i^{\mathrm{standard}}
=
\mathrm{Attn}
\left(
q_i,
K_{\{0,\ldots,i\}},
V_{\{0,\ldots,i\}}
\right).
```

Causal blockwise Linformer attention:

```math
O_i^{\mathrm{Linformer}}
=
\mathrm{Attn}
\left(
q_i,
K_{\mathcal{L}_i},
V_{\mathcal{L}_i}
\right)
+
\mathrm{Attn}
\left(
q_i,
\widetilde{K}_{\mathcal{G}_i},
\widetilde{V}_{\mathcal{G}_i}
\right),
```

where

```math
\mathcal{L}_i
=
\{\max(0,i-w+1),\ldots,i\},
```

and

```math
\mathcal{G}_i
=
\{a:(a+1)b-1\le i\}.
```

For this project:

```math
K=64,\qquad b=4,\qquad w = b - 1 = 3.
```

The implementation preserves autoregressive causality, supports variable runtime prefix lengths, and introduces only 18,432 parameters beyond the standard model. Its global branch has compressed linear-size attention, while the current dense local branch remains the main piece that must be optimized before the complete implementation achieves linear attention complexity.
