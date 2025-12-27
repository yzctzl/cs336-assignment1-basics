import math

import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        """
        Construct a linear transformation module.

        Args:
            in_features: int final dimension of the input
            out_features: int final dimension of the output
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        # call the superclass constructor
        super().__init__()
        # linear transformation = xW^T, for row-major W ∈ R^d_out×d_in and row-vector x ∈ R^1×din
        _w = torch.empty((out_features, in_features), device=device, dtype=dtype)
        # parameter initialization: N(µ = 0, σ2 = 2/(d_in+d_out)), truncated at [−3σ, 3σ]
        w_std = math.sqrt(2 / (in_features + out_features))
        # construct and store your parameter as W (not W^⊤) for memory ordering reasons
        w = nn.init.trunc_normal_(_w, mean=0, std=w_std, a=-3 * w_std, b=3 * w_std)
        # putting it in an nn.Parameter
        self.weight = nn.Parameter(w)

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        """
        Apply the linear transformation to the input.
        """
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """
        Construct an embedding module.

        Args:
            num_embeddings: int Size of the vocabulary
            embedding_dim: int Dimension of the embedding vectors, i.e., dmodel
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        # call the superclass constructor
        super().__init__()
        # parameter initialization: N(µ = 0, σ2 = 1) truncated at [−3, 3]
        _e = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        e = nn.init.trunc_normal_(_e, mean=0, std=1, a=-3, b=3)
        # initialize embedding matrix as a nn.Parameter, now it's learnable parameter
        self.weight = nn.Parameter(e)

    def forward(self, token_ids: Int[Tensor, " ..."]) -> Float[Tensor, " ... d_model"]:
        """
        Lookup the embedding vectors for the given token IDs.
        """
        # bypasses MatMul via hardware-accelerated Gather for $O(1)$ dense feature lookup
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    def __init__(
        self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        """
        Construct the RMSNorm module. This function should accept the following parameters:

        Args:
            d_model: int Hidden dimension of the model
            eps: float = 1e-5 Epsilon value for numerical stability
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model = d_model
        # ε is a hyperparameter that is often fixed at 1e-5
        self.eps = eps
        # g(weight) is a learnable "gain" parameter
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Float[Tensor, " ... d_model"]) -> Float[Tensor, " ... d_model"]:
        """
        Process an input tensor of shape (batch_size, sequence_length, d_model)
        and return a tensor of the same shape.

        RMSNorm(a_i) = a_i / RMS(a) * g_i, RMS(a) = sqrt(sum(a_i^2) / d_model + eps)
        """
        # upcast input to torch.float32 to prevent overflow when you square the input
        in_dtype = x.dtype
        x = x.to(torch.float32)

        # mean is more better than sum/d_model
        ms = x.pow(2).mean(dim=-1, keepdim=True)
        # rsqrt is more efficent than /, rsqrt <=> RSQRT in CUDA
        rms = torch.rsqrt(ms + self.eps)
        # use * instand of /, is more cheaper
        result = x * rms * self.weight

        return result.to(in_dtype)


def SiLU(x: Float[Tensor, " ..."]) -> Float[Tensor, " ..."]:
    """
    SiLU(x) = x · sigmoid(x)
    """
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(
        self, d_model: int, d_ff: int, device: torch.device | None = None, dtype: torch.dtype | None = None
    ) -> None:
        """
        Construct a SwiGLU activation module.

        This implementation leverages modular linear layers to encapsulate gated activation
        logic, enhancing code structure and PyTorch autograd compatibility.

        Note: This implementation does not use merged matrices or torch.nn.functional
        for simplicity, although they would be more efficient, only one memory access.

        Args:
            d_model: int Input and output dimension
            d_ff: int Intermediate hidden dimension
            device: torch.device | None = None Device to store the parameters on
            dtype: torch.dtype | None = None Data type of the parameters

        """
        super().__init__()
        self.gate = Linear(d_model, 2 * d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        """
        SiLU(x) = x · sigmoid(x),
        GLU(x, W1, W2) = sigmoid(W1x) ⊙ W2x,
        FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x),
        where x ∈ R^d_model, W1, W3 ∈ R^d_ff×d_model, W2 ∈ R^d_model×d_ff
        and canonically, d_ff = 8/3 d_model
        """
        _gate: Float[Tensor, "... 2_d_ff"] = self.gate(x)
        _w1, _w3 = _gate.chunk(2, dim=-1)
        return self.w2(SiLU(_w1) * _w3)


class RotaryPositionalEmbedding(nn.Module):
    freqs_complex: Tensor

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None) -> None:
        """
        Construct the RoPE module and create buffers if needed.

        Args:
            theta: float Θ value for the RoPE
            d_k: int dimension of query and key vectors
            max_seq_len: int Maximum sequence length that will be inputted
            device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        # [0, ..., d/2]
        dim = torch.arange(0, d_k, 2.0, dtype=torch.float32, device=device)
        # the angle theta
        freq_theta = 1.0 / (theta ** (dim / d_k))
        # [0, ..., max_seq_len]
        position = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        # position * theta
        freqs = torch.outer(position, freq_theta)

        # cache complex
        emb_complex = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_complex", emb_complex, persistent=False)

    def forward(
        self, x: Float[Tensor, "... seq_len d_k"], token_positions: Int[Tensor, " ... seq_len"]
    ) -> Float[Tensor, " ... seq_len d_k"]:
        """
        Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.

        Note that you should tolerate x with an arbitrary number of batch dimensions. You should
        assume that the token positions are a tensor of shape (..., seq_len) specifying the token
        positions of x along the sequence dimension.

        You should use the token positions to slice your (possibly precomputed) cos and sin tensors
        along the sequence dimension.
        """
        x_dtype = x.dtype
        seq_len = x.shape[-2]

        # get corresponding rotation factors
        if token_positions is None:
            freqs = self.freqs_complex[:seq_len]
        else:
            freqs = self.freqs_complex[token_positions]

        # reshape input x and convert to complex view
        # (..., d_k) -> (..., d_k/2, 2) -> (..., d_k/2) complex
        # Note: view_as_complex requires the last dimension to be 2, and data should typically be float32
        x_float = x.float().reshape(*x.shape[:-1], -1, 2)
        x_complex = torch.view_as_complex(x_float)

        # Complex multiplication performs the rotation: (x0 + ix1) * (cos + isin)
        # Based on broadcasting rules, freqs will automatically match the dimensions of x_complex
        x_out_complex = x_complex * freqs

        # (..., d_k/2) complex -> (..., d_k/2, 2) real -> (..., d_k)
        x_out = torch.view_as_real(x_out_complex).flatten(-2)

        return x_out.to(x_dtype)


def softmax(x: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    """
    Computes numerically stable Softmax along the specified dimension.

    Key Implementation Details:
    1. Numerical Stability: Subtracts the maximum value along 'dim' from x
       before exponentiation to prevent overflow (exp(x) -> inf).
    2. Dimensional Logic:
       - Uses `keepdim=True` to ensure the max/sum tensors remain broadcastable
         with the input shape.
       - The operation treats each "fiber" along the i-th dimension as an
         independent vector to be normalized.
    3. Mathematical Mapping: Maps real-valued logits to a probability
       distribution (0, 1) where only the target dimension sums to 1.0.
    """
    # for each fiber by dim subtract the max
    x = x - x.max(dim=dim, keepdim=True).values
    # exp apply all element in x
    x_exp = x.exp()
    # for each fiber do standardization
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    Q: Float[Tensor, "... seq_len d_k"],
    K: Float[Tensor, "... seq_len d_k"],
    V: Float[Tensor, "... seq_len d_k"],
    mask: Bool[Tensor, "seq_len seq_len"] | None = None,
) -> Float[Tensor, "... seq_len d_k"]:
    """
    The scaled dot-product attention function

    Given key (K), query (Q), and value (V) tensors, return
    the output of your scaled dot product attention implementation.

    Args:
        Q (Float[Tensor, " ... queries d_k"]): Query tensor
        K (Float[Tensor, " ... keys d_k"]): Key tensor
        V (Float[Tensor, " ... values d_v"]): Values tensor
        mask (Bool[Tensor, " ... queries keys"] | None): Mask tensor
    """
    d_k = Q.shape[-1]
    # attention score = QK^T / sqrt(d_k)
    scores = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys") / math.sqrt(d_k)
    # it will be much more efficient to use masking than to compute attention on subsequences
    if mask is not None:
        masked = torch.where(mask, scores, float("-inf"))
    else:
        masked = scores
    # Attention(Q, K, V) = softmax(scores)V
    return einsum(softmax(masked, -1), V, "... queries seq_len, ... seq_len d_k -> ... queries d_k")


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) layer with optional Rotary Positional Embeddings (RoPE).

    Mathematical Representation:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
        where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    This module implements the attention mechanism by:
    1. Projecting the input tensor into Query (Q), Key (K), and Value (V) spaces using linear layers.
    2. Splitting the projected d_model dimension into multiple heads of dimension d_k.
    3. Optionally applying Rotary Positional Embeddings (RoPE) to the Q and K tensors.
    4. Computing Scaled Dot-Product Attention independently for each head, incorporating a
       causal mask to prevent tokens from attending to future positions.
    5. Concatenating the head outputs and applying a final linear projection (O) to
       map the concatenated vectors back to the d_model space.

    combining the key, query, and value projections into a single weight matrix
    so you only need a single matrix multiply.
    """

    mask: Tensor  # causal mask

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # linearity: [Q K V] = [xW_Q^T xW_K^T x W_V^T] = x[W_Q^T W_K^T W_V^T]
        # Due to linearity, we can use a single linear layer to project into all heads
        # simultaneously instead of separate layers, which is more computationally efficient.
        self.qkv_proj = Linear(d_model, 3 * d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = None
        if theta is not None:
            # RoPE is applied to each head independently, so we use the dimension
            # of each head (d_k) instead of the total model dimension (d_model).
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)

        # causal mask: query is the row, and key is the col, mask is col <= row, so use tril
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(
        self,
        x: Float[Tensor, " ... seq_len d_in"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, " ... seq_len d_out"]:
        QKV: Float[Tensor, "... 3_d_model"] = self.qkv_proj(x)
        Q, K, V = QKV.chunk(3, dim=-1)

        # Split the head dimension and move it to the -3 dimension to allow
        # parallel attention computation across heads.
        q = rearrange(Q, "... seq_len (head d_k) -> ... head seq_len d_k", head=self.num_heads)
        k = rearrange(K, "... seq_len (head d_k) -> ... head seq_len d_k", head=self.num_heads)
        v = rearrange(V, "... seq_len (head d_k) -> ... head seq_len d_k", head=self.num_heads)

        if self.rope is not None:
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        seq_len = x.shape[-2]
        dot_att = scaled_dot_product_attention(q, k, v, mask=self.mask[:seq_len, :seq_len])
        re_att = rearrange(dot_att, "... head seq_len d_k -> ... seq_len (head d_k)")
        return self.output_proj(re_att)


class TransformerBlock(nn.Module):
    """
    y' = x + MultiHeadSelfAttention(RMSNorm(x))
    y = y' + SwiGLU(RMSNorm(y'))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, device, dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device, dtype)

    def forward(
        self,
        x: Float[Tensor, " batch seq_len d_model"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, " batch seq_len d_model"]:
        _norm_x = self.ln1(x)
        _attn_x = self.attn(_norm_x, token_positions).add(x)

        _norm_attn = self.ln2(_attn_x)
        return self.ffn(_norm_attn).add(_attn_x)


class TransformerLM(nn.Module):
    """
    A Decoder-only Transformer Language Model for autoregressive sequence modeling.

    Mathematical Representation:
        h_0 = Embedding(x)
        h_i = TransformerBlock_i(h_{i-1}), i = 1, ..., L
        logits = Linear(RMSNorm(h_L))

    This architecture implements a stack of transformer blocks, each featuring
    Multi-Head Self-Attention with Rotary Positional Embeddings (RoPE) and
    SwiGLU feed-forward networks. The model uses RMSNorm for pre-normalization
    and concludes with a linear language modeling head to project hidden states
    back to the vocabulary space for next-token prediction.
    """

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        **kwargs
    ) -> None:
        super().__init__()
        pad_vocab = kwargs.get('pad_vocab', False)
        tie_weight = kwargs.get('tie_weight', False)

        self.num_layers = num_layers
        self.vocab_size = vocab_size
        # pad num embeddings with 64
        if pad_vocab:
            vocab_size = ((vocab_size + 64 - 1) // 64) * 64
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
            )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

        # tie weights
        if tie_weight:
            self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self,
        x: Int[Tensor, " batch seq_len"],
        token_positions: Int[Tensor, " ... seq_len"] | None = None,
    ) -> Float[Tensor, " batch seq_len vocab_size"]:
        """
        Tensor with the predicted unnormalized next-word distribution for each token.
        """
        _emb = self.token_embeddings(x)

        _x = _emb
        for layer in self.layers:
            _x = layer(_x, token_positions)

        _x = self.ln_final(_x)
        return self.lm_head(_x)
