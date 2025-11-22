from typing import List
from needle.autograd import Tensor
import needle.backend_ndarray.ndarray as ndarray
from needle import ops
import needle.init as init
import numpy as np
from .nn_sequence import Embedding
from .nn_basic import (
    Parameter, 
    Module, 
    ReLU,
    Dropout,
    LayerNorm1d,
    Linear,
    Sequential
)


class MultiHeadAttention(Module):
    """
    The multi-head self attention module.
    """
    def __init__(
        self,
        *,
        dropout = 0.,
        causal = False,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        self.causal = causal
        self.dropout = Dropout(dropout)

    def create_causal_mask(self, i, j, device):
        """
        return a triangular causal mask.
        Input: i, j: the shape of the mask to be created
        """
        mask = -np.finfo(np.float32).max * np.triu(
            np.ones((1, 1, i, j), dtype=np.float32), j - i + 1)

        return ndarray.array(
            mask, device=device)

    def matmul(self, a, b_transpose):
        """
        batched matrix multiplication;
        """
        # a: (B, H, Tq, D); b_transpose: (B, H, D, Tk)
        B, H, Tq, D = a.shape
        _, _, D2, Tk = b_transpose.shape
        assert D == D2

        A3 = a.reshape((B * H, Tq, D))
        B3 = b_transpose.reshape((B * H, D, Tk))

        As = [t for t in ops.split(A3, axis=0)]
        Bs = [t for t in ops.split(B3, axis=0)]
        outs = [ops.matmul(x, y) for x, y in zip(As, Bs)]
        C = ops.stack(outs, axis=0)
        return C.reshape((B, H, Tq, Tk))

    def softmax(self, logit):
        """
        The softmax function; 
        """
        max_val = Tensor(
            logit.realize_cached_data().max(axis=3),
            device=logit.device,
            dtype=logit.dtype,
            requires_grad=False
        )

        max_val = max_val.reshape((*logit.shape[:-1], 1))
        max_val = max_val.broadcast_to(logit.shape)

        probs = ops.exp(logit - max_val)

        denom = probs.sum(axes=3)
        denom = denom.reshape((*logit.shape[:-1], 1))
        denom = denom.broadcast_to(logit.shape)

        return probs / denom

    def forward(
        self,
        q, k, v,
    ):
        """
        The forward function of the MultiHeadAttention activation function.
        Input: three states q, k, v, with shape (batch_size, num_head, seq_len, dim_head)
        Output: the activation output `result` and attention softmax probability `probs` (with dropout applied)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, keys_values_len, k_dim = k.shape
        _, _, _, v_dim = v.shape

        assert q_dim == k_dim == v_dim

        result = None
        probs = None

        ### BEGIN YOUR SOLUTION
        # Attention scores: (B, H, Tq, Tk)
        # Use a Python float to trigger scalar multiply (no broadcasting for 0-d tensors)
        inv_scale = float(1.0 / np.sqrt(np.float32(q_dim)))
        scores = self.matmul(q, ops.transpose(k)) * inv_scale

        # Causal mask if enabled
        if self.causal:
            mask_nd = self.create_causal_mask(queries_len, keys_values_len, q.device)
            mask = Tensor(mask_nd, device=q.device, dtype=q.dtype, requires_grad=False)
            mask = ops.broadcast_to(mask, scores.shape)
            scores = scores + mask

        # Softmax over last dim and apply dropout
        probs = self.softmax(scores)
        probs = self.dropout(probs)

        # Weighted sum of values: (B, H, Tq, D)
        result = self.matmul(probs, v)
        ### END YOUR SOLUTION

        return result, probs


class AttentionLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        *,
        k_features: int = None,
        v_features: int = None,
        out_features: int = None,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        if k_features is None:
            k_features = q_features
        if v_features is None:
            v_features = q_features
        if out_features is None:
            out_features = q_features

        self.q_features = q_features
        self.k_features = k_features
        self.v_features = v_features
        self.out_features = out_features

        self.num_head = num_head
        self.dim_head = dim_head

        self.prenorm_q = LayerNorm1d(
            q_features, device=device, dtype=dtype)
        self.prenorm_k = LayerNorm1d(
            k_features, device=device, dtype=dtype)
        self.prenorm_v = LayerNorm1d(
            v_features, device=device, dtype=dtype)

        inner_dim = num_head * dim_head
        
        self.q_projection = Linear(
            q_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.k_projection = Linear(
            k_features, inner_dim, bias=False,
            device=device, dtype=dtype)
        self.v_projection = Linear(
            v_features, inner_dim, bias=False,
            device=device, dtype=dtype)

        self.attn = MultiHeadAttention(
            dropout=dropout, causal=causal,
            device=device, dtype=dtype)

        self.out_projection = Linear(
            inner_dim, out_features, bias=False,
            device=device, dtype=dtype)

    def forward(
        self,
        q, k=None, v=None,
    ):
        """
        The forward function of the self-attention layer.
        Input: `q` with shape (batch_size, q_len, q_dim)
               `k` (if not None) with shape (batch_size, kv_len, k_dim)
               `v` (if not None) with shape (batch_size, kv_len, v_dim)
        Output: the output `result` with shape (batch_size, kv_len, out_features)
        """

        if k is None:
            k = q
        if v is None:
            v = q

        batch_size, queries_len, q_dim = q.shape
        _, keys_values_len, k_dim = k.shape
        _, _, v_dim = v.shape

        result = None

        ### BEGIN YOUR SOLUTION
        # Prenorm and linear projections
        inner_dim = self.num_head * self.dim_head

        # Flatten (B, T, D) -> (B*T, D) for LayerNorm and Linear (2D ops)
        B, Tq, _ = q.shape
        _, Tk, _ = k.shape
        _, Tv, _ = v.shape

        q_flat = q.reshape((B * Tq, q_dim))
        k_flat = k.reshape((B * Tk, k_dim))
        v_flat = v.reshape((B * Tv, v_dim))

        qn = self.prenorm_q(q_flat)
        kn = self.prenorm_k(k_flat)
        vn = self.prenorm_v(v_flat)

        qp = self.q_projection(qn)  # (B*Tq, H*D)
        kp = self.k_projection(kn)  # (B*Tk, H*D)
        vp = self.v_projection(vn)  # (B*Tv, H*D)

        # Restore time dim and split heads: (B, T, H, D) -> (B, H, T, D)
        qp = qp.reshape((B, Tq, inner_dim)).reshape((B, Tq, self.num_head, self.dim_head))
        kp = kp.reshape((B, Tk, inner_dim)).reshape((B, Tk, self.num_head, self.dim_head))
        vp = vp.reshape((B, Tv, inner_dim)).reshape((B, Tv, self.num_head, self.dim_head))

        qp = ops.transpose(qp, axes=(1, 2))
        kp = ops.transpose(kp, axes=(1, 2))
        vp = ops.transpose(vp, axes=(1, 2))

        # Multi-head attention activation
        attn_out, probs = self.attn(qp, kp, vp)  # (B, H, Tq, D)
        self.probs = probs

        # Merge heads back: (B, Tq, H*D)
        attn_out = ops.transpose(attn_out, axes=(1, 2))  # (B, Tq, H, D)
        attn_out = attn_out.reshape((B * Tq, inner_dim))

        # Output projection to requested features, then reshape back to (B, Tq, out)
        result = self.out_projection(attn_out)
        result = result.reshape((B, Tq, self.out_features))
        ### END YOUR SOLUTION

        return result


class TransformerLayer(Module):

    def __init__(
        self,
        q_features: int,
        num_head: int,
        dim_head: int,
        hidden_size: int,
        *,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype

        ### BEGIN YOUR SOLUTION
        self.attn = AttentionLayer(
            q_features,
            num_head,
            dim_head,
            dropout=dropout,
            causal=causal,
            device=device,
            dtype=dtype,
        )

        # Feed-forward block (prenorm variant)
        self.ff_norm = LayerNorm1d(q_features, device=device, dtype=dtype)
        self.ff_linear1 = Linear(q_features, hidden_size, device=device, dtype=dtype)
        self.ff_linear2 = Linear(hidden_size, q_features, device=device, dtype=dtype)
        self.relu = ReLU()
        self.dropout = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        ### END YOUR SOLUTION

    def forward(
        self,
        x
    ):
        """
        The forward function of a Transformer Layer.
        Input: the hidden states from previous layers `x` with shape (batch_size, seq_len, x_dim)
        Ouput: the hidden states after the Transformer Layer `x` with shape (batch_size, seq_len, x_dim)
        """

        batch_size, seq_len, x_dim = x.shape

        ### BEGIN YOUR SOLUTION
        # Self-attention with residual and dropout
        attn_out = self.attn(x)  # (B, T, D)
        x = x + self.dropout(attn_out)

        # Feed-forward with prenorm, residual, and dropout
        B, T, D = x.shape
        y = x.reshape((B * T, D))
        y = self.ff_norm(y)
        y = self.ff_linear1(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.ff_linear2(y)
        y = self.dropout2(y)
        y = y.reshape((B, T, D))
        x = x + y
        ### END YOUR SOLUTION

        return x


class Transformer(Module):

    def __init__(
        self,
        embedding_size: int,
        hidden_size: int,
        num_layers: int, 
        *,
        num_head: int = 8,
        dim_head: int = 32,
        dropout = 0.,
        causal = True,
        device = None,
        dtype = "float32",
        batch_first = False,
        sequence_len = 2048
    ):

        super().__init__()

        self.device = device
        self.dtype = dtype
        self.batch_first = batch_first

        ### BEGIN YOUR SOLUTION
        self.sequence_len = sequence_len
        self.num_layers = num_layers
        self.embedding_size = embedding_size

        # Positional embedding (learned)
        self.pos_embedding = Embedding(sequence_len, embedding_size, device=device, dtype=dtype)

        # Transformer layers
        self.layers: List[TransformerLayer] = []
        for _ in range(num_layers):
            self.layers.append(
                TransformerLayer(
                    q_features=embedding_size,
                    num_head=num_head,
                    dim_head=dim_head,
                    hidden_size=hidden_size,
                    dropout=dropout,
                    causal=causal,
                    device=device,
                    dtype=dtype,
                )
            )
        ### END YOUR SOLUTION

    def forward(
        self,
        x, h=None
    ):

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        ### BEGIN YOUR SOLUTION
        # x: (B, T, D)
        B, T, D = x.shape

        # Build positional indices of shape (T, B)
        pos = np.tile(np.arange(T, dtype=np.int32).reshape(T, 1), (1, B))
        pos_t = Tensor(pos.astype("float32"), device=x.device, dtype="float32", requires_grad=False)
        pos_emb = self.pos_embedding(pos_t)  # (T, B, D)
        pos_emb_bt = ops.transpose(pos_emb, axes=(0, 1))  # (B, T, D)

        x = x + pos_emb_bt

        for layer in self.layers:
            x = layer(x)
        ### END YOUR SOLUTION

        if not self.batch_first:
            x = ops.transpose(x, axes=(0, 1))

        return x, init.zeros_like(x)
