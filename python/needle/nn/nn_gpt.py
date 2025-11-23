from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from needle.autograd import Tensor
from needle import ops
import needle.init as init

from .nn_basic import (
    Module,
    Linear,
    LayerNorm1d,
    Dropout,
)
from .nn_sequence import Embedding


def _batched_matmul(a: Tensor, bT: Tensor) -> Tensor:
    """
    Batched matmul like in nn_transformer.MultiHeadAttention.
    a: (B, H, Tq, D), bT: (B, H, D, Tk) -> (B, H, Tq, Tk)
    """
    B, H, Tq, D = a.shape
    _, _, D2, Tk = bT.shape
    assert D == D2
    A3 = a.reshape((B * H, Tq, D))
    B3 = bT.reshape((B * H, D, Tk))
    As = [t for t in ops.split(A3, axis=0)]
    Bs = [t for t in ops.split(B3, axis=0)]
    outs = [ops.matmul(x, y) for x, y in zip(As, Bs)]
    C = ops.stack(outs, axis=0)
    return C.reshape((B, H, Tq, Tk))


def _softmax_last_dim(logits: Tensor) -> Tensor:
    """
    Numerically-stable softmax over last dimension.
    logits: (..., K)
    """
    max_val = Tensor(
        logits.realize_cached_data().max(axis=len(logits.shape) - 1),
        device=logits.device,
        dtype=logits.dtype,
        requires_grad=False,
    )
    shape_expanded = (*logits.shape[:-1], 1)
    max_val = max_val.reshape(shape_expanded).broadcast_to(logits.shape)
    probs = ops.exp(logits - max_val)
    denom = probs.sum(axes=len(probs.shape) - 1)
    denom = denom.reshape(shape_expanded).broadcast_to(logits.shape)
    return probs / denom


def _causal_mask_block(T_new: int, T_prev: int, Tk: int, device) -> Tensor:
    """
    Create a causal mask for a block of T_new queries attending over Tk keys,
    where the first T_prev keys are all visible, and within the new block
    queries cannot attend to future positions inside the block.
    Returns: mask tensor of shape (1, 1, T_new, Tk) with 0 for allowed and -inf for masked.
    """
    mask = np.zeros((1, 1, T_new, Tk), dtype=np.float32)
    if Tk > T_prev and T_new > 0:
        future = np.triu(np.ones((T_new, Tk - T_prev), dtype=np.float32), k=1)
        # place into the slice starting from T_prev
        mask[:, :, :, T_prev: ] = -np.finfo(np.float32).max * future.reshape(1, 1, T_new, Tk - T_prev)
    return Tensor(mask, device=device, dtype="float32", requires_grad=False)


class CausalSelfAttention(Module):
    def __init__(self, embed_dim: int, num_head: int, dim_head: int, dropout: float = 0.0, device=None, dtype="float32"):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_head = num_head
        self.dim_head = dim_head
        inner = num_head * dim_head
        self.q_proj = Linear(embed_dim, inner, bias=False, device=device, dtype=dtype)
        self.k_proj = Linear(embed_dim, inner, bias=False, device=device, dtype=dtype)
        self.v_proj = Linear(embed_dim, inner, bias=False, device=device, dtype=dtype)
        self.out_proj = Linear(inner, embed_dim, bias=False, device=device, dtype=dtype)
        self.dropout = Dropout(dropout)
        self.device = device
        self.dtype = dtype

    def forward(self, x: Tensor, cache: Optional[Dict[str, List[Tensor]]] = None) -> Tuple[Tensor, Dict[str, List[Tensor]]]:
        """
        x: (B, T_new, D)
        cache: dict with keys 'k', 'v' mapping to lists (length T_prev) of Tensors shaped (B, H, 1, Dh)
        Returns: (y, new_cache) where y is (B, T_new, D)
        """
        B, T_new, D = x.shape
        H, Dh = self.num_head, self.dim_head
        inner = H * Dh

        # Projections
        xf = x.reshape((B * T_new, D))
        q = self.q_proj(xf).reshape((B, T_new, inner)).reshape((B, T_new, H, Dh))
        k = self.k_proj(xf).reshape((B, T_new, inner)).reshape((B, T_new, H, Dh))
        v = self.v_proj(xf).reshape((B, T_new, inner)).reshape((B, T_new, H, Dh))

        # To (B, H, T, Dh)
        q = ops.transpose(q, axes=(1, 2))
        k = ops.transpose(k, axes=(1, 2))
        v = ops.transpose(v, axes=(1, 2))

        # Build K/V over context
        use_cache = cache is not None
        T_prev = 0
        if use_cache and cache.get("k"):
            T_prev = len(cache["k"])  # tokens already cached

        if use_cache:
            # Append current block as slices for incremental decoding
            k_list: List[Tensor] = []
            v_list: List[Tensor] = []
            if cache.get("k"):
                k_list.extend(cache["k"])  # elements (B,H,Dh)
                v_list.extend(cache["v"])  # elements (B,H,Dh)

            if T_new == 1:
                k_slices = [k.reshape((B, H, Dh))]
                v_slices = [v.reshape((B, H, Dh))]
            else:
                k_slices = [t for t in ops.split(k, axis=2)]
                v_slices = [t for t in ops.split(v, axis=2)]

            k_list.extend(k_slices)
            v_list.extend(v_slices)

            K_all = ops.stack(k_list, axis=2)
            V_all = ops.stack(v_list, axis=2)
        else:
            # Training/full-block case: avoid per-time slicing/lists
            K_all = k
            V_all = v

        # Compute attention for the new T_new queries only
        inv_scale = float(1.0 / np.sqrt(np.float32(Dh)))
        scores = _batched_matmul(q, ops.transpose(K_all)) * inv_scale

        # Causal mask for the block
        Tk = K_all.shape[2]
        mask = _causal_mask_block(T_new=T_new, T_prev=T_prev, Tk=Tk, device=x.device)
        mask = ops.broadcast_to(mask, scores.shape)
        scores = scores + mask

        probs = _softmax_last_dim(scores)
        probs = self.dropout(probs)

        # Weighted sum: (B,H,T_new,Tk) @ (B,H,Tk,Dh) -> (B,H,T_new,Dh)
        B, H, Tq, Tk = probs.shape
        _, _, Tk2, Dh = V_all.shape
        assert Tk == Tk2
        A3 = probs.reshape((B * H, Tq, Tk))
        B3 = V_all.reshape((B * H, Tk, Dh))
        As = [t for t in ops.split(A3, axis=0)]
        Bs = [t for t in ops.split(B3, axis=0)]
        outs = [ops.matmul(x, y) for x, y in zip(As, Bs)]
        out = ops.stack(outs, axis=0).reshape((B, H, Tq, Dh))
        out = ops.transpose(out, axes=(1, 2))  # (B,T_new,H,Dh)
        out = out.reshape((B * T_new, inner))
        out = self.out_proj(out)
        out = out.reshape((B, T_new, D))

        # Prepare updated cache
        new_cache = {"k": k_list, "v": v_list} if use_cache else {"k": [], "v": []}
        return out, new_cache


class GPTBlock(Module):
    def __init__(self, embed_dim: int, num_head: int, dim_head: int, mlp_hidden: int, dropout: float = 0.0, device=None, dtype="float32"):
        super().__init__()
        self.ln1 = LayerNorm1d(embed_dim, device=device, dtype=dtype)
        self.attn = CausalSelfAttention(embed_dim, num_head, dim_head, dropout=dropout, device=device, dtype=dtype)
        self.ln2 = LayerNorm1d(embed_dim, device=device, dtype=dtype)
        self.ff1 = Linear(embed_dim, mlp_hidden, device=device, dtype=dtype)
        self.ff2 = Linear(mlp_hidden, embed_dim, device=device, dtype=dtype)
        self.dropout = Dropout(dropout)

    def forward(self, x: Tensor, cache: Optional[Dict[str, List[Tensor]]] = None) -> Tuple[Tensor, Dict[str, List[Tensor]]]:
        B, T, D = x.shape
        y = x.reshape((B * T, D))
        y = self.ln1(y).reshape((B, T, D))
        attn_out, new_cache = self.attn(y, cache)
        x = x + self.dropout(attn_out)

        B, T, D = x.shape
        y2 = x.reshape((B * T, D))
        y2 = self.ln2(y2)
        y2 = self.ff1(y2)
        # Use ReLU as simple activation
        y2 = ops.relu(y2)
        y2 = self.dropout(y2)
        y2 = self.ff2(y2)
        y2 = self.dropout(y2).reshape((B, T, D))
        x = x + y2
        return x, new_cache


class GPTModel(Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_layers: int,
        num_head: int,
        dim_head: int,
        mlp_hidden: int,
        max_seq_len: int = 128,
        dropout: float = 0.0,
        device=None,
        dtype="float32",
        batch_first: bool = True,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.batch_first = batch_first

        self.token_embedding = Embedding(vocab_size, embed_dim, device=device, dtype=dtype)
        self.pos_embedding = Embedding(max_seq_len, embed_dim, device=device, dtype=dtype)
        self.dropout = Dropout(dropout)
        self.layers: List[GPTBlock] = []
        for _ in range(num_layers):
            self.layers.append(
                GPTBlock(embed_dim, num_head, dim_head, mlp_hidden, dropout=dropout, device=device, dtype=dtype)
            )
        self.ln_f = LayerNorm1d(embed_dim, device=device, dtype=dtype)
        self.device = device
        self.dtype = dtype

    def init_cache(self, batch_size: int) -> List[Dict[str, List[Tensor]]]:
        return [ {"k": [], "v": []} for _ in range(self.num_layers) ]

    def truncate_cache(self, cache: List[Dict[str, List[Tensor]]], new_len: int) -> None:
        for lv in cache:
            if lv["k"]:
                lv["k"] = lv["k"][:new_len]
            if lv["v"]:
                lv["v"] = lv["v"][:new_len]

    def forward(self, input_ids: Tensor, cache: Optional[List[Dict[str, List[Tensor]]]] = None) -> Tuple[Tensor, List[Dict[str, List[Tensor]]]]:
        """
        input_ids: (B, T) if batch_first else (T, B)
        Returns logits: (B, T, V) if batch_first else (T, B, V)
        Also returns updated cache.
        """
        x = input_ids
        if not self.batch_first:
            x = ops.transpose(x, axes=(1, 0))  # (B, T)
        B, T = x.shape

        # Build embeddings
        # Embedding expects (seq_len, bs)
        x_TB = ops.transpose(x, axes=(1, 0))  # (T, B)
        tok_TBD = self.token_embedding(x_TB)
        # Determine absolute position offset from cache length
        pos_offset = 0
        if cache is not None and len(cache) > 0 and cache[0].get("k"):
            pos_offset = len(cache[0]["k"])  # tokens already cached
        # Positional indices shape (T, B) with offset
        pos_idx = np.tile((np.arange(T, dtype=np.int32) + pos_offset).reshape(T, 1), (1, B))
        pos = Tensor(pos_idx.astype("float32"), device=tok_TBD.device, dtype="float32", requires_grad=False)
        pos_emb = self.pos_embedding(pos)
        pos_emb = ops.transpose(pos_emb, axes=(0, 1))  # (B, T, D)
        h = ops.transpose(tok_TBD, axes=(0, 1))  # (B, T, D)
        h = h + pos_emb
        h = self.dropout(h)

        # Prepare cache: avoid building caches during training to save memory
        if cache is None:
            if self.training:
                cache = [None] * self.num_layers  # signal no caching
            else:
                cache = self.init_cache(B)

        # Pass through blocks with caching
        new_cache: List[Dict[str, List[Tensor]]] = []
        out = h
        for i, layer in enumerate(self.layers):
            layer_cache_in = None if (isinstance(cache, list) and len(cache) > i and cache[i] is None) else cache[i]
            out, layer_cache = layer(out, layer_cache_in)
            new_cache.append(layer_cache)

        # Final layer norm and logits via tied embedding weight
        B, T, D = out.shape
        y = out.reshape((B * T, D))
        y = self.ln_f(y)
        # Tie weights: logits = y @ W^T
        W = self.token_embedding.weight  # (V, D)
        logits2d = ops.matmul(y, ops.transpose(W))  # (B*T, V)
        logits = logits2d.reshape((B, T, self.vocab_size))

        if not self.batch_first:
            logits = ops.transpose(logits, axes=(1, 0, 2))
        return logits, new_cache
