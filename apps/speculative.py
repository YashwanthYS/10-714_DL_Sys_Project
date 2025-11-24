import time
from typing import List, Tuple
import argparse
import os

import numpy as np

import sys
sys.path.append("python")

import needle as ndl
import needle.nn as nn
from needle import ops
from needle.autograd import Tensor


def _best_device():
    """Prefer CUDA if available, else CPU."""
    try:
        cu = ndl.cuda()
        if hasattr(cu, "enabled") and cu.enabled():
            return cu
    except Exception:
        pass
    return ndl.cpu()


def _fmt_eta(start_time: float, step: int, total: int) -> str:
    now = time.perf_counter()
    elapsed = max(1e-6, now - start_time)
    rate = step / elapsed if elapsed > 0 else 0.0
    remain = (total - step) / rate if rate > 0 else 0.0
    return f"elapsed={elapsed:.1f}s, eta={remain:.1f}s"


def _to_float(t: Tensor) -> float:
    """Robustly convert scalar-like Tensor to Python float without deprecation warnings."""
    arr = t.numpy()
    try:
        return float(arr.reshape(-1)[0])
    except Exception:
        # Fallback: use mean if unexpectedly non-scalar
        import numpy as _np
        return float(_np.asarray(arr).mean())


def _to_tensor_ids(ids: List[int], device) -> Tensor:
    arr = np.array(ids, dtype=np.float32).reshape(1, -1)
    return Tensor(arr, device=device, dtype="float32", requires_grad=False)


def _argmax_last(logits_2d: np.ndarray) -> int:
    # logits_2d: (1, V) from slicing (B,1,V)
    return int(np.argmax(logits_2d, axis=1)[0])


class SpeculativeDecoder:
    def __init__(self, vocab_size=1000, max_seq_len=128, k=3, device=None, draft_model: nn.Module=None, verify_model: nn.Module=None,
                 draft_embed: int = 96, draft_layers: int = 1, verify_embed: int = 128, verify_layers: int = 2, num_heads: int = 4):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.k = k
        self.device = device if device is not None else _best_device()

        # Draft model (configurable)
        dD = draft_embed
        H_D = num_heads
        Dh_D = dD // H_D
        self.draft = draft_model or nn.GPTModel(
            vocab_size=vocab_size,
            embed_dim=dD,
            num_layers=draft_layers,
            num_head=H_D,
            dim_head=Dh_D,
            mlp_hidden=4 * dD,
            max_seq_len=max_seq_len,
            dropout=0.0,
            device=self.device,
            dtype="float32",
            batch_first=True,
        )

        # Verifier model (configurable)
        dV = verify_embed
        H_V = num_heads
        Dh_V = dV // H_V
        self.verify = verify_model or nn.GPTModel(
            vocab_size=vocab_size,
            embed_dim=dV,
            num_layers=verify_layers,
            num_head=H_V,
            dim_head=Dh_V,
            mlp_hidden=4 * dV,
            max_seq_len=max_seq_len,
            dropout=0.0,
            device=self.device,
            dtype="float32",
            batch_first=True,
        )

        self.cache_d = self.draft.init_cache(batch_size=1)
        self.cache_v = self.verify.init_cache(batch_size=1)
        self._v_next_pred: Optional[int] = None

    def reset_caches(self):
        self.cache_d = self.draft.init_cache(batch_size=1)
        self.cache_v = self.verify.init_cache(batch_size=1)

    def warmup(self, prompt_ids: List[int]):
        # Advance both caches on the prompt in one block
        inp = _to_tensor_ids(prompt_ids, self.device)
        logits_d, self.cache_d = self.draft(inp, self.cache_d)
        logits_v, self.cache_v = self.verify(inp, self.cache_v)
        # store verifier's next-token prediction after the prompt (last position)
        self._v_next_pred = _argmax_last(logits_v.numpy()[:, -1, :])

    def draft_step(self, cur_id: int, max_steps: int) -> Tuple[List[int], List[Tensor]]:
        tokens: List[int] = []
        for _ in range(max_steps):
            inp = _to_tensor_ids([cur_id], self.device)
            logits, self.cache_d = self.draft(inp, self.cache_d)
            # logits shape (1,1,V)
            nxt = _argmax_last(logits.numpy()[:, -1, :])
            tokens.append(nxt)
            cur_id = nxt
        return tokens, []

    def verify_block(self, block: List[int]) -> Tuple[int, int, List[int]]:
        """
        Strict verifier: step token-by-token with KV cache to avoid alignment issues.
        Returns (match_len, mismatch_token, preds_seen) where preds_seen are the verifier's
        greedy predictions at each step.
        """
        preds_seen: List[int] = []
        m = 0
        mismatch_tok = -1
        for t in block:
            inp = _to_tensor_ids([t], self.device)
            logits, new_cache_v = self.verify(inp, self.cache_v)
            pred = _argmax_last(logits.numpy()[:, -1, :])
            preds_seen.append(pred)
            # advance cache regardless; verifier has now consumed this token
            self.cache_v = new_cache_v
            if pred == t:
                m += 1
            else:
                mismatch_tok = pred
                break
        # store next-token prediction for the next block if fully matched
        if m == len(block):
            self._v_next_pred = pred  # last pred equals next-token after last consumed
        return m, mismatch_tok, preds_seen

    def truncate_cache(self, cache, new_len):
        # Helper to truncate list-based caches to target length
        for lv in cache:
            if lv["k"]:
                lv["k"] = lv["k"][:new_len]
            if lv["v"]:
                lv["v"] = lv["v"][:new_len]

    def decode(self, prompt_ids: List[int], gen_tokens: int = 64) -> Tuple[List[int], dict]:
        # ensure deterministic inference
        self.draft.eval()
        self.verify.eval()
        self.reset_caches()
        self._v_next_pred = None
        self.warmup(prompt_ids)
        out = list(prompt_ids)
        accepted = 0
        verify_latencies = []
        while len(out) < len(prompt_ids) + gen_tokens and len(out) < self.max_seq_len:
            # Draft up to k tokens from D
            cur = out[-1]
            prev_len = len(self.cache_d[0]["k"]) if self.cache_d and self.cache_d[0]["k"] else 0
            block, _ = self.draft_step(cur, min(self.k, self.max_seq_len - len(out)))

            # Verify once in V
            t0 = time.perf_counter()
            match_len, mismatch_tok, _ = self.verify_block(block)
            verify_latencies.append((time.perf_counter() - t0) * 1000.0)

            if match_len == len(block):
                # Accept all
                out.extend(block)
                accepted += match_len
            else:
                # Accept prefix, then take V's token at mismatch
                accepted += match_len
                # Truncate caches back to prev_len + match_len
                self.truncate_cache(self.cache_v, prev_len + match_len)
                self.truncate_cache(self.cache_d, prev_len)

                # Advance D on accepted prefix (feed each accepted token)
                for t in block[:match_len]:
                    inp = _to_tensor_ids([t], self.device)
                    _, self.cache_d = self.draft(inp, self.cache_d)
                    out.append(t)

                # Step V on mismatch token to commit it
                if mismatch_tok >= 0 and len(out) < self.max_seq_len:
                    inp_v = _to_tensor_ids([mismatch_tok], self.device)
                    _, self.cache_v = self.verify(inp_v, self.cache_v)

                    # Also advance D on the mismatch token
                    inp_d = _to_tensor_ids([mismatch_tok], self.device)
                    _, self.cache_d = self.draft(inp_d, self.cache_d)

                    out.append(mismatch_tok)

        total_new = len(out) - len(prompt_ids)
        tokens_per_sec = total_new / max(1e-6, sum(verify_latencies) / 1000.0) if verify_latencies else 0.0
        p50 = float(np.percentile(np.array(verify_latencies), 50)) if verify_latencies else 0.0
        metrics = {
            "generated": total_new,
            "accepted": accepted,
            "acceptance_rate": (accepted / total_new) if total_new > 0 else 0.0,
            "verify_latency_p50_ms": p50,
            "tokens_per_sec_over_verify_time": tokens_per_sec,
        }
        return out, metrics


class CopyTaskTrainer:
    """Tiny trainer for a synthetic copy-language-modeling task.
    Objective: predict next token equal to current token.
    """
    def __init__(self, device=None):
        self.device = device if device is not None else _best_device()

    def _batch(self, vocab_size: int, seq_len: int, batch_size: int) -> Tuple[Tensor, Tensor]:
        X = np.random.randint(0, vocab_size, size=(batch_size, seq_len), dtype=np.int32)
        # Inputs are first T-1, targets are next tokens
        inp = X[:, :-1]
        tgt = X[:, 1:]
        inp_t = Tensor(inp.astype(np.float32), device=self.device, dtype="float32", requires_grad=False)
        tgt_t = Tensor(tgt.astype(np.float32), device=self.device, dtype="float32", requires_grad=False)
        return inp_t, tgt_t

    def train_model(self, model: nn.Module, vocab_size: int, seq_len: int = 32, batch_size: int = 16, steps: int = 200, lr: float = 1e-3, name: str = "train"):
        optim = ndl.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        loss_fn = nn.SoftmaxLoss()
        model.train()
        start = time.perf_counter()
        last_bucket = -1
        for it in range(steps):
            inp, tgt = self._batch(vocab_size, seq_len, batch_size)
            logits, _ = model(inp)  # (B, T-1, V)
            B, Tm1, V = logits.shape
            # Flatten and compute loss
            logits2d = logits.reshape((B * Tm1, V))
            tgt_flat = tgt.reshape((B * Tm1,))
            loss = loss_fn(logits2d, tgt_flat)
            optim.reset_grad()
            loss.backward()
            optim.step()
            # progress log every ~5% or at end
            bucket = int((it + 1) * 20 / steps)
            if bucket != last_bucket or it + 1 == steps:
                last_bucket = bucket
                print(f"[{name}] step {it+1}/{steps}, loss={_to_float(loss):.4f}, {_fmt_eta(start, it+1, steps)}")

    def train_both(self, decoder: 'SpeculativeDecoder', vocab_size: int, seq_len: int = 32, batch_size: int = 16, steps: int = 200, lr: float = 1e-3):
        # Train draft and verify on same task for better agreement
        self.train_model(decoder.draft, vocab_size, seq_len, batch_size, steps, lr, name="draft")
        self.train_model(decoder.verify, vocab_size, seq_len, batch_size, steps, lr, name="verify")


def profile_k(decoder: SpeculativeDecoder, prompt_ids: List[int], gen_tokens: int, ks: List[int]) -> List[dict]:
    rows = []
    for kk in ks:
        # reuse trained models but vary k
        d = SpeculativeDecoder(vocab_size=decoder.vocab_size, max_seq_len=decoder.max_seq_len, k=kk, device=decoder.device, draft_model=decoder.draft, verify_model=decoder.verify)
        out, metrics = d.decode(prompt_ids, gen_tokens=gen_tokens)
        rows.append({
            "k": kk,
            "toks_per_s": metrics["tokens_per_sec_over_verify_time"],
            "accept_rate": metrics["acceptance_rate"],
            "p50_ms": metrics["verify_latency_p50_ms"],
        })
    return rows


class TinyCharTokenizer:
    """A trivial tokenizer mapping characters to ids with modulo vocab size.
    Not suitable for real text, but fine for a toy demo.
    """
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size

    def encode(self, text: str) -> List[int]:
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, ids: List[int]) -> str:
        # Map ids back to printable ascii range for display
        chars = [chr((i % 95) + 32) for i in ids]
        return "".join(chars)


def run_copy_demo(args):
    np.random.seed(args.seed)
    device = _best_device()
    vocab = args.vocab_size
    max_len = args.max_seq_len
    decoder = SpeculativeDecoder(vocab_size=vocab, max_seq_len=max_len, k=args.k, device=device,
                                 draft_embed=args.draft_embed, draft_layers=args.draft_layers,
                                 verify_embed=args.verify_embed, verify_layers=args.verify_layers,
                                 num_heads=args.num_heads)
    print(f"Using device: {device}")

    # Optional: quick copy-task training to improve acceptance
    trainer = CopyTaskTrainer(device=device)
    trainer.train_both(decoder, vocab_size=vocab, seq_len=args.seq_len, batch_size=args.batch_size, steps=args.steps, lr=args.lr)

    tok = TinyCharTokenizer(vocab_size=vocab)
    prompt_text = args.prompt if args.prompt else "hello speculative decoding"
    prompt = tok.encode(prompt_text)
    out, metrics = decoder.decode(prompt, gen_tokens=args.gen_tokens)
    print("Prompt text:", prompt_text)
    print("Prompt ids:", prompt)
    print("Output (ids):", out)
    print("Output text:", tok.decode(out))
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.profile_k:
        ks = args.ks if args.ks else [2,3,4]
        print(f"\nProfile across k in {ks}:")
        table = profile_k(decoder, prompt, gen_tokens=args.gen_tokens, ks=ks)
        for row in table:
            print(f"  k={row['k']}: tokens/sec={row['toks_per_s']:.2f}, acceptance={row['accept_rate']:.2f}, p50_ms={row['p50_ms']:.1f}")




class SimpleWordTokenizer:
    def __init__(self, text: str, lowercase: bool = True, vocab_cap: int = 5000):
        if lowercase:
            text = text.lower()
        toks = text.split()
        # count
        freq = {}
        for t in toks:
            freq[t] = freq.get(t, 0) + 1
        specials = ["<unk>"]
        words = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
        vocab = specials + [w for w, _ in words]
        if vocab_cap is not None:
            vocab = vocab[:vocab_cap]
        self.itos = vocab
        self.stoi = {w: i for i, w in enumerate(vocab)}

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text: str) -> List[int]:
        toks = text.lower().split()
        unk = self.stoi.get("<unk>")
        return [self.stoi.get(t, unk) for t in toks]

    def decode(self, ids: List[int]) -> str:
        return " ".join(self.itos[i % len(self.itos)] for i in ids)


class WordLMTrainer:
    def __init__(self, device=None):
        self.device = device if device is not None else _best_device()

    def train_model(self, model: nn.Module, data_ids: np.ndarray, vocab_size: int, seq_len: int = 16, batch_size: int = 8, steps: int = 500, lr: float = 1e-3, name: str = "train"):
        optim = ndl.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)
        loss_fn = nn.SoftmaxLoss()
        N = data_ids.shape[0]
        model.train()
        start = time.perf_counter()
        last_bucket = -1
        for it in range(steps):
            if N <= seq_len:
                start_idx = [0] * batch_size
            else:
                start_idx = np.random.randint(0, N - seq_len - 1, size=(batch_size,))
            batch_x = np.stack([data_ids[s: s + seq_len] for s in start_idx], axis=0)
            inp = batch_x[:, :-1]
            tgt = batch_x[:, 1:]
            inp_t = Tensor(inp.astype(np.float32), device=self.device, dtype="float32", requires_grad=False)
            tgt_t = Tensor(tgt.astype(np.float32), device=self.device, dtype="float32", requires_grad=False)
            logits, _ = model(inp_t)
            B, Tm1, V = logits.shape
            loss = loss_fn(logits.reshape((B * Tm1, V)), tgt_t.reshape((B * Tm1,)))
            optim.reset_grad()
            loss.backward()
            optim.step()
            bucket = int((it + 1) * 20 / steps)
            if bucket != last_bucket or it + 1 == steps:
                last_bucket = bucket
                print(f"[{name}] step {it+1}/{steps}, loss={_to_float(loss):.4f}, {_fmt_eta(start, it+1, steps)}")

    def align_argmax(self, decoder: 'SpeculativeDecoder', data_ids: np.ndarray, vocab_size: int, seq_len: int = 16, batch_size: int = 8, steps: int = 200, lr: float = 1e-3):
        optim = ndl.optim.Adam(decoder.draft.parameters(), lr=lr, weight_decay=0.0)
        ce = nn.SoftmaxLoss()
        N = data_ids.shape[0]
        start = time.perf_counter()
        last_bucket = -1
        for it in range(steps):
            if N <= seq_len:
                start_idx = [0] * batch_size
            else:
                start_idx = np.random.randint(0, N - seq_len - 1, size=(batch_size,))
            batch_x = np.stack([data_ids[s: s + seq_len] for s in start_idx], axis=0)
            inp = batch_x[:, :-1]
            tgt = batch_x[:, 1:]
            inp_t = Tensor(inp.astype(np.float32), device=self.device, dtype="float32", requires_grad=False)
            tgt_t = Tensor(tgt.astype(np.float32), device=self.device, dtype="float32", requires_grad=False)
            with np.errstate(all='ignore'):
                lv, _ = decoder.verify(inp_t)
            teacher = np.argmax(lv.numpy(), axis=2).astype(np.float32)
            teacher_t = Tensor(teacher, device=self.device, dtype="float32", requires_grad=False)
            ld, _ = decoder.draft(inp_t)
            B, Tm1, V = ld.shape
            loss = ce(ld.reshape((B * Tm1, V)), teacher_t.reshape((B * Tm1,)))
            optim.reset_grad()
            loss.backward()
            optim.step()
            bucket = int((it + 1) * 20 / steps)
            if bucket != last_bucket or it + 1 == steps:
                last_bucket = bucket
                print(f"[align] step {it+1}/{steps}, loss={_to_float(loss):.4f}, {_fmt_eta(start, it+1, steps)}")

    def align_kd(self, decoder: 'SpeculativeDecoder', data_ids: np.ndarray, vocab_size: int, seq_len: int = 16, batch_size: int = 8, steps: int = 200, lr: float = 1e-3, alpha: float = 0.3, temperature: float = 2.0):
        """Knowledge Distillation: match draft to verify using soft targets.
        Loss = alpha * CE(draft, true) + (1-alpha) * KL(teacher || student) at temperature.
        """
        optim = ndl.optim.Adam(decoder.draft.parameters(), lr=lr, weight_decay=0.0)
        ce = nn.SoftmaxLoss()
        N = data_ids.shape[0]
        start = time.perf_counter()
        last_bucket = -1
        for it in range(steps):
            if N <= seq_len:
                start_idx = [0] * batch_size
            else:
                start_idx = np.random.randint(0, N - seq_len - 1, size=(batch_size,))
            batch_x = np.stack([data_ids[s: s + seq_len] for s in start_idx], axis=0)
            inp = batch_x[:, :-1]
            tgt = batch_x[:, 1:]
            inp_t = Tensor(inp.astype(np.float32), device=self.device, dtype="float32", requires_grad=False)
            tgt_t = Tensor(tgt.astype(np.float32), device=self.device, dtype="float32", requires_grad=False)

            # Teacher logits (no grad) -> soft probs
            with np.errstate(all='ignore'):
                logits_v, _ = decoder.verify(inp_t)
            Bv, Tv, V = logits_v.shape
            lv2 = logits_v.reshape((Bv * Tv, V)) / float(temperature)
            logsum_t = ops.logsumexp(lv2, axes=(1,)).reshape((Bv * Tv, 1))
            teacher_log_probs = lv2 - ops.broadcast_to(logsum_t, lv2.shape)
            teacher_probs = ops.exp(teacher_log_probs).detach()

            # Student logits
            logits_d, _ = decoder.draft(inp_t)
            B, Tm1, V = logits_d.shape
            ld2 = logits_d.reshape((B * Tm1, V)) / float(temperature)
            logsum_s = ops.logsumexp(ld2, axes=(1,)).reshape((B * Tm1, 1))
            student_log_probs = ld2 - ops.broadcast_to(logsum_s, ld2.shape)

            # Hard CE and soft KL
            loss_true = ce(logits_d.reshape((B * Tm1, V)), tgt_t.reshape((B * Tm1,)))
            # KL(teacher||student) = sum p_t * (log p_t - log p_s)
            kl = ops.summation(teacher_probs * (teacher_log_probs - student_log_probs)) / (B * Tm1)
            loss = loss_true * alpha + kl * (1.0 - alpha) * (temperature * temperature)

            optim.reset_grad()
            loss.backward()
            optim.step()

            bucket = int((it + 1) * 20 / steps)
            if bucket != last_bucket or it + 1 == steps:
                last_bucket = bucket
                print(f"[align-kd] step {it+1}/{steps}, loss={_to_float(loss):.4f}, {_fmt_eta(start, it+1, steps)}")


def _init_draft_from_verify(decoder: 'SpeculativeDecoder') -> bool:
    """Copy overlapping weights from verify into draft to jump-start agreement.
    Only works when embedding dims and head dims match; copies first N draft layers.
    Returns True if copied, else False.
    """
    d = decoder.draft
    v = decoder.verify
    ok = True
    try:
        # token and positional embeddings must match dims
        if d.embed_dim != v.embed_dim:
            return False
        # copy embeddings
        d.token_embedding.weight.data = v.token_embedding.weight.data
        d.pos_embedding.weight.data = v.pos_embedding.weight.data
        # copy first len(d.layers) transformer blocks
        for dl, vl in zip(d.layers, v.layers):
            dl.ln1.weight.data = vl.ln1.weight.data
            dl.ln1.bias.data = vl.ln1.bias.data
            dl.attn.q_proj.weight.data = vl.attn.q_proj.weight.data
            dl.attn.k_proj.weight.data = vl.attn.k_proj.weight.data
            dl.attn.v_proj.weight.data = vl.attn.v_proj.weight.data
            dl.attn.out_proj.weight.data = vl.attn.out_proj.weight.data
            dl.ln2.weight.data = vl.ln2.weight.data
            dl.ln2.bias.data = vl.ln2.bias.data
            dl.ff1.weight.data = vl.ff1.weight.data
            dl.ff2.weight.data = vl.ff2.weight.data
        d.ln_f.weight.data = v.ln_f.weight.data
        d.ln_f.bias.data = v.ln_f.bias.data
        return ok
    except Exception:
        return False


def run_toy_demo(args):
    np.random.seed(args.seed)
    device = _best_device()
    # Load or build tiny corpus (prefer shakespeare_2k if present, else create from shakespeare.txt)
    if args.toy_corpus:
        path = args.toy_corpus
    else:
        path = os.path.join("data", "shakespeare_2k.txt")
        if not os.path.exists(path):
            full = os.path.join("data", "shakespeare.txt")
            if os.path.exists(full):
                try:
                    with open(full, "r", encoding="utf-8") as fin, open(path, "w", encoding="utf-8") as fout:
                        for i, line in enumerate(fin):
                            if i >= 2000:
                                break
                            fout.write(line)
                    print(f"Created subset corpus: {path}")
                except Exception:
                    pass
        if not os.path.exists(path):
            # fallback tiny toy text
            path = os.path.join("data", "toy_corpus.txt")
            if not os.path.exists(path):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(("the quick brown fox jumps over the lazy dog\n") * 200)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    tok = SimpleWordTokenizer(text, lowercase=True, vocab_cap=args.vocab_cap)
    data_ids = np.array(tok.encode(text), dtype=np.int32)
    vocab = tok.vocab_size
    max_len = args.max_seq_len
    decoder = SpeculativeDecoder(vocab_size=vocab, max_seq_len=max_len, k=args.k, device=device,
                                 draft_embed=args.draft_embed, draft_layers=args.draft_layers,
                                 verify_embed=args.verify_embed, verify_layers=args.verify_layers,
                                 num_heads=args.num_heads)
    print(f"Using device: {device}")

    # Default training order for better acceptance:
    # 1) Train VERIFY first
    # 2) Initialize DRAFT from VERIFY (same embed dim recommended)
    # 3) Align DRAFT to VERIFY (KD or argmax)
    trainer = WordLMTrainer(device=device)
    if args.steps > 0:
        trainer.train_model(decoder.verify, data_ids, vocab, seq_len=args.seq_len, batch_size=args.batch_size, steps=args.steps, lr=args.lr, name="verify")
    copied = _init_draft_from_verify(decoder)
    print("Initialized draft from verify:", copied)
    if args.align_steps > 0:
        if args.align_mode == "kd":
            trainer.align_kd(decoder, data_ids, vocab, seq_len=args.seq_len, batch_size=args.batch_size, steps=args.align_steps, lr=args.lr, alpha=args.align_alpha, temperature=args.align_temp)
        else:
            trainer.align_argmax(decoder, data_ids, vocab, seq_len=args.seq_len, batch_size=args.batch_size, steps=args.align_steps, lr=args.lr)

    # Prompt and decode
    prompt_text = args.prompt if args.prompt else "the quick brown fox"
    prompt_ids = tok.encode(prompt_text)
    out, metrics = decoder.decode(prompt_ids, gen_tokens=args.gen_tokens)
    print("Prompt text:", prompt_text)
    print("Prompt ids:", prompt_ids)
    print("Output (ids):", out)
    print("Output text:", tok.decode(out))
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if args.profile_k:
        ks = args.ks if args.ks else [2,3,4]
        print(f"\nProfile across k in {ks}:")
        table = profile_k(decoder, prompt_ids, gen_tokens=args.gen_tokens, ks=ks)
        for row in table:
            print(f"  k={row['k']}: tokens/sec={row['toks_per_s']:.2f}, acceptance={row['accept_rate']:.2f}, p50_ms={row['p50_ms']:.1f}")


def main_demo():
    parser = argparse.ArgumentParser(description="Speculative decoding demo")
    parser.add_argument("--mode", choices=["copy", "toy"], default="copy")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gen-tokens", type=int, default=64)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--profile-k", action="store_true")
    parser.add_argument("--ks", type=int, nargs="*", default=None)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draft-embed", type=int, default=256)
    parser.add_argument("--verify-embed", type=int, default=256)
    parser.add_argument("--draft-layers", type=int, default=1)
    parser.add_argument("--verify-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    # toy mode extras
    parser.add_argument("--toy-corpus", type=str, default="", help="path to simple corpus text (optional)")
    parser.add_argument("--vocab-cap", type=int, default=5000)
    parser.add_argument("--align-steps", type=int, default=200)
    parser.add_argument("--align-mode", choices=["kd","argmax"], default="kd")
    parser.add_argument("--align-alpha", type=float, default=0.3)
    parser.add_argument("--align-temp", type=float, default=2.0)
    parser.add_argument("--init-draft-from-verify", action="store_true")
    args = parser.parse_args()

    if args.mode == "copy":
        run_copy_demo(args)
    else:
        run_toy_demo(args)


if __name__ == "__main__":
    main_demo()
