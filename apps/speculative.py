import time
from typing import List, Tuple
import argparse
import os

import numpy as np

import sys
sys.path.append("python")

import needle as ndl
import needle.nn as nn
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

    def reset_caches(self):
        self.cache_d = self.draft.init_cache(batch_size=1)
        self.cache_v = self.verify.init_cache(batch_size=1)

    def warmup(self, prompt_ids: List[int]):
        # Advance both caches on the prompt in one block
        inp = _to_tensor_ids(prompt_ids, self.device)
        _, self.cache_d = self.draft(inp, self.cache_d)
        _, self.cache_v = self.verify(inp, self.cache_v)

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
        Run verifier over the drafted block; returns (match_len, mismatch_token, v_preds)
        mismatch_token is the verifier's token at first mismatch index, or -1 if full match.
        v_preds is the verifier's greedy predictions for each position in block.
        """
        inp = _to_tensor_ids(block, self.device)
        logits, new_cache_v = self.verify(inp, self.cache_v)
        preds = np.argmax(logits.numpy(), axis=2).astype(int).reshape(-1).tolist()
        m = 0
        mismatch_tok = -1
        for i, (d, v) in enumerate(zip(block, preds)):
            if d == v:
                m += 1
            else:
                mismatch_tok = v
                break
        # tentatively set cache to the full block advance; caller may truncate
        self.cache_v = new_cache_v
        return m, mismatch_tok, preds

    def truncate_cache(self, cache, new_len):
        # Helper to truncate list-based caches to target length
        for lv in cache:
            if lv["k"]:
                lv["k"] = lv["k"][:new_len]
            if lv["v"]:
                lv["v"] = lv["v"][:new_len]

    def decode(self, prompt_ids: List[int], gen_tokens: int = 64) -> Tuple[List[int], dict]:
        self.reset_caches()
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




def main_demo():
    parser = argparse.ArgumentParser(description="Speculative decoding demo (copy mode only)")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gen-tokens", type=int, default=64)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--profile-k", action="store_true")
    parser.add_argument("--ks", type=int, nargs="*", default=None)
    parser.add_argument("--vocab-size", type=int, default=1000)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draft-embed", type=int, default=96)
    parser.add_argument("--verify-embed", type=int, default=128)
    parser.add_argument("--draft-layers", type=int, default=1)
    parser.add_argument("--verify-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    args = parser.parse_args()

    run_copy_demo(args)


if __name__ == "__main__":
    main_demo()
