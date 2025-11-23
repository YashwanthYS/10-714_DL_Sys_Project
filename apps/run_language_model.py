import argparse
import os
import sys
from typing import List, Tuple

sys.path.append("python")

import numpy as np

import needle as ndl
import needle.nn as nn
from apps.models import LanguageModel
from needle.autograd import Tensor


def best_device():
    try:
        cu = ndl.cuda()
        if hasattr(cu, "enabled") and cu.enabled():
            return cu
    except Exception:
        pass
    return ndl.cpu()


def load_ptb_words(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = f.read().strip().splitlines()
    # append <eos> per line and split by whitespace
    toks: List[str] = []
    for ln in lines:
        toks.extend(ln.strip().split() + ["<eos>"])
    return toks


def build_vocab(words: List[str], max_vocab: int | None = None) -> Tuple[dict, list]:
    # count freq and keep most common
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1
    it = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    specials = ["<unk>"]
    vocab = specials + [w for w, _ in it]
    if max_vocab is not None:
        vocab = vocab[:max_vocab]
    stoi = {w: i for i, w in enumerate(vocab)}
    return stoi, vocab


def encode_words(words: List[str], stoi: dict) -> np.ndarray:
    unk = stoi.get("<unk>")
    ids = np.array([stoi.get(w, unk) for w in words], dtype=np.int32)
    return ids


def batchify(ids: np.ndarray, batch_size: int) -> np.ndarray:
    # Trim to multiple of batch
    n_batch = ids.shape[0] // batch_size
    data = ids[: n_batch * batch_size]
    data = data.reshape(batch_size, n_batch).T  # shape (nbatch, batch)
    return data


def get_batch(batches: np.ndarray, i: int, bptt: int, device, dtype="float32") -> Tuple[Tensor, Tensor]:
    nbatch, bs = batches.shape
    seq = min(bptt, nbatch - 1 - i)
    x = batches[i : i + seq]  # (seq, bs)
    y = batches[i + 1 : i + 1 + seq]  # (seq, bs)
    x_t = Tensor(x.astype(np.float32), device=device, dtype=dtype, requires_grad=False)
    y_t = Tensor(y.astype(np.float32).reshape(-1), device=device, dtype=dtype, requires_grad=False)
    return x_t, y_t


def generate(model: LanguageModel, stoi: dict, itos: list, prompt: str, max_len: int, device):
    model.eval()
    words = prompt.strip().split()
    ids = [stoi.get(w, stoi.get("<unk>")) for w in words]
    h = None
    out_ids = list(ids)
    for _ in range(max_len):
        x = Tensor(np.array(out_ids[-1:], dtype=np.float32).reshape(1, 1), device=device, dtype="float32", requires_grad=False)
        logits, h = model(x, h)
        # logits shape (1*1, V)
        prob = np.argmax(logits.numpy(), axis=1)[0]
        out_ids.append(int(prob))
    # decode
    out_words = [itos[i % len(itos)] for i in out_ids]
    return " ".join(out_words)


def main():
    p = argparse.ArgumentParser(description="Train and sample the RNN/LSTM LanguageModel on PTB")
    p.add_argument("--ptb-dir", type=str, default=os.path.join("data", "ptb"))
    p.add_argument("--seq-len", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--embed", type=int, default=200)
    p.add_argument("--hidden", type=int, default=200)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--model", choices=["rnn", "lstm"], default="lstm")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--vocab-cap", type=int, default=10000)
    p.add_argument("--prompt", type=str, default="the quick brown fox")
    p.add_argument("--gen-len", type=int, default=50)
    args = p.parse_args()

    device = best_device()
    print(f"Using device: {device}")

    # Load corpus and build vocab
    words = load_ptb_words(os.path.join(args.ptb_dir, "train.txt"))
    stoi, itos = build_vocab(words, max_vocab=args.vocab_cap)
    ids = encode_words(words, stoi)
    batches = batchify(ids, args.batch_size)

    # Model
    vocab_size = len(itos)
    lm = LanguageModel(
        embedding_size=args.embed,
        output_size=vocab_size,
        hidden_size=args.hidden,
        num_layers=args.layers,
        seq_model=args.model,
        device=device,
        dtype="float32",
    )
    opt = ndl.optim.Adam(lm.parameters(), lr=args.lr, weight_decay=0.0)
    loss_fn = nn.SoftmaxLoss()

    # Train
    lm.train()
    nbatch = batches.shape[0]
    step = 0
    start = time.perf_counter()
    while step < args.steps:
        i = (step * args.seq_len) % (nbatch - 1)
        x_t, y_t = get_batch(batches, i, args.seq_len, device)
        logits, _ = lm(x_t)
        S, V = logits.shape
        loss = loss_fn(logits, y_t)
        opt.reset_grad()
        loss.backward()
        opt.step()
        step += 1
        if step % 50 == 0 or step == args.steps:
            elapsed = time.perf_counter() - start
            ppl = float(np.exp(float(loss.numpy())))
            print(f"step {step}/{args.steps}, loss={float(loss.numpy()):.4f}, ppl~{ppl:.1f}, elapsed={elapsed:.1f}s")

    # Sample
    text = generate(lm, stoi, itos, args.prompt, args.gen_len, device)
    print("\nPrompt:", args.prompt)
    print("Sample:", text)


if __name__ == "__main__":
    main()

