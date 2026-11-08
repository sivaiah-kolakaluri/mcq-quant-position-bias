#!/usr/bin/env python3
"""
Position-bias evaluation under quantization (locked plan, Weeks 1-2).

For every question, runs ALL cyclic shifts of the answer options, so the correct
answer appears exactly once in every position (this also covers the
"correct answer fixed at each position" perturbation). Each prompt is scored by
the next-token log-likelihood of the option letters (A, B, C, ...), so
formatting failures in quantized models can't be mistaken for position bias.

Precisions (bitsandbytes):
  16bit -> bf16 if the GPU supports it, else fp16
  int8  -> LLM.int8 (load_in_8bit)
  nf4   -> 4-bit NF4 (load_in_4bit, quant_type="nf4")

Benchmarks:
  arc       allenai/ai2_arc, ARC-Challenge test (4-option questions only)
  mmlu_pro  TIGER-Lab/MMLU-Pro test (10-option questions only)
  gpqa      Idavidrein/gpqa, gpqa_main (gated: accept terms on HF + `huggingface-cli login`)

Output: one CSV per (model, precision, benchmark) in --out_dir.

Example:
  pip install -U transformers accelerate bitsandbytes datasets pandas scipy tqdm
  python eval_position_bias.py --model <hf-model-id> --precision nf4 --benchmark arc --n 1000
"""
import argparse
import gc
import json
import os
import random

# Reduce GPU memory fragmentation (must be set before torch is imported).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

LETTERS = "ABCDEFGHIJ"


# ----------------------------------------------------------------------------- data
def load_benchmark(name: str, n: int, seed: int):
    """Return a list of dicts: qid, subject, question, choices (list), answer (index)."""
    rows = []
    if name == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        for ex in ds:
            labels = list(ex["choices"]["label"])
            if len(labels) != 4 or ex["answerKey"] not in labels:
                continue
            rows.append(dict(qid=ex["id"], subject="science", question=ex["question"].strip(),
                             choices=[c.strip() for c in ex["choices"]["text"]],
                             answer=labels.index(ex["answerKey"])))
    elif name == "mmlu_pro":
        ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        for ex in ds:
            if len(ex["options"]) != 10:
                continue
            rows.append(dict(qid=str(ex["question_id"]), subject=ex["category"],
                             question=ex["question"].strip(),
                             choices=[c.strip() for c in ex["options"]],
                             answer=int(ex["answer_index"])))
    elif name == "gpqa":
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        for i, ex in enumerate(ds):
            # GPQA lists the correct answer first, so fix a seeded "original" order.
            opts = [ex["Correct Answer"], ex["Incorrect Answer 1"],
                    ex["Incorrect Answer 2"], ex["Incorrect Answer 3"]]
            order = list(range(4))
            random.Random(f"{seed}-{i}").shuffle(order)
            rows.append(dict(qid=str(ex.get("Record ID") or f"gpqa-{i}"),
                             subject=ex.get("High-level domain", "unknown"),
                             question=ex["Question"].strip(),
                             choices=[opts[j].strip() for j in order],
                             answer=order.index(0)))
    else:
        raise ValueError(f"unknown benchmark {name}")

    # Same seed + same dataset order -> same subsample for every model/precision.
    if n and n < len(rows):
        rows = random.Random(seed).sample(rows, n)
    return rows


def cyclic_shifts(choices, answer):
    """Yield (shift, rotated_choices, correct_pos). new[i] = old[(i + k) % n]."""
    n = len(choices)
    for k in range(n):
        yield k, choices[k:] + choices[:k], (answer - k) % n


def build_prompt(question, choices):
    lines = ["The following is a multiple choice question. "
             "Answer with the letter of the correct option.", "",
             f"Question: {question}"]
    lines += [f"{L}. {c}" for L, c in zip(LETTERS, choices)]
    lines.append("Answer:")
    return "\n".join(lines)


# ---------------------------------------------------------------------------- model
def load_model(name: str, precision: str):
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) \
        else torch.float16
    kw = dict(device_map="auto")
    if precision == "16bit":
        kw["dtype"] = dtype
    elif precision == "int8":
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif precision == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
    else:
        raise ValueError(precision)

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"           # so the last position is the real last token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(name, **kw)
    model.eval()
    return model, tok


def letter_token_ids(tok, n_opts):
    """Token id of each letter as it appears right after 'Answer:' (i.e. ' A')."""
    base = tok.encode("Answer:", add_special_tokens=False)
    ids = []
    for L in LETTERS[:n_opts]:
        full = tok.encode("Answer: " + L, add_special_tokens=False)
        if full[:len(base)] != base or len(full) != len(base) + 1:
            raise ValueError(f"letter '{L}' is not a single token after 'Answer:' "
                             f"for this tokenizer; adjust build_prompt/letter_token_ids")
        ids.append(full[-1])
    return ids


@torch.no_grad()
def _score_batch(model, tok, texts, letter_ids):
    enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
    try:
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
    except TypeError:  # older transformers
        logits = model(**enc).logits[:, -1, :]
    return torch.log_softmax(logits.float(), dim=-1)[:, letter_ids].cpu().tolist()


def _score_safe(model, tok, texts, letter_ids):
    """Score a batch; on CUDA out-of-memory, split it in half and retry.

    The retry happens OUTSIDE the except block: inside it, the exception's
    traceback still holds the failed attempt's GPU tensors, so the retry
    would have almost no free memory.
    """
    oom = False
    try:
        return _score_batch(model, tok, texts, letter_ids)
    except torch.OutOfMemoryError:
        if len(texts) == 1:
            raise  # a single prompt doesn't fit: nothing more we can do
        oom = True
    if oom:
        gc.collect()
        torch.cuda.empty_cache()
        mid = len(texts) // 2
        return (_score_safe(model, tok, texts[:mid], letter_ids) +
                _score_safe(model, tok, texts[mid:], letter_ids))


def score_prompts(model, tok, prompts, letter_ids, batch_size):
    """Return log-probs (over the full vocab) of each option letter, per prompt."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))  # less padding
    out = [None] * len(prompts)
    for s in tqdm(range(0, len(order), batch_size), desc="scoring"):
        idx = order[s:s + batch_size]
        lp = _score_safe(model, tok, [prompts[i] for i in idx], letter_ids)
        for i, row in zip(idx, lp):
            out[i] = row
    return out


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True, choices=["16bit", "int8", "nf4"])
    ap.add_argument("--benchmark", required=True, choices=["arc", "mmlu_pro", "gpqa"])
    ap.add_argument("--n", type=int, default=1000, help="questions to subsample (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tag = args.model.replace("/", "__")
    out_path = os.path.join(args.out_dir, f"{tag}__{args.precision}__{args.benchmark}.csv")
    if os.path.exists(out_path) and not args.overwrite:
        print(f"exists, skipping: {out_path}")
        return

    questions = load_benchmark(args.benchmark, args.n, args.seed)
    n_opts = len(questions[0]["choices"])
    print(f"{args.benchmark}: {len(questions)} questions x {n_opts} shifts")

    meta, prompts = [], []
    for q in questions:
        for k, choices, correct_pos in cyclic_shifts(q["choices"], q["answer"]):
            meta.append(dict(qid=q["qid"], subject=q["subject"], shift=k, correct_pos=correct_pos))
            prompts.append(build_prompt(q["question"], choices))

    model, tok = load_model(args.model, args.precision)
    letter_ids = letter_token_ids(tok, n_opts)
    logprobs = score_prompts(model, tok, prompts, letter_ids, args.batch_size)

    rows = []
    for m, lp in zip(meta, logprobs):
        pred = max(range(n_opts), key=lambda j: lp[j])
        rows.append(dict(model=args.model, precision=args.precision, benchmark=args.benchmark,
                         **m, pred_pos=pred, correct=int(pred == m["correct_pos"]),
                         logprobs=json.dumps([round(x, 5) for x in lp])))
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    orig = df[df["shift"] == 0]["correct"].mean()
    avg = df["correct"].mean()
    by_pos = df.groupby("correct_pos")["correct"].mean().round(3).to_dict()
    print(f"saved {out_path}\n  acc(original order)={orig:.3f}  acc(all shifts)={avg:.3f}\n"
          f"  acc when correct answer is at position: "
          f"{ {LETTERS[k]: v for k, v in by_pos.items()} }")


if __name__ == "__main__":
    main()
