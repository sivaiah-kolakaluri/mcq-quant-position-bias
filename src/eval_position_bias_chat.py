#!/usr/bin/env python3
"""
Chat-template variant of eval_position_bias.py (pre-registered sensitivity check,
docs/PREREGISTRATION_chat_template.md).

Everything is imported from eval_position_bias.py unchanged, except the prompt wrapper:
the plain prompt without its final "Answer:" line becomes one user message, formatted with
the model's own chat template (thinking disabled), and the assistant turn is prefilled with
"Answer:". Scoring is unchanged (next-token log-probabilities of " A", " B", ...).
The rendered text already contains the template's special tokens, so it is tokenized with
add_special_tokens=False.

Example:
  python eval_position_bias_chat.py --model Qwen/Qwen3.5-4B --precision nf4 --benchmark arc
  python eval_position_bias_chat.py --model Qwen/Qwen3.5-4B --precision 16bit --benchmark arc --n 5 --smoke
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_position_bias as E  # noqa: E402
import pandas as pd             # noqa: E402
import torch                    # noqa: E402


def build_chat_prompt(tok, question, choices):
    plain = E.build_prompt(question, choices)
    assert plain.endswith("\nAnswer:")
    user = plain[: -len("\nAnswer:")]
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    # SmolLM3's template writes the current date into the prompt; pin it so every run sees the
    # same text (date of the pre-registration and smoke test)
    text = re.sub(r"Today Date: \d{1,2} \w+ \d{4}", "Today Date: 08 November 2026", text)
    return text + "Answer:"


@torch.no_grad()
def _score_batch_chat(model, tok, texts, letter_ids):
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    try:
        logits = model(**enc, logits_to_keep=1).logits[:, -1, :]
    except TypeError:
        logits = model(**enc).logits[:, -1, :]
    return torch.log_softmax(logits.float(), dim=-1)[:, letter_ids].cpu().tolist()


E._score_batch = _score_batch_chat   # _score_safe looks this name up at call time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", required=True, choices=["16bit", "int8", "nf4"])
    ap.add_argument("--benchmark", required=True, choices=["arc", "mmlu_pro", "gpqa"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--out_dir", default="results_chat")
    ap.add_argument("--smoke", action="store_true", help="print the rendered prompt and letter mass; write nothing")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.model.replace('/', '__')}__{args.precision}__{args.benchmark}.csv")
    if os.path.exists(out_path) and not args.smoke:
        print(f"exists, skipping: {out_path}")
        return

    questions = E.load_benchmark(args.benchmark, args.n, args.seed)
    n_opts = len(questions[0]["choices"])
    model, tok = E.load_model(args.model, args.precision)
    letter_ids = E.letter_token_ids(tok, n_opts)

    meta, prompts = [], []
    for q in questions:
        for k, choices, correct_pos in E.cyclic_shifts(q["choices"], q["answer"]):
            meta.append(dict(qid=q["qid"], subject=q["subject"], shift=k, correct_pos=correct_pos))
            prompts.append(build_chat_prompt(tok, q["question"], choices))

    logprobs = E.score_prompts(model, tok, prompts, letter_ids, args.batch_size)
    mass = sorted(sum(math.exp(v) for v in lp) for lp in logprobs)
    if args.smoke:
        print("=== rendered prompt (first) ===\n" + prompts[0] + "<<END>>")
        print("letter tokens:", [tok.decode([i]) for i in letter_ids])
        print(f"median letter mass: {mass[len(mass) // 2]:.3f} over {len(mass)} prompts")
        return

    rows = []
    for m, lp in zip(meta, logprobs):
        pred = max(range(n_opts), key=lambda j: lp[j])
        rows.append(dict(model=args.model, precision=args.precision, benchmark=args.benchmark,
                         **m, pred_pos=pred, correct=int(pred == m["correct_pos"]),
                         logprobs=json.dumps([round(x, 5) for x in lp])))
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"saved {out_path}  acc={pd.DataFrame(rows).correct.mean():.3f}  median letter mass={mass[len(mass) // 2]:.3f}")


if __name__ == "__main__":
    main()
