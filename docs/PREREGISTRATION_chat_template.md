# Pre-registration: chat-template sensitivity check

Written on Nov 8, 2026, before any chat-template run was made. It responds to the external audit
(finding F02: validity of letter-only scoring for models that put little probability on the
option letters). Anything done differently from this plan will be reported as a deviation.

## Why
With the paper's plain-text prompt, two models put little next-token probability on the option
letters: SmolLM3-3B (median 27%) and Qwen3.5-4B (median 17%), against 40-92% for the other five.
Both are chat/reasoning models. The question is whether the paper's conclusions for these two
models depend on that format.

## What changes, and what doesn't
- **Models:** SmolLM3-3B and Qwen3.5-4B only.
- **Precisions, benchmarks, questions, seeds, rotations:** identical to the main study (16-bit,
  LLM.int8(), NF4; ARC-Challenge 1,000, GPQA 448, MMLU-Pro 1,000; seed 0; all cyclic shifts).
  That's 18 runs.
- **The only change is the prompt wrapper.** The instruction, question and options (the main
  prompt without its final `Answer:` line) become one user message. It is formatted with the
  model's own chat template (`apply_chat_template`, `add_generation_prompt=True`,
  `enable_thinking=False`), and the assistant turn is prefilled with `Answer:`. Scoring is
  unchanged: the next-token log-probabilities of " A", " B", ... over the full vocabulary. The
  rendered text is tokenized without adding extra special tokens.
- **Code:** `src/eval_position_bias_chat.py`, which imports `src/eval_position_bias.py` and
  changes only the prompt construction and tokenization flag. Batch size 8.
- **Analysis:** `analysis/analyze.py`, unchanged, run on the 18 new files as their own set
  (Holm correction over this set's 12 tests per metric: 2 models x 3 benchmarks x 2 quantized
  precisions).

## Questions and how they will be answered
1. **Letter mass.** Does the chat template raise the median letter probability mass? Reported
   per run, whatever it is.
2. **Do the paper's conclusions for these two models hold under the chat template?** Each is
   checked per benchmark and reported as holds / does not hold:
   - (a) 8-bit: no significant change in answer consistency, and every RStd change smaller than
     one point.
   - (b) NF4: answer consistency lower than at 16-bit (direction), with Holm-significant
     decreases reported.
   - (c) NF4 correctness-flip rate higher than the 8-bit correctness-flip rate.
   - (d) The direction of each model's NF4 RStd change (increase or decrease), compared with
     the plain-text result.
3. **Answer-change rate** (share of prompts whose chosen option changes vs 16-bit) is reported
   alongside the correctness-flip rate.

## Reporting commitment
All 18 runs and all four checks will be reported in the paper, whichever way they come out. If
a check fails, the paper's claims for these two models will be narrowed accordingly. If a run
cannot be completed within the GPU budget, that is reported, and the remaining runs are
analysed as planned.

## Not part of this check
No repeated-run baseline is made for the chat template, the other five models are not rerun,
and no other prompt variants are tried.
