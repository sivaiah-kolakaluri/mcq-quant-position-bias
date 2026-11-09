# Reproducibility notes

## Hardware
All evaluations ran on NVIDIA T4 GPUs (16 GB): the first three runs on Google Colab (one T4),
the other 60 on Kaggle (2x T4, one run per GPU at a time).

## Software
Packages were installed with `pip install -U transformers accelerate bitsandbytes datasets` at the
start of each session.
- Recorded in the Colab session (Oct 30, 2026): transformers 5.22.0, bitsandbytes 0.50.2, pandas 2.2.3.
- Kaggle sessions (Nov 1-3, 2026; 60 of the 63 runs) installed the latest releases on those dates;
  exact versions were not recorded. Because LLM.int8() and NF4 kernels change between releases, flip rates for these
  runs may not reproduce exactly with other `bitsandbytes`/`transformers` versions.
- Number format: bfloat16 (inferred, see below). The script uses bfloat16 when
  `torch.cuda.is_bf16_supported()` is true, else float16, for the 16-bit weights and the NF4
  compute; 8-bit runs keep the model default (bfloat16) for unquantized layers. No log of the original
  runs records the number format, but the stored data are consistent with bfloat16 in all 63 runs: differences between letter log-probabilities are
  multiples of 2^-4 to 2^-2, the spacing of bfloat16 logits, in all but 9 of the 331,632 prompts (those 9
  are multiples of 2^-7, still consistent with bfloat16 at smaller logit sizes)
  (`python analysis/dtype_check.py results/per_prompt`). The same coarse spacing causes the
  4.9% of prompts with tied top letters.

## Fixed settings
- Question sampling: seed 0 (ARC-Challenge: 1,000 of the 4-option test questions; MMLU-Pro: 1,000
  of the 10-option test questions; GPQA main: all 448 questions, one seeded option order per
  question before rotation).
- All cyclic rotations of the options for every question.
- Prompt: plain text, no chat template, ending in `Answer:`; prediction = option letter with the
  highest next-token log-probability.
- Batch size 16, left padding, prompts sorted by length.
- 8-bit: `BitsAndBytesConfig(load_in_8bit=True)`; 4-bit: `BitsAndBytesConfig(load_in_4bit=True,
  bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=<16-bit dtype>)`.

## Run history and code versions
1. Qwen3-4B on ARC-Challenge at 16-bit, 8-bit and NF4 (Colab): first version of the evaluation
   script, without the out-of-memory retry. These ARC prompts are short and never ran out of memory.
2. Kaggle run 1 (32 runs): out-of-memory retry that splits a failing batch. Two runs
   (Qwen3.5-4B, GPQA, 16-bit and 8-bit) still failed because the retry started before the failed
   batch's memory was released; they were redone in run 2.
3. Kaggle runs 2 and 3 (28 runs): retry fixed (memory released before retrying), identical to
   `src/eval_position_bias.py` in this repository.
The retry only changes how prompts are grouped into batches, not the prompts or the scoring.

## Statistics
- Paired bootstrap over questions, 20,000 resamples for changes vs 16-bit; two-sided
  p = 2(k+1)/(B+1); Holm correction over the 42 tests of each metric.
- Kendall's tau (tau-b) intervals: 2,000 resamples; split-half noise baseline: 1,000 random splits.
- Prior correction: 5% calibration questions (minimum 5), 20 random splits, same splits for all
  precisions of a model, metrics on the held-out 95%.
- Bootstrap seeds are derived from the benchmark and model names (see `seed()` in
  `analysis/analyze.py`), so results are reproducible exactly.

## Numerical baseline (repeated 16-bit runs)
- Batch size 8 (original: 16), otherwise identical settings and prompts; 2x T4 on Kaggle.
- Session 1 (Nov 3, 2026): ARC-Challenge and GPQA, 14 runs; versions in
  `results/rerun16_bs8/versions.txt` (torch 2.11.0+cu130, transformers 5.22.1, bitsandbytes 0.50.2,
  accelerate 1.15.0, datasets 5.1.0, bf16 supported).
- Session 2 (Nov 7, 2026): MMLU-Pro, 7 runs; versions in `results/rerun16_bs8/versions_session2.txt`
  (same, except transformers 5.22.2).
- Session 2 received the 14 session-1 files as an input dataset and copied them into its output;
  all 14 were verified byte-identical (`cmp`) to the files downloaded after session 1.
- All 21 runs are consistent with bfloat16 (`python analysis/dtype_check.py results/rerun16_bs8`).
- Analysis: `python analysis/numerical_baseline.py results/per_prompt results/rerun16_bs8 results/analysis`.

## Changes after the external audit (Nov 8, 2026)
- `analysis/analyze.py`, prior correction: ties after subtracting the prior are now detected within
  1e-9 instead of by exact float equality. Exact equality let platform-specific rounding noise
  (~1e-15) decide some near-ties, so `calibration_fix.csv` differed slightly between computers.
  With the tolerance, changing the order of the arithmetic changes no output.
- `analysis/gen_numbers.py` and `analysis/make_tables.py`: values are rounded to 9 decimals
  before round-half-up, so that values stored as e.g. 0.5499999999999999 (exactly 0.55) round to
  0.6 as intended.

## Chat-template sensitivity check (pre-registered)
- Plan: `docs/PREREGISTRATION_chat_template.md`, pushed to GitHub on Nov 8, 2026 (23:46 UTC) before
  any run; implementation note (pinned date in SmolLM3's template) and a background correction were
  added on Nov 9 before the full run's results were seen.
- Runs: Kaggle, Nov 9, 2026, 2x T4, batch size 8, `src/eval_position_bias_chat.py`; versions in
  `results/chat_template/versions_chat.txt` (same as rerun session 2). All 18 files have the full
  set of prompts, identical to the main runs' prompts.
- Analysis: `analyze.py` on the 18 files as their own set, then `chat_template_checks.py`.

## Not done
- No GPTQ or AWQ runs (planned for a later version).
