# Results

## per_prompt/
One CSV per run, named `<model with / replaced by __>__<precision>__<benchmark>.csv`
(7 models x 3 precisions x 3 benchmarks = 63 files). One row per scored prompt
(question x rotation): ARC-Challenge 1,000 x 4, GPQA 448 x 4, MMLU-Pro 1,000 x 10.

| Column | Meaning |
|---|---|
| `model` | Hugging Face model id |
| `precision` | `16bit` (bfloat16), `int8` (LLM.int8()), `nf4` (bitsandbytes NF4) |
| `benchmark` | `arc`, `gpqa`, `mmlu_pro` |
| `qid` | Question id in the source dataset (no question text is stored) |
| `subject` | Category from the dataset (MMLU-Pro category, GPQA high-level domain; `science` for ARC) |
| `shift` | Rotation k: option shown at position j is original option (j + k) mod n |
| `correct_pos` | 0-based position of the correct option in this prompt (0 = A) |
| `pred_pos` | 0-based position of the highest letter log-probability (first one if tied; the analysis handles ties itself) |
| `correct` | 1 if `pred_pos == correct_pos` |
| `logprobs` | JSON list of natural-log probabilities of the letters A, B, ... (softmax over the full vocabulary), rounded to 5 decimals |

RStd for a run: per-position accuracy = mean correctness over rows with each `correct_pos`;
RStd = 100 x population standard deviation of these accuracies (the paper's analysis uses the
tie-aware version, see `analysis/analyze.py`).

## rerun16_bs8/
Repeated 16-bit runs with batch size 8 (numerical baseline), same columns as `per_prompt/`;
`versions.txt` (ARC-Challenge, GPQA; Nov 3) and `versions_session2.txt` (MMLU-Pro; Nov 7) list the
package versions of the two sessions.

## analysis/
Outputs of `analysis/analyze.py` and `analysis/numerical_baseline.py`, used for the paper's tables,
figures and numbers.

## figures/
The paper's figures.
