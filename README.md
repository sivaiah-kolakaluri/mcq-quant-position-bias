# Beyond Accuracy: Quantization, Position Bias and Answer Consistency

Code and per-prompt results for the paper
**"Beyond Accuracy: How 8-bit and 4-bit Quantization Change Position Bias and Answer Consistency
in Multiple-Choice LLM Evaluation"** (Sivaiah Kolakaluri, 2026; arXiv link to follow).

We evaluate 7 small open-weight models at 16-bit, 8-bit (bitsandbytes LLM.int8()) and 4-bit
(bitsandbytes NF4) on ARC-Challenge, GPQA and MMLU-Pro. Every question is scored under all cyclic
rotations of its options, so the correct answer appears once in every position
(63 runs, 331,632 scored prompts).

## Main findings

- 8-bit barely changes multiple-choice behaviour (position bias and answer consistency), although it
  flips more answers than a repeated 16-bit run in all 21 model-benchmark pairs.
- 4-bit NF4 lowers answer consistency, changes position bias in a model-dependent way that is more
  often an increase, and flips 2.1-2.8 times as many answers as 8-bit; the least accurate models
  tend to be affected most.
- A simple prior correction removes much of the extra position bias, but not the loss of consistency.

## Repository layout

| Path | Contents |
|---|---|
| `src/eval_position_bias.py` | Scores one model x precision x benchmark: builds all cyclic rotations, reads next-token log-probabilities of the option letters, writes one CSV |
| `src/eval_position_bias_chat.py` | Chat-template variant for the pre-registered sensitivity check (`docs/PREREGISTRATION_chat_template.md`); imports the main script and changes only the prompt wrapper |
| `src/run_all.py` | Runs all 63 evaluations, one worker per GPU, skipping finished runs |
| `analysis/analyze.py` | All metrics from the per-prompt CSVs: tie-aware accuracy, RStd, consistency, flips, paired bootstrap tests with Holm correction, ranking analyses, prior correction |
| `analysis/make_tables.py`, `figs.py`, `figs2.py` | Paper tables (LaTeX) and figures |
| `analysis/gen_numbers.py` | Writes every number quoted in the paper text as a LaTeX macro |
| `analysis/data_checks.py`, `dtype_check.py` | Completeness and sanity checks; checks that the stored values are consistent with bfloat16 in every run |
| `analysis/chat_template_checks.py` | The four pre-registered chat-template checks (`results/analysis/chat_template_checks.csv`) |
| `analysis/tie_policy_check.py` | Recomputes all changes with first-letter tie-breaking and compares their direction (robustness check) |
| `analysis/numerical_baseline.py` | Repeated 16-bit run vs original: flips, tie-only floor, and a placebo test of the paper's bootstrap |
| `results/chat_template/` | 18 pre-registered chat-template runs (SmolLM3-3B, Qwen3.5-4B), same columns, plus `versions_chat.txt` |
| `results/analysis_chat/` | `analyze.py` outputs for the chat-template set (its own Holm family of 12 tests) |
| `results/rerun16_bs8/` | 21 repeated 16-bit runs at batch size 8 (numerical baseline), same columns, plus the package versions of both sessions |
| `results/per_prompt/` | 63 CSVs, one per run (columns below; full description in `results/README.md`) |
| `results/analysis/` | Analysis outputs used in the paper |
| `results/figures/` | Paper figures (PDF and PNG) |
| `docs/PREREGISTRATION_chat_template.md` | Pre-registered plan for the chat-template sensitivity check (written before any of those runs) |
| `docs/REPRODUCIBILITY.md` | Environment, run history, exact settings, known deviations |

Per-prompt columns: `model, precision, benchmark, qid, subject, shift, correct_pos, pred_pos,
correct, logprobs` (log-probabilities of the option letters A, B, ... over the full vocabulary,
rounded to 5 decimals).

## Environment and which session produced which files

| Result files | Where / when | Versions |
|---|---|---|
| `results/per_prompt/`: Qwen3-4B on ARC-Challenge (3 runs) | Colab, Oct 30, 2026 | transformers 5.22.0, bitsandbytes 0.50.2 |
| `results/per_prompt/`: the other 60 runs | Kaggle, Nov 1-3, 2026 | latest releases on those dates; exact versions not recorded |
| `results/rerun16_bs8/`: ARC-Challenge and GPQA (14 runs) | Kaggle, Nov 3, 2026 | torch 2.11.0+cu130, **transformers 5.22.1**, bitsandbytes 0.50.2, accelerate 1.15.0, datasets 5.1.0 (`versions.txt`) |
| `results/rerun16_bs8/`: MMLU-Pro (7 runs) | Kaggle, Nov 7, 2026 | same, except **transformers 5.22.2** (`versions_session2.txt`) |

All runs used NVIDIA T4 GPUs. The stored values are consistent with bfloat16 in every run
(`analysis/dtype_check.py`); no log of the original runs records the number format directly.

## Numerical baseline (`results/rerun16_bs8/`)

Every 16-bit run was repeated once with batch size 8 instead of 16, on identical prompts and
otherwise identical settings (21 files, same columns as `per_prompt/`). The runs were made in two
Kaggle sessions; the second session copied in the 14 files of the first, and these were verified to
be byte-identical to the first session's output. `analysis/numerical_baseline.py` compares each
repeated run with the original 16-bit run (flip rate, tie-only floor, placebo test of the paper's
bootstrap) and writes `results/analysis/numerical_baseline*.csv`.

## No benchmark text is redistributed

The result files contain question identifiers, positions and log-probabilities only, no question
or option text. `src/eval_position_bias.py` rebuilds the prompts from the public Hugging Face
datasets (`allenai/ai2_arc`, `Idavidrein/gpqa`, `TIGER-Lab/MMLU-Pro`). GPQA is gated: accept its
terms on Hugging Face before running.

## Reproduce the analysis (CPU only, about 5 minutes)

```bash
pip install -r requirements.txt
python analysis/data_checks.py results/per_prompt
python analysis/analyze.py results/per_prompt results/analysis
python analysis/make_tables.py results/analysis tables/
python analysis/figs.py results/per_prompt results/analysis figures/
python analysis/figs2.py results/analysis figures/
python analysis/numerical_baseline.py results/per_prompt results/rerun16_bs8 results/analysis
python analysis/tie_policy_check.py results/per_prompt results/analysis
python analysis/analyze.py results/chat_template results/analysis_chat
python analysis/chat_template_checks.py results/chat_template results/analysis_chat results/analysis
python analysis/gen_numbers.py results/analysis numbers.tex
```

## Rerun the evaluations (GPU)

```bash
hf auth login                    # gated models (Llama) and GPQA; older versions: huggingface-cli login
python src/run_all.py --dry_run  # lists missing runs
python src/run_all.py --gpus 0 1
```

One run on a single NVIDIA T4 takes about 5 minutes (ARC, 8-bit) to 2.7 hours (MMLU-Pro, 16-bit
or NF4). The paper's runs used free Colab and Kaggle T4 GPUs; see `docs/REPRODUCIBILITY.md`.

## Tie handling

In 4.9% of prompts two or more letters share the highest stored log-probability. Metrics treat a
tie as a uniform random choice among the tied letters, independently for every prompt and run,
and report the exact expected value. This is computed analytically, with no random sampling and no
seed (see `analysis/analyze.py`).

## Known limitations

- The numerical baseline is one repeated 16-bit run per model and benchmark (batch size 8 instead
  of 16); other GPUs or library versions are not covered. 8-bit flip rates exceed the rerun's in all
  21 pairs.
- Scoring uses next-token letter log-probabilities under a single prompt template.
- Only bitsandbytes LLM.int8() and NF4 are covered (no GPTQ or AWQ).

## Citation

See `CITATION.cff`.

## License

Code: MIT (see `LICENSE`). The benchmarks and models keep their own licenses.
