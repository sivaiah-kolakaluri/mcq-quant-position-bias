# Decision log

The main scientific and methodological decisions in this project, in order, with the reason for
each. Dates are given where they were recorded; "before Oct 30" covers the planning phase before
the first run.

## Planning (before Oct 30, 2026)

1. **Research question.** The starting idea was whether the order of answer options changes model
   rankings on multiple-choice benchmarks. Because prior work already shows this at full precision
   (Alzahrani et al., ACL 2024; Pezeshkpour and Hruschka, 2024; Zheng et al., ICLR 2024), the
   question was narrowed to whether *weight quantization* changes position bias and answer
   consistency.
2. **Locked plan.** Three research questions (bias and consistency; ranking stability; whether
   averaging over rotations protects rankings), 16-bit / LLM.int8() / NF4, ARC-Challenge, GPQA and
   MMLU-Pro, all cyclic rotations of the options, next-token letter log-likelihood scoring, and
   Kendall's tau with bootstrap intervals. The rule was not to deviate from this plan unless
   necessary; the deviations below are recorded as such.
3. **Models.** Seven small open-weight models (1.7B-4B) from five families, chosen so that every
   precision fits on one 16 GB T4 GPU.
4. **Sampling.** Seed 0; 1,000 ARC-Challenge questions (4 options), 1,000 MMLU-Pro questions (10
   options), and all 448 GPQA main questions (one seeded option order per question).

## Running the experiments (Oct 30 - Nov 3, 2026)

5. **Compute.** The first three runs (Qwen3-4B on ARC-Challenge) used Google Colab. Because of
   Colab's GPU limits, the other 60 runs moved to Kaggle (2x T4), one model per GPU at a time.
6. **Out-of-memory handling.** Long GPQA and MMLU-Pro prompts ran out of memory at batch size 16, so
   a failing batch is split in half and retried, with the retry moved outside the exception handler
   so that the failed batch's memory is released first. This changes batching only, not prompts or
   scoring.

## Analysis (Nov 3 - 7, 2026)

7. **Tie handling.** An independent check found that breaking ties by taking the first letter
   favours position A. Ties are therefore treated as a uniform random choice, and every metric
   reports its exact expected value, computed analytically with no sampling.
8. **Statistics.** The paired bootstrap was raised from 2,000 to 20,000 resamples, because 2,000
   was too coarse for the Holm correction over 42 tests per metric. A split-half noise baseline and
   a paired test of the stability difference replaced informal comparisons of confidence intervals.
9. **Prior correction.** A simplified version of PriDe (Zheng et al., 2024): position priors
   estimated on 5% of questions, evaluated on the other 95%, averaged over 20 splits that are shared
   across precisions.
10. **Reporting.** Every number in the paper is generated as a LaTeX macro from the analysis
    outputs, so that no number is typed by hand. The released files contain question IDs and
    log-probabilities only, not benchmark text (GPQA asks that its questions not be republished).
11. **Number format.** The run logs don't record the 16-bit dtype. It was inferred as bfloat16 from
    the code and from the spacing of the stored log-probabilities, and the paper says it is
    inferred, not logged.
12. **Numerical baseline (deviation from the locked plan).** To separate quantization from
    run-to-run noise, every 16-bit run was repeated with batch size 8 instead of 16 (Nov 3: ARC and
    GPQA; Nov 7: MMLU-Pro). It turns the paper's largest limitation into a measured quantity and
    leaves the main runs unchanged.
13. **Claims kept narrow.** Statements were revised to match the evidence, for example: a
    correlation between the 8-bit/noise ratio and the tie rate was dropped because it follows from
    how the ratio is built, and the paper states that in 6 of 21 pairs the 8-bit accuracy change
    exceeds the repeat-run spread.
14. **One paper, not two.** Splitting the study into two papers was considered and rejected, because
    each half would rest on the same runs and make a weaker claim. GPTQ/AWQ is left for a later
    version.

## After the external audit (Nov 8 - 9, 2026)

15. **Code fixes.** Two reproducibility issues found by the audit were fixed: ties after the prior
    correction are now detected within 1e-9 instead of by exact float equality (results had varied
    slightly between computers), and values on a rounding boundary are rounded correctly. One number
    in the text changed by 0.1.
16. **Scope and wording.** Claims are scoped to letter-probability scoring with a plain-text prompt;
    "flips" means correctness flips, and answer-change rates are reported alongside; borderline Holm
    results are labelled as sensitive to the bootstrap draws; figure intervals are labelled
    pointwise; a non-significant change in ranking stability is not read as evidence of no change.
17. **Robustness from existing data.** The conclusions were rechecked without the two models with
    low letter probability, and under first-letter tie-breaking.
18. **Chat-template check (deviation from the locked plan, pre-registered).** Because the audit's
    main concern was the validity of letter scoring for SmolLM3-3B and Qwen3.5-4B, and enough free
    GPU time remained, both models were re-evaluated with their own chat templates (18 runs). The
    plan was pushed to GitHub before any run (`docs/PREREGISTRATION_chat_template.md`, Nov 8, 23:46
    UTC); a smoke test then led to pinning a date in SmolLM3's template before the full run. All
    four checks are reported in the paper, including the ones that did not fully hold.
