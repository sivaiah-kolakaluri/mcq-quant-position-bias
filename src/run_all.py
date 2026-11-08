#!/usr/bin/env python3
"""
Runs every model x benchmark x precision evaluation that does not have a result file yet,
using one worker per visible GPU (the paper's runs used Kaggle's 2x T4). Each worker takes the
next model from a shared queue and runs all of its benchmarks and precisions.

Usage:  python src/run_all.py [--gpus 0 1] [--max_hours 11] [--dry_run]
Results go to results/per_prompt/ (one CSV per run); existing files are skipped, so the script
can be restarted after an interruption.
"""
import argparse, os, shutil, subprocess, threading, time

MODELS = ["Qwen/Qwen3-4B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3.5-4B",
          "meta-llama/Llama-3.2-3B-Instruct", "HuggingFaceTB/SmolLM3-3B",
          "microsoft/Phi-4-mini-instruct", "ibm-granite/granite-4.1-3b"]
BENCHES, PRECS = ["arc", "gpqa", "mmlu_pro"], ["16bit", "int8", "nf4"]
# rough hours per run on one T4 (from the paper's runs); a job is skipped if it cannot finish in time
EST = {"arc": {"16bit": .4, "int8": .1, "nf4": .4},
       "gpqa": {"16bit": .55, "int8": .15, "nf4": .55},
       "mmlu_pro": {"16bit": 2.6, "int8": .6, "nf4": 2.6}}
OUT = "results/per_prompt"
EVAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_position_bias.py")


def out_path(m, p, b):
    return os.path.join(OUT, f"{m.replace('/', '__')}__{p}__{b}.csv")


def todo_for(m):
    return [(b, p) for b in BENCHES for p in PRECS if not os.path.exists(out_path(m, p, b))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--max_hours", type=float, default=11.0)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()
    start = time.time()
    queue = sorted([m for m in MODELS if todo_for(m)], key=lambda m: -len(todo_for(m)))
    print(f"{sum(len(todo_for(m)) for m in MODELS)} runs to do (out of {len(MODELS) * 9})")
    for m in queue:
        print(f"  {m}: {len(todo_for(m))} runs")
    if args.dry_run:
        return
    lock = threading.Lock()

    def worker(gpu):
        while True:
            with lock:
                if not queue:
                    return
                m = queue.pop(0)
            for b, p in todo_for(m):
                if (time.time() - start) / 3600 + EST[b][p] > args.max_hours:
                    print(f"[GPU{gpu}] skip {m} | {p} | {b} (not enough time left)", flush=True)
                    continue
                t = time.time()
                r = subprocess.run(["python", EVAL, "--model", m, "--precision", p, "--benchmark", b,
                                    "--n", "1000", "--batch_size", "16", "--out_dir", OUT],
                                   env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                                   capture_output=True, text=True)
                lines = [l for l in (r.stdout + r.stderr).splitlines() if "MatMul8bitLt" not in l]
                print(f"[GPU{gpu}] {m} | {p} | {b} | {(time.time() - t) / 60:.0f} min\n" + "\n".join(lines[-3:]) + "\n",
                      flush=True)
            shutil.rmtree(os.path.expanduser(f"~/.cache/huggingface/hub/models--{m.replace('/', '--')}"),
                          ignore_errors=True)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    print("DONE. Result files:", len(os.listdir(OUT)))


if __name__ == "__main__":
    main()
