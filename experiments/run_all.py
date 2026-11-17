"""
experiments/run_all.py
Runs the whole study in order, skipping any step whose output already exists (so a
disconnected Colab session can simply be re-run and it resumes).

    python experiments/run_all.py              # everything except the holdout
    python experiments/run_all.py --holdout    # the one 2025 holdout run (after reviewing folds)
    python experiments/run_all.py --force      # recompute every step

Steps (outputs in results/, logs in results/logs/):
  1. token causality audit, Fold 1
  2. HAR-RV + GARCH baselines, Folds 1-6
  3. GPT leak audit, Fold 1, seeds 0 1 2 (next hour, H = 12)
  4. GPT leak audit, Folds 2-6, seed 0 (H = 12)
  5. negative control on simulated GARCH, seed 0 (H = 12)
  6. GPT leak audit, Fold 1, next day (H = 288), direct head only
  7. cross-fold inflation tables (H = 12 and H = 288)
  8. figures
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import subprocess
import time

PY = sys.executable


def step(name, cmd, output, force, log_dir="results/logs"):
    if output and os.path.exists(output) and not force:
        print(f"[skip] {name}: {output} exists")
        return
    os.makedirs(log_dir, exist_ok=True)
    log = os.path.join(log_dir, name.replace(" ", "_").replace(",", "") + ".log")
    print(f"\n[run ] {name}\n       {' '.join(cmd)}\n       log: {log}", flush=True)
    t0 = time.time()
    with open(log, "w") as lf:
        env = dict(os.environ, PYTHONUNBUFFERED="1")  # live progress in Colab
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
        for line in p.stdout:
            lf.write(line)
            if "Warning" not in line and "warn(" not in line:
                print("       " + line, end="", flush=True)
        p.wait()
    if p.returncode != 0:
        raise SystemExit(f"Step '{name}' failed (exit {p.returncode}); see {log}")
    print(f"[done] {name} in {(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", action="store_true", help="score the 2025 holdout (once)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--epochs", default="5")
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    args = ap.parse_args()
    f, ep = args.force, ["--epochs", args.epochs]

    if args.holdout:
        step("holdout 2025", [PY, "experiments/run_holdout.py", "--seeds", "0", *ep], "results/HOLDOUT_SCORED.json", False)
        step("figures", [PY, "presentation/generate_figures.py"], None, True)
        return

    step("boundary audit fold 1", [PY, "experiments/run_boundary_experiment.py", "--fold", "1"],
         "results/boundary_mismatch_results.json", f)
    for k in args.folds:
        step(f"baselines fold {k}", [PY, "experiments/run_baselines.py", "--fold", str(k)],
             f"results/fold_{k}_baselines.json", f)
    if 1 in args.folds:
        step("leak audit fold 1 seeds 0-2", [PY, "experiments/run_leak_experiments.py", "--fold", "1",
                                             "--seeds", "0", "1", "2", *ep], "results/losses/fold_1_h12.npz", f)
    for k in [k for k in args.folds if k != 1]:
        step(f"leak audit fold {k}", [PY, "experiments/run_leak_experiments.py", "--fold", str(k), "--seeds", "0", *ep],
             f"results/losses/fold_{k}_h12.npz", f)
    step("negative control", [PY, "experiments/run_oracle_experiment.py", "--seeds", "0", *ep],
         "results/negative_control_results.json", f)
    if 1 in args.folds:
        step("leak audit fold 1 next day", [PY, "experiments/run_leak_experiments.py", "--fold", "1", "--horizon", "288",
                                            "--seeds", "0", "--no-generative", *ep], "results/losses/fold_1_h288.npz", f)
    step("aggregate h12", [PY, "experiments/aggregate_folds.py", "--horizon", "12"], None, True)
    if os.path.exists("results/losses/fold_1_h288.npz"):
        step("aggregate h288", [PY, "experiments/aggregate_folds.py", "--horizon", "288"], None, True)
    step("figures", [PY, "presentation/generate_figures.py"], None, True)
    print("\nAll fold experiments done. Review results/inflation_table_h12.md, freeze the code, "
          "then run:  python experiments/run_all.py --holdout")


if __name__ == "__main__":
    main()
