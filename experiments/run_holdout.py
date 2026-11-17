"""
experiments/run_holdout.py
Scores the 2025 holdout ONCE with the frozen code: train on 2021-2024, test on 2025.

Protocol (proposal: "2025 held out and scored only once at the end"):
  * run this only after the six folds are done and the code is frozen (tagged);
  * the first real run writes results/HOLDOUT_SCORED.json (time, git commit, settings);
    any later run refuses to score again unless --i-know-this-rescores is passed, and
    that flag is recorded in the output so a re-score is never silent;
  * --dry-run exercises the identical code path on the Fold 6 split (test 2024) with a
    small subsample, so the script can be tested without looking at 2025.

Usage:
    python experiments/run_holdout.py --dry-run          # test the code path, never touches 2025
    python experiments/run_holdout.py --seeds 0          # the one real run
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import subprocess
from datetime import datetime, timezone

from data.folds import load_fold, load_holdout
from experiments.run_leak_experiments import parse_args, run_audit, print_table

LOCK = "results/HOLDOUT_SCORED.json"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main():
    extra = ["--dry-run", "--i-know-this-rescores"]
    flags = {f: f in sys.argv for f in extra}
    args = parse_args([a for a in sys.argv[1:] if a not in extra])

    if flags["--dry-run"]:
        args.fold = "holdout_dry_run (fold 6 split)"
        args.max_train = args.max_train or 512
        args.max_test = args.max_test or 200
        args.losses_out = None
        df, split = load_fold(6)
        res = run_audit(df, split, args)
        print_table(res)
        print("\nDry run finished on the Fold 6 split. 2025 was not loaded; nothing was written.")
        return

    if os.path.exists(LOCK) and not flags["--i-know-this-rescores"]:
        with open(LOCK) as f:
            prev = json.load(f)
        raise SystemExit(f"The 2025 holdout was already scored at {prev['scored_at_utc']} "
                         f"(commit {prev['git_commit'][:10]}). Results: {prev['results']}. Refusing to score again.")
    if args.max_train or args.max_test:
        raise SystemExit("Subsampling is not allowed on the real holdout run (use --dry-run to test).")

    args.fold = "holdout_2025"
    args.losses_out = f"results/losses/holdout_h{args.horizon}.npz"
    df, split = load_holdout()
    print(f"Holdout: train {df['open_time'].iloc[0]} -> {df['open_time'].iloc[split - 1]}, "
          f"test {df['open_time'].iloc[split]} -> {df['open_time'].iloc[-1]}")
    res = run_audit(df, split, args)
    res["rescored"] = flags["--i-know-this-rescores"] and os.path.exists(LOCK)
    print_table(res)

    os.makedirs("results", exist_ok=True)
    out = f"results/holdout_h{args.horizon}_leak_audit.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    lock = {"scored_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": git_commit(), "results": out, "seeds": args.seeds, "epochs": args.epochs,
            "horizon": args.horizon, "rescored": res["rescored"]}
    with open(LOCK, "w") as f:
        json.dump(lock, f, indent=2)
    print(f"\nSaved {out} and {LOCK}")


if __name__ == "__main__":
    main()
