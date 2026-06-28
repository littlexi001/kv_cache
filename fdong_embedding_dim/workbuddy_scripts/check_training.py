#!/usr/bin/env python3
"""Training monitor: check progress of CRS and/or baseline runs from metrics.jsonl.

Usage:
  python3 workbuddy_scripts/check_training.py                          # check both runs (default)
  python3 workbuddy_scripts/check_training.py --run crs                # only CRS
  python3 workbuddy_scripts/check_training.py --run baseline           # only baseline
  python3 workbuddy_scripts/check_training.py --run baseline --watch   # live-tailing
  python3 workbuddy_scripts/check_training.py --plot                   # show combined loss curves
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


OUTPUT_DIR = os.path.expanduser(
    '/Users/bytedance/kv_cache/fdong_embedding_dim/outputs/small_lm_crs')
RUNS = {
    'crs': os.path.join(OUTPUT_DIR, 'crs_alpha0_3', 'metrics.jsonl'),
    'baseline': os.path.join(OUTPUT_DIR, 'baseline', 'metrics.jsonl'),
}


# ---------- Health check thresholds ----------
# These are reasonable ranges for a 109M LM on ~32k vocab in early training.
# Adjust after observing real data.

HEALTH_RULES = {
    'loss_range': (3.0, 12.0),           # expected loss range at step 0-5000
    'grad_norm_range': (0.01, 50.0),      # gradient norm should not explode or vanish
    'ppl_range': (20, 200000),            # perplexity sanity check
    'loss_divergence_factor': 0.95,        # if loss goes > 0.95 * initial_loss, may be diverging
    'min_improvement_pct': 2.0,            # expect at least 2% loss drop over last 500 steps
    'grad_norm_spike_factor': 5.0,         # if grad_norm spikes 5x vs recent avg, warn
}


def load_metrics(run_name):
    path = RUNS[run_name]
    if not os.path.exists(path):
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def diagnostic(records, label):
    """Print a diagnostic summary for a run."""
    if not records:
        print(f"\n  [{label}]  No metrics found yet. Training has not logged anything, "
              f"or still loading data / warming up.\n")
        return

    n = len(records)
    latest = records[-1]
    first = records[0]

    print(f"\n{'='*65}")
    print(f"  {label}  ({n} log entries)")
    print(f"{'='*65}")

    # ---- Latest status ----
    print(f"\n  ▸ Latest  (step {latest['step']:>6d}):")
    print(f"    loss      = {latest['loss']:.4f}")
    print(f"    ppl       = {latest['ppl']:.1f}")
    print(f"    grad_norm = {latest['grad_norm']:.3f}")
    print(f"    tok/s     = {latest.get('tok_per_sec', '?')}")
    elapsed = latest.get('elapsed_sec', 0)
    if elapsed > 0:
        h, m = divmod(elapsed, 3600)
        m, s = divmod(m, 60)
        print(f"    elapsed   = {int(h)}h {int(m)}m {s:.0f}s")
    if 'tokens' in latest:
        print(f"    tokens    = {latest['tokens']:,}")

    # ---- Progress ----
    print(f"\n  ▸ Progress:")
    print(f"    step range:  {first['step']} → {latest['step']}")
    print(f"    loss range:  {first['loss']:.4f} → {latest['loss']:.4f}  "
          f"({(1 - latest['loss']/first['loss'])*100:+.1f}%)")

    # ---- Trend (last few records) ----
    window = min(10, n)
    recent = records[-window:]
    avg_recent_loss = sum(r['loss'] for r in recent) / window
    avg_recent_gn = sum(r['grad_norm'] for r in recent) / window
    print(f"\n    last {window} windows avg: loss={avg_recent_loss:.4f}, grad_norm={avg_recent_gn:.3f}")

    # ---- Health checks ----
    print(f"\n  ▸ Health checks:")

    warnings = 0
    r = HEALTH_RULES

    # 1. Loss range
    if not (r['loss_range'][0] < latest['loss'] < r['loss_range'][1]):
        print(f"    ⚠️  loss={latest['loss']:.4f} outside expected [{r['loss_range'][0]}, {r['loss_range'][1]}]")
        warnings += 1
    else:
        print(f"    ✅ loss in range")

    # 2. Gradient norm
    if not (r['grad_norm_range'][0] < latest['grad_norm'] < r['grad_norm_range'][1]):
        print(f"    ⚠️  grad_norm={latest['grad_norm']:.3f} outside [{r['grad_norm_range'][0]}, {r['grad_norm_range'][1]}]")
        warnings += 1
    else:
        print(f"    ✅ grad_norm in range")

    # 3. Divergence check
    if latest['loss'] > first['loss'] * r['loss_divergence_factor']:
        print(f"    ⚠️  loss is {latest['loss']/first['loss']*100:.0f}% of initial → may be diverging")
        warnings += 1
    else:
        print(f"    ✅ loss trending down")

    # 4. Recent improvement
    if n >= 10:
        old_window = records[max(0, n-20):max(0, n-10)]
        if old_window:
            avg_old = sum(r['loss'] for r in old_window) / len(old_window)
            change_pct = (avg_old - avg_recent_loss) / avg_old * 100
            if change_pct < r['min_improvement_pct'] and avg_recent_loss > 1.0:
                print(f"    ℹ️  recent loss change {change_pct:+.1f}% over last ~10 windows (plateau?)")
            else:
                print(f"    ✅ loss continuing to improve ({change_pct:+.1f}% over last ~10 windows)")

    # 5. Gradient spike
    if n >= 5:
        recent_gns = [r['grad_norm'] for r in records[-5:]]
        avg_gn = sum(recent_gns[:-1]) / max(len(recent_gns)-1, 1)
        if recent_gns[-1] > avg_gn * r['grad_norm_spike_factor']:
            print(f"    ⚠️  grad_norm spike: latest={recent_gns[-1]:.3f} vs avg={avg_gn:.3f}")
            warnings += 1
        else:
            print(f"    ✅ no gradient spike")

    # ---- Summary ----
    if warnings == 0:
        print(f"\n  ✅ All health checks passed — training looks normal.")
    else:
        print(f"\n  ⚠️  {warnings} warning(s) — review the flagged items above.")


def plot_comparison(crs_records, base_records):
    """Print a side-by-side ASCII comparison table."""
    if not crs_records and not base_records:
        print("\n  No data to compare yet.\n")
        return

    print(f"\n{'='*80}")
    print(f"  Loss Comparison (CRS vs Baseline)")
    print(f"{'='*80}")

    # Build a step-aligned comparison
    all_steps = set()
    crs_by_step = {}
    base_by_step = {}

    for r in (crs_records or []):
        s = r['step']
        all_steps.add(s)
        crs_by_step[s] = r

    for r in (base_records or []):
        s = r['step']
        all_steps.add(s)
        base_by_step[s] = r

    if not all_steps:
        print("  No data yet.\n")
        return

    steps = sorted(all_steps)
    # Show every Nth step
    show_every = max(1, len(steps) // 15)
    header = f"  {'Step':>8s}  {'CRS loss':>10s}  {'CRS ppl':>8s}  {'Base loss':>10s}  {'Base ppl':>8s}"
    print(header)
    print(f"  {'-'*(len(header)-4)}")

    for s in steps[::show_every]:
        crs = crs_by_step.get(s, {})
        base = base_by_step.get(s, {})
        crs_loss = f"{crs.get('loss', float('nan')):.4f}" if crs else '-'
        crs_ppl = f"{crs.get('ppl', float('nan')):.0f}" if crs else '-'
        base_loss = f"{base.get('loss', float('nan')):.4f}" if base else '-'
        base_ppl = f"{base.get('ppl', float('nan')):.0f}" if base else '-'
        print(f"  {s:>8d}  {crs_loss:>10s}  {crs_ppl:>8s}  {base_loss:>10s}  {base_ppl:>8s}")

    # Also show final status
    print(f"\n  ▸ Final comparison:")
    if crs_records:
        print(f"    CRS final loss  = {crs_records[-1]['loss']:.4f}  "
              f"(ppl={crs_records[-1]['ppl']:.0f})")
    if base_records:
        print(f"    Baseline final  = {base_records[-1]['loss']:.4f}  "
              f"(ppl={base_records[-1]['ppl']:.0f})")

    if crs_records and base_records:
        delta = base_records[-1]['loss'] - crs_records[-1]['loss']
        print(f"    Δ (base - CRS)  = {delta:+.4f}")


def watch_tail(run_name, interval=5):
    """Live-tailing of metrics file."""
    path = RUNS[run_name]
    print(f"\n  Watching {path} (refresh every {interval}s). Ctrl+C to stop.\n")
    last_size = os.path.getsize(path) if os.path.exists(path) else 0

    try:
        while True:
            if os.path.exists(path):
                size = os.path.getsize(path)
                if size != last_size:
                    # Read new content
                    with open(path) as f:
                        f.seek(last_size)
                        new_lines = f.read().strip().split('\n')
                    for line in new_lines:
                        if line.strip():
                            r = json.loads(line.strip())
                            print(f"  [{run_name}] step={r['step']:<6d}  "
                                  f"loss={r['loss']:.4f}  ppl={r['ppl']:.0f}  "
                                  f"grad_norm={r['grad_norm']:.3f}")
                    last_size = size
            else:
                print(f"  Waiting for {path}...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n  Stopped watching.")


def main():
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument('--run', type=str, default=None,
                        choices=['crs', 'baseline'],
                        help='Which run to check (default: both)')
    parser.add_argument('--watch', action='store_true',
                        help='Live-tail metrics (requires --run)')
    parser.add_argument('--plot', action='store_true',
                        help='Show loss comparison table')
    parser.add_argument('--interval', type=int, default=5,
                        help='Seconds between checks in watch mode')
    args = parser.parse_args()

    if args.watch:
        if not args.run:
            print("Error: --watch requires --run (crs or baseline)")
            sys.exit(1)
        watch_tail(args.run, args.interval)
        return

    if args.plot:
        crs = load_metrics('crs')
        base = load_metrics('baseline')
        plot_comparison(crs, base)
        return

    runs_to_check = [args.run] if args.run else ['crs', 'baseline']
    for run_name in runs_to_check:
        records = load_metrics(run_name)
        diagnostic(records, run_name.upper())


if __name__ == '__main__':
    main()
