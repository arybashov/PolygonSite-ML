"""
Verify that py/sim.py and src/simulation.js produce identical initial
observations for the same seeds.

Run:  python test_sim_parity.py [--seeds 20] [--ndrones 5] [--nanti 5]
"""
import argparse
import json
import subprocess
import sys
import numpy as np
from py.sim import Simulation, DEFAULT_PARAMS

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--seeds',   type=int, default=20)
    p.add_argument('--ndrones', type=int, default=None)
    p.add_argument('--nanti',   type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    params = {**DEFAULT_PARAMS}
    if args.ndrones is not None: params['ndrones'] = args.ndrones
    if args.nanti   is not None: params['nanti']   = args.nanti

    # ── JS observations ──────────────────────────────────────────────────────
    cmd = ['node', 'scripts/dump_obs.mjs',
           '--seeds',   str(args.seeds),
           '--ndrones', str(int(params['ndrones'])),
           '--nanti',   str(int(params['nanti']))]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print('node error:', proc.stderr)
        sys.exit(1)
    js_obs = json.loads(proc.stdout)

    # ── Python observations ──────────────────────────────────────────────────
    sim = Simulation(params)
    py_obs = [sim.reset(seed).tolist() for seed in range(args.seeds)]

    # ── Compare ──────────────────────────────────────────────────────────────
    tol = 1e-5
    failures = []
    for seed in range(args.seeds):
        js = np.array(js_obs[seed], dtype=np.float32)
        py = np.array(py_obs[seed], dtype=np.float32)
        diff = np.abs(js - py)
        if diff.max() > tol:
            bad = np.where(diff > tol)[0]
            failures.append((seed, diff.max(), bad, js[bad], py[bad]))

    if not failures:
        print(f'OK  {args.seeds} seeds — max diff < {tol}')
    else:
        for seed, max_d, bad, js_v, py_v in failures:
            print(f'FAIL seed={seed}  max_diff={max_d:.2e}  indices={bad.tolist()}')
            for i, jv, pv in zip(bad, js_v, py_v):
                print(f'  [{i:3d}]  js={jv:.6f}  py={pv:.6f}  Δ={abs(jv-pv):.2e}')
        sys.exit(1)

if __name__ == '__main__':
    main()
