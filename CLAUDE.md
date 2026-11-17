# Polar Codes: MEng project

My MEng project on polar codes: polar transform, channel polarization, frozen-bit
selection and successive-cancellation (SC) decoding, in Python/numpy.
Git repo, mirrored to https://github.com/jadenzou1-dotcom/polar-codes-higher-alphabet.
This folder is the single source of truth (replaced the old `Polar proj` folder on
2026-11-16). Work happens on the Mac only; GitHub is just the published copy.

## Workflow
- Commit and push straight to `main` when asked ("push the Polar changes").
  No branches or PRs: solo project, the repo should simply mirror this folder.

## Layout
- `baseline/`: BEC and BSC (`polar_bec.py`, `polar_BSC.py`, `graph_BEC.py`, `graph_BSC.py`).
- `higher_alphabet/`: q-ary extension.
  - `alphabet_bsc_fast.py`: mod-q transform, q-ary channel, q-ary SC decode.
  - `graph_alphabet.py`: BLER vs K for q=5 (~25 min).
  - `plot_normalized_capacity.py`: normalized subchannel capacity for q = 2–10, 15,
    20, 25, 50 plus the **speed of polarization** measure: area between the
    sorted capacity curve and the ideal step at capacity
    (`area_from_theoretical_step()`). Batched + multiprocess (~4 min on 14 cores);
    batched functions verified against the original per-trial
    `estimate_subchannel_prob_vector` to ~1e-16.
  - `capacity_cache/`: committed .npy capacities so the plot reloads instantly.
  - `results/`: committed figures, CSV/MD tables, logs.
- `jz2396_MEng_Project_Report.pdf`, `jadenpolarposter.pdf`: final report and poster
  (cover q ∈ {2,3,5,7} only).

## Status (as of 2026-11-16)
- Report and poster finalized July 2026.
- Results beyond the report (N=512, p=0.1, 500 trials): area/N falls from 0.066
  (q=2) to 0.035 (q=50), ~47% smaller, monotone in q including non-prime q.
- `np.trapezoid` needs numpy >= 2.0 (pinned in requirements.txt).
