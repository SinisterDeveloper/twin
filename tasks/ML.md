# ML Workflow — Track 2: "Multiverse of Monsoon" (`ML.md`)

**Owner:** Person A (ML Lead) — *with Person B running parallel `advect_resnet` experiments on A100 #2 during Hours 1–10 (see `SW.md`).*  
**Compute:** Google Colab Pro (High-RAM A100) via SSH / `colab/run_pipeline.ipynb` (keep heavy downloads & training off local Wi-Fi).  
**Repo:** `https://github.com/SinisterDeveloper/twin.git` (`main`)

---

## Sync Gates Overview (ML ↔ SW)

| Gate | Target Time | Shared Milestone | Blocking Condition |
| :--- | :--- | :--- | :--- |
| **Gate 0** | **Hour 0.5** | Colab A100s ready & smoke test green | `python -m twin.cli smoke-test` passes on both A100s |
| **Gate 1** | **Hour 2.0** | Data downloaded & `artifacts/` locked | `artifacts/clim_4harm.npz`, `norm_stats.npz`, `static.npy` built with `--train-end 2018-12-31` |
| **Gate 2** | **Hour 6.0** | Baselines trained & scored on Val (2019–2021) | Both `convnext_unet` and `advect_resnet` beat persistence baseline on 25-window Val |
| **Gate 3** | **Hour 10.0** | Hyperparameters frozen + Final 2005–2021 retrain | Best 2-family ensemble selected; ONNX weights exported & committed via Git LFS |
| **Gate 4** | **Hour 14.0** | `serve.py` live & contract-verified | `/health` + `/simulate` pass `monsoon_r1/schema_validator.py` & `< 5.0 s` latency |
| **Gate 5** | **Hour 18.0** | Full end-to-end freeze | Final ONNX weights + `serve.py` + `dashboard/` + 2-page PDF verified on clean clone |

---

## Phase 0 — Environment & Smoke Check (Hours 0.0 – 0.5)

- [ ] **0.1** Connect to Colab High-RAM A100 #1 (via SSH or `colab/run_pipeline.ipynb`).
- [ ] **0.2** Clone repo & pull LFS:
  ```bash
  git clone https://github.com/SinisterDeveloper/twin.git
  cd twin
  pip install -r requirements.txt
  ```
- [ ] **0.3** Run synthetic contract & model smoke test:
  ```bash
  python -m twin.cli smoke-test
  ```
- [ ] **Gate 0 Check:** Confirm `smoke-test` passes with zero errors and GPU is visible (`nvidia-smi`). Signal Person B (`SW.md`) to confirm Gate 0.

---

## Phase 1 — Data Ingestion & Strict Split Artifacts (Hours 0.5 – 2.0)

- [ ] **1.1** Download dataset inside Colab (never over local Wi-Fi):
  ```bash
  python -m twin.cli download-data --data-dir data
  ```
- [ ] **1.2** Verify `.nc` is gitignored (`git status` must NOT list `data/*.nc`; `monsoon_r1` anti-cheat penalizes committed `.nc` files).
- [ ] **1.3** Build strict temporal holdout artifacts (`train <= 2018-12-31`, `val = 2019-2021`):
  ```bash
  python -m twin.cli prepare-artifacts --nc data/india_climate_train.nc --artifacts-dir artifacts --train-end 2018-12-31
  ```
- [ ] **1.4** Inspect generated files in `artifacts/` (`clim_4harm.npz`, `norm_stats.npz`, `static.npy`) and verify shapes match `(32, 36)` and 6 dynamic + 2 static channels.
- [ ] **Gate 1 Check:** Commit & push `artifacts/` (if small `.npz`/`.npy`) or share exact checksums/build command with Person B so both A100s use **identical** `train-end 2018-12-31` normalization stats.

---

## Phase 2 — Baseline Training: Family A (`convnext_unet`) (Hours 2.0 – 6.0)

> While Person B runs Family B (`advect_resnet`) on A100 #2, Person A owns Family A (`convnext_unet`).

- [ ] **2.1** Train baseline `convnext_unet` (Stage 1 single-step + Stage 2 rollout curriculum `4,8,10`):
  ```bash
  python -m twin.cli train --model convnext_unet --epochs-s1 25 --epochs-s2 25 --rollout-schedule 4,8,10 --batch-size 64 --lr 1e-3 --seed 42
  ```
- [ ] **2.2** Evaluate on 2019–2021 Val split (25-window 5-regime + 3-shock scorer):
  ```bash
  python -m twin.cli eval --ckpts checkpoints/convnext_unet_s42/best.pt --val-start 2019-01-01 --out-json results/convnext_s42.json
  ```
- [ ] **2.3** Log all 4 rubric pillars from `results/convnext_s42.json`:
  - Forecast Skill (SS_T2M, SS_TP, CSI_10, sharpness/blur ratio `0.40–2.50`)
  - Shock Locality & Causality (`warm_ocean`, `monsoon_stall`, `null` tolerance `<= 1e-3`, plume speed `<= 8.0 cells/day`, `LOCAL_MIN_FRAC >= 0.60`)
  - 10-Day Stability (`STABLE_MAX_STD <= 5.0`)
- [ ] **Gate 2 Check (Hour 6.0):** Sync with Person B (`SW.md`) to compare `convnext_unet` vs `advect_resnet` Val JSONs. Pick top ablation levers (e.g., `--w-rain`, `--w-grad`, `--w-var`, `--pushforward-noise`, `--lr`).

---

## Phase 3 — Targeted Tuning & Diversity (Hours 6.0 – 10.0)

- [ ] **3.1** Run 2–3 targeted ablations on `convnext_unet` (keep `--val-start 2019-01-01` strictly untouched):
  - Sweep rain/gradient sharpness weights (`--w-rain`, `--w-grad`) if `blur_ratio` drifts below `0.40` or `CSI_10` lags.
  - Sweep `--pushforward-noise` (`0.02` vs `0.05`) if 10-day rollout diverges or `STABLE_MAX_STD` rises.
- [ ] **3.2** Collect Person B's best `advect_resnet` checkpoint & Val score.
- [ ] **3.3** Evaluate 2-family ensemble (`convnext_unet` + `advect_resnet`) on Val (2019–2021):
  ```bash
  python -m twin.cli eval --ckpts checkpoints/convnext_best.pt checkpoints/advect_best.pt --val-start 2019-01-01 --out-json results/ensemble_val.json
  ```
- [ ] **3.4** Freeze hyperparameters once ensemble beats single models on Val.
- [ ] **3.5** **Final Retrain (2005–2021):** Rebuild artifacts with full date range (only after hyperparameters are frozen!) and retrain both winning configs (`convnext_unet` + `advect_resnet`, optionally 2 seeds if time permits):
  ```bash
  python -m twin.cli prepare-artifacts --nc data/india_climate_train.nc --artifacts-dir artifacts --train-end 2021-12-31
  # Retrain winning convnext_unet and advect_resnet configs with frozen hyperparameters
  ```
- [ ] **3.6** Export ONNX models:
  ```bash
  python -m twin.cli export-onnx --ckpt checkpoints/convnext_final.pt --static artifacts/static.npy --out artifacts/convnext_unet.onnx
  python -m twin.cli export-onnx --ckpt checkpoints/advect_final.pt --static artifacts/static.npy --out artifacts/advect_resnet.onnx
  ```
- [ ] **Gate 3 Check (Hour 10.0):** Push final `artifacts/*.onnx` + `artifacts/*.npz` + `artifacts/static.npy` (via Git LFS) to `origin/main`. Hand off to Person B for Dashboard + validation integration.

---

## Phase 4 — Inference Server Optimization & Contract Hardening (Hours 10.0 – 14.0)

- [ ] **4.1** Verify `serve.py` loads the final ONNX ensemble (CPU-only `onnxruntime` path for Round-1 evaluator: `<= 6 GB RAM`, `<= 4 CPU cores`, `<= 5.0 s` per `/simulate` call).
- [ ] **4.2** Benchmark cold-start + `/simulate` latency across 8 consecutive 10-day rollout requests (including `warm_ocean`, `monsoon_stall`, and `null` shock perturbations).
- [ ] **4.3** Verify zero banned patterns (no `open-meteo`, `weatherbench2`, `cdsapi`, `imdlib`, no hardcoded dates, no `.nc` files in repo).
- [ ] **Gate 4 Check (Hour 14.0):** Run `uvicorn serve:app --port 8000` and let Person B run `monsoon_r1/schema_validator.py` + `monsoon_r1/client.py` against it. All checks must pass `< 5.0 s`.

---

## Phase 5 — Final Model Verification & Demo Prep (Hours 14.0 – 20.0)

- [ ] **5.1** Supply exact model architecture diagram, loss formulation (`log1p(tp)`, 4-harmonic climatology, residual tendency `z_{t+1} = z_t + delta`), and Val 2019–2021 table numbers to Person B for the **2-Page Project Brief PDF**.
- [ ] **5.2** Test `dashboard/` live against `serve.py` with Person B (verify shock sliders, null shock identity, and multi-day rollout playback).
- [ ] **5.3** Prepare 3 crisp technical answers for Desk Review:
  1. Why 4-harmonic climatology + residual tendency prevents seasonal drift.
  2. How bounded receptive field + `grid_sample` advection guarantees physical shock locality (`<= 8 cells/day`).
  3. Why strict 2019–2021 temporal holdout proves zero overfitting before the 2005–2021 final retrain.
- [ ] **Gate 5 Check (Hour 18.0):** Clean `git clone` test of `main` in a fresh directory (`serve.py` starts, `/health` & `/simulate` pass, `git lfs ls-files` shows ONNX weights).
