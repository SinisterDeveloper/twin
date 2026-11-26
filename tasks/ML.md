# ML & Inference Workflow — Track 2: "Multiverse of Monsoon" (`ML.md`)

**Owner:** Person A (ML & Inference Lead) — *owns all data prep, model training (`convnext_unet` + `advect_resnet`), Val tuning, ensemble, ONNX export, and `serve.py` inference performance.*  
**Compute:** Google Colab Pro (High-RAM A100s) via SSH / `colab/run_pipeline.ipynb` (keep heavy downloads & training off local Wi-Fi).  
**Repo:** `https://github.com/SinisterDeveloper/twin.git` (`main`)

---

## Sync Gates Overview (ML ↔ SW)

| Gate | Target Time | Shared Milestone | Blocking Condition |
| :--- | :--- | :--- | :--- |
| **Gate 0** | **Hour 0.5** | Environments ready & smoke test green | `python -m twin.cli smoke-test` passes on Colab A100; SW test harness ready |
| **Gate 1** | **Hour 2.0** | Data downloaded & `artifacts/` locked | `artifacts/clim_4harm.npz`, `norm_stats.npz`, `static.npy` built (`--train-end 2018-12-31`) |
| **Gate 2** | **Hour 6.0** | Both model baselines trained & scored on Val | `convnext_unet` + `advect_resnet` beat persistence baseline on 25-window Val (`2019–2021`) |
| **Gate 3** | **Hour 10.0** | Hyperparameters frozen + Final 2005–2021 retrain | Best 2-family ensemble ONNX weights exported & pushed via Git LFS for SW integration |
| **Gate 4** | **Hour 14.0** | `serve.py` live & contract-verified | `/health` + `/simulate` pass SW's `schema_validator.py` & `< 5.0 s` latency; `dashboard/` live |
| **Gate 5** | **Hour 18.0** | Full end-to-end freeze | Clean clone passes Round-1 checks, `dashboard/` demo works, 2-Page PDF exported |

---

## Phase 0 — Colab A100 Setup & Smoke Check (Hours 0.0 – 0.5)

- [ ] **0.1** Connect to Colab High-RAM A100(s) (via SSH or `colab/run_pipeline.ipynb`).
- [ ] **0.2** Clone repo & install dependencies:
  ```bash
  git clone https://github.com/SinisterDeveloper/twin.git
  cd twin
  pip install -r requirements.txt
  ```
- [ ] **0.3** Run synthetic contract & model smoke test:
  ```bash
  python -m twin.cli smoke-test
  ```
- [ ] **Gate 0 Check:** Confirm `smoke-test` passes with zero errors and GPU is visible (`nvidia-smi`). Signal Person B (`SW.md`).

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
- [ ] **1.4** Verify generated shapes in `artifacts/` (`clim_4harm.npz`, `norm_stats.npz`, `static.npy`) match `(32, 36)` and 6 dynamic + 2 static channels.
- [ ] **Gate 1 Check:** Push `artifacts/` (small `.npz`/`.npy`) to `origin/main` so SW (`dashboard/` coastline/orography & mock/smoke tests) has exact `static.npy` and normalization metadata.

---

## Phase 2 — Train Both Model Family Baselines (Hours 2.0 – 6.0)

> Run `convnext_unet` and `advect_resnet` in parallel across two Colab A100 tabs (or sequentially on one A100).

- [ ] **2.1** Train Family A (`convnext_unet` — 2-scale ConvNeXt-UNet with GroupNorm & FiLM `doy_sincos`):
  ```bash
  python -m twin.cli train --model convnext_unet --epochs-s1 25 --epochs-s2 25 --rollout-schedule 4,8,10 --batch-size 64 --lr 1e-3 --seed 42
  ```
- [ ] **2.2** Train Family B (`advect_resnet` — differentiable `grid_sample` wind advection + local FiLM ResBlocks):
  ```bash
  python -m twin.cli train --model advect_resnet --epochs-s1 25 --epochs-s2 25 --rollout-schedule 4,8,10 --batch-size 64 --lr 1e-3 --seed 42
  ```
- [ ] **2.3** Evaluate both baselines on 2019–2021 Val split (25-window 5-regime + 3-shock scorer):
  ```bash
  python -m twin.cli eval --ckpts checkpoints/convnext_unet_s42/best.pt --val-start 2019-01-01 --out-json results/convnext_s42.json
  python -m twin.cli eval --ckpts checkpoints/advect_resnet_s42/best.pt --val-start 2019-01-01 --out-json results/advect_s42.json
  ```
- [ ] **2.4** Log all 4 rubric pillars from both JSONs:
  - Forecast Skill (`SS_T2M`, `SS_TP`, `CSI_10`, `blur_ratio` in `0.40–2.50`)
  - Shock Locality & Causality (`warm_ocean`, `monsoon_stall`, `null <= 1e-3`, plume speed `<= 8.0 cells/day`, `LOCAL_MIN_FRAC >= 0.60`)
  - 10-Day Stability (`STABLE_MAX_STD <= 5.0`)
- [ ] **Gate 2 Check (Hour 6.0):** Confirm both baselines beat persistence on Val. Share baseline Val numbers with Person B for early brief drafting.

---

## Phase 3 — Tuning, Ensemble, Final Retrain & ONNX Export (Hours 6.0 – 10.0)

- [ ] **3.1** Run targeted ablations on Val (`2019–2021` strictly untouched):
  - Sweep rain/gradient sharpness weights (`--w-rain`, `--w-grad`) if `blur_ratio < 0.40` or `CSI_10` lags.
  - Sweep `--pushforward-noise` (`0.02` vs `0.05`) or `--w-var` if 10-day rollout diverges or `STABLE_MAX_STD` rises.
- [ ] **3.2** Evaluate 2-family ensemble (`convnext_unet` + `advect_resnet`) on Val (2019–2021):
  ```bash
  python -m twin.cli eval --ckpts checkpoints/convnext_best.pt checkpoints/advect_best.pt --val-start 2019-01-01 --out-json results/ensemble_val.json
  ```
- [ ] **3.3** Freeze hyperparameters once ensemble beats single models on Val.
- [ ] **3.4** **Final Retrain (2005–2021):** Rebuild artifacts with full date range (only after hyperparameters are frozen!) and retrain both winning configs:
  ```bash
  python -m twin.cli prepare-artifacts --nc data/india_climate_train.nc --artifacts-dir artifacts --train-end 2021-12-31
  # Retrain winning convnext_unet and advect_resnet configs with frozen hyperparameters
  ```
- [ ] **3.5** Export ONNX models:
  ```bash
  python -m twin.cli export-onnx --ckpt checkpoints/convnext_final.pt --static artifacts/static.npy --out artifacts/convnext_unet.onnx
  python -m twin.cli export-onnx --ckpt checkpoints/advect_final.pt --static artifacts/static.npy --out artifacts/advect_resnet.onnx
  ```
- [ ] **Gate 3 Check (Hour 10.0):** Push final `artifacts/*.onnx` + `artifacts/*.npz` + `artifacts/static.npy` (via Git LFS) and `results/ensemble_val.json` to `origin/main`. Hand off to Person B for SW integration & dashboard wiring.

---

## Phase 4 — Inference Server (`serve.py`) Optimization (Hours 10.0 – 14.0)

- [ ] **4.1** Verify `serve.py` loads the final ONNX ensemble via CPU `onnxruntime` (Round-1 limits: `<= 6 GB RAM`, `<= 4 CPU cores`, `<= 5.0 s` per `/simulate` call).
- [ ] **4.2** Profile cold-start + `/simulate` latency across 8 consecutive 10-day rollout requests (`warm_ocean`, `monsoon_stall`, `null`). Optimize thread settings / batching if latency approaches `5.0 s`.
- [ ] **4.3** Confirm `serve.py` has zero external weather calls, zero hardcoded dates, and clean `/health` + `/simulate` responses.
- [ ] **Gate 4 Check (Hour 14.0):** Run `uvicorn serve:app --port 8000` and pass Person B's automated `schema_validator.py`, latency, and `dashboard/` live integration tests.

---

## Phase 5 — Final Model Sign-off & Technical Defense Prep (Hours 14.0 – 20.0)

- [ ] **5.1** Provide Person B with exact final Val (`2019–2021`) metrics table, loss formulation (`log1p(tp)`, 4-harmonic climatology, residual tendency `z_{t+1} = z_t + delta`), and architecture summary for the **2-Page Project Brief PDF**.
- [ ] **5.2** Co-verify `dashboard/` live shock behavior (`warm_ocean`, `monsoon_stall`, `null`) against `serve.py` for physical sanity.
- [ ] **5.3** Prepare 3 crisp technical answers for Desk Review:
  1. Why 4-harmonic climatology + residual tendency prevents seasonal drift.
  2. How bounded receptive field + `grid_sample` advection guarantees physical shock locality (`<= 8 cells/day`).
  3. Why strict 2019–2021 temporal holdout proves zero overfitting before the 2005–2021 final retrain.
- [ ] **Gate 5 Check (Hour 18.0):** Joint sign-off with Person B on clean `git clone` of `main` before Hour 20.0 submission.
