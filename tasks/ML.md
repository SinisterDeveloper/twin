# ML & Inference Workflow — Track 2: "Multiverse of Monsoon" (`ML.md`)

**Owner:** Person A (ML & Inference Lead) — *owns all data prep, pre-flight tests, training (`convnext_unet`, `advect_resnet`, `local_swin`, `pde_net`), Val tuning, simplex ensemble, ONNX export, and `serve.py` inference performance.*  
**Compute:** Run all steps sequentially on **your primary Colab Pro High-RAM A100** (~10–12h GPU total out of 100h quota; use the second teammate's A100 only as optional overflow). Never run heavy dataset downloads over local Wi-Fi.  
**Repo:** `https://github.com/SinisterDeveloper/twin.git` (`main`)

---

## Sync Gates Overview (ML ↔ SW)

| Gate | Target Time | Shared Milestone | Blocking Condition |
| :--- | :--- | :--- | :--- |
| **Gate 0** | **Hour 0.5** | Colab ready & pre-flight suite green | `pytest -q tests/test_ml_pipeline.py` passes (advection, Day-1 locality, ONNX parity, `< 500 ms` CPU speed) |
| **Gate 1** | **Hour 1.5** | Data downloaded & `artifacts/` locked | `artifacts/clim.npy`, `raw_clim.npy`, `clim_uv_coef.npy`, `norm_stats.json`, `static.npy` built (`--train-end 2018-12-31`) |
| **Gate 2** | **Hour 5.5** | All 4 families trained & scored on Val | `convnext_unet`, `advect_resnet`, `local_swin`, `pde_net` beat persistence/clim on 25-window Val (`2019–2021`) + 10-window causality |
| **Gate 3** | **Hour 10.5** | Generalization check + Final 2005–2021 retrain | Secondary fold (`2016–2018`) passes, simplex weights frozen, final 4-model ONNX + `ensemble_weights.json` pushed via Git LFS |
| **Gate 4** | **Hour 14.0** | `serve.py` live & contract-verified | `/health` + `/simulate` pass SW's `schema_validator.py`, `< 0.5 s` CPU latency, & `dashboard/` live integration |
| **Gate 5** | **Hour 18.0** | Full end-to-end freeze | Clean clone passes Round-1 checks, `dashboard/` demo works, 2-Page PDF exported |

---

## Phase 0 — Colab A100 Setup & Pre-Flight Verification (Hours 0.0 – 0.5)

- [ ] **0.1** Connect to your primary **Colab High-RAM A100** (SSH or `colab/run_pipeline.ipynb`).
- [ ] **0.2** Clone repo & install dependencies:
  ```bash
  git clone https://github.com/SinisterDeveloper/twin.git
  cd twin
  pip install -r requirements.txt pytest
  ```
- [ ] **0.3** Run the full zero-data pre-flight test suite (DO NOT train if any check fails!):
  ```bash
  pytest -q tests/test_ml_pipeline.py
  python -m twin.cli smoke-test
  ```
- [ ] **Gate 0 Check:** Confirm `nvidia-smi` shows A100 and all 6 pre-flight tests (`doy_sincos`, scorer parity, RK2 physical wind direction, Day-1 locality `>= 0.60`, ONNX parity & `< 500 ms` CPU speed, anti-cheat) pass. Signal Person B (`SW.md`).

---

## Phase 1 — Data Ingestion & Strict Split Artifacts (Hours 0.5 – 1.5)

- [ ] **1.1** Download dataset inside Colab (never over local Wi-Fi):
  ```bash
  python -m twin.cli download-data --data-dir data
  ```
- [ ] **1.2** Verify `.nc` is gitignored (`git status` must NOT list `data/*.nc`; `monsoon_r1` anti-cheat penalizes committed `.nc` files).
- [ ] **1.3** Build strict temporal holdout artifacts (`train <= 2018-12-31`, `val = 2019–2021`):
  ```bash
  python -m twin.cli prepare-artifacts --nc data/india_climate_train.nc --artifacts-dir artifacts --train-end 2018-12-31
  ```
- [ ] **1.4** Verify `artifacts/` contains `static.npy` (`6, 32, 36`), `clim.npy`, `raw_clim.npy`, `clim_uv_coef.npy` (`9, 2, 32, 36`), and `norm_stats.json`.
- [ ] **Gate 1 Check:** Commit and push `artifacts/` (excluding `.onnx` for now) to `origin/main` so SW (`dashboard/` coastline/orography & mock/smoke tests) has exact `static.npy` and normalization metadata.

---

## Phase 2 — Train All 4 Model Families on Primary A100 (Hours 1.5 – 5.5)

> With GPU-resident tensors (`GPURolloutBatcher`), the 171 MB dataset lives in A100 VRAM; each 3-stage EMA run finishes rapidly on one A100.

- [ ] **2.1** Train Family A (`convnext_unet` v2 — bounded RF ±7 cells):
  ```bash
  python -m twin.cli train --model convnext_unet --out-dir checkpoints/convnext_s42 --seed 42
  ```
- [ ] **2.2** Train Family B (`advect_resnet` v2 — RK2 physical wind semi-Lagrangian advection):
  ```bash
  python -m twin.cli train --model advect_resnet --out-dir checkpoints/advect_s42 --seed 42
  ```
- [ ] **2.3** Train Family C (`local_swin` — shifted 4×4 window local WeatherFormer):
  ```bash
  python -m twin.cli train --model local_swin --out-dir checkpoints/swin_s42 --seed 42
  ```
- [ ] **2.4** Train Family D (`pde_net` — explicit finite-difference $\nabla, \nabla^2, -\mathbf{u}_{\text{phys}}\cdot\nabla$ operator net):
  ```bash
  python -m twin.cli train --model pde_net --out-dir checkpoints/pde_s42 --seed 42
  ```
- [ ] **2.5** Score every individual model on 25-window Val (`2019–2021`) + 10-window causality:
  ```bash
  for m in convnext advect swin pde; do
    python -m twin.cli eval --ckpts checkpoints/${m}_s42/*_best.pt --val-start 2019-01-01 --out-json results/${m}_s42.json
  done
  ```
- [ ] **Gate 2 Check:** Confirm all 4 families achieve `all_shocks_pass=true`, `anti_blur_pts == 10.0`, and clear `SS > 0.12` / `CSI SS > 0.10`.

---

## Phase 3 — Anti-Overfitting Audit, Tuning, Ensemble & Final Retrain (Hours 5.5 – 10.5)

- [ ] **3.1** **Secondary Temporal Fold Audit (anti-overfitting proof):**
  ```bash
  python -m twin.cli prepare-artifacts --nc data/india_climate_train.nc --artifacts-dir artifacts_fold15 --train-end 2015-12-31
  python -m twin.cli train --artifacts-dir artifacts_fold15 --out-dir checkpoints/fold15_convnext --model convnext_unet --train-end 2015-12-31 --val-end 2018-12-31 --seed 42
  python -m twin.cli eval --artifacts-dir artifacts_fold15 --ckpts checkpoints/fold15_convnext/convnext_unet_best.pt --val-start 2016-01-01 --val-end 2018-12-31 --out-json results/fold15_val.json
  ```
- [ ] **3.2** Restore `2018-12-31` Val artifacts and optimize per-channel simplex ensemble weights strictly on Val (`2019–2021`), plus run targeted `--w-csi` / `--w-var` sweeps only if any channel lags:
  ```bash
  python -m twin.cli prepare-artifacts --nc data/india_climate_train.nc --artifacts-dir artifacts --train-end 2018-12-31
  python -m twin.cli fit-ensemble --ckpts checkpoints/convnext_s42/convnext_unet_best.pt checkpoints/advect_s42/advect_resnet_best.pt checkpoints/swin_s42/local_swin_best.pt checkpoints/pde_s42/pde_net_best.pt --val-start 2019-01-01 --out artifacts/ensemble_weights.json
  python -m twin.cli eval --ckpts checkpoints/convnext_s42/convnext_unet_best.pt checkpoints/advect_s42/advect_resnet_best.pt checkpoints/swin_s42/local_swin_best.pt checkpoints/pde_s42/pde_net_best.pt --weights-json artifacts/ensemble_weights.json --val-start 2019-01-01 --out-json results/ensemble_val.json
  ```
- [ ] **3.3** Train second seed (`seed=137`) on top 2 performing families if time/quota permits, and freeze all hyperparameters.
- [ ] **3.4** **Final Retrain (`2005–2021`):** Rebuild artifacts across all available years (`--train-end 2021-12-31`) only AFTER hyperparameters are frozen, then retrain all 4 families:
  ```bash
  python -m twin.cli prepare-artifacts --nc data/india_climate_train.nc --artifacts-dir artifacts --train-end 2021-12-31
  for m in convnext_unet advect_resnet local_swin pde_net; do
    python -m twin.cli train --model $m --train-end 2021-12-31 --val-end 2021-12-31 --out-dir checkpoints/final_${m} --seed 42
  done
  ```
- [ ] **3.5** Export all 4 ONNX models & check CPU latency (`< 500 ms` total ensemble):
  ```bash
  python -m twin.cli export-onnx --ckpt checkpoints/final_convnext_unet/convnext_unet_best.pt --static artifacts/static.npy --out artifacts/convnext_unet.onnx
  python -m twin.cli export-onnx --ckpt checkpoints/final_advect_resnet/advect_resnet_best.pt --static artifacts/static.npy --out artifacts/advect_resnet.onnx
  python -m twin.cli export-onnx --ckpt checkpoints/final_local_swin/local_swin_best.pt --static artifacts/static.npy --out artifacts/local_swin.onnx
  python -m twin.cli export-onnx --ckpt checkpoints/final_pde_net/pde_net_best.pt --static artifacts/static.npy --out artifacts/pde_net.onnx
  ```
- [ ] **Gate 3 Check:** Push `artifacts/*.onnx`, `artifacts/ensemble_weights.json`, `artifacts/clim*.npy`, `artifacts/norm_stats.json`, `artifacts/static.npy` (via Git LFS), and `results/ensemble_val.json` to `origin/main`.

---

## Phase 4 — Inference Server (`serve.py`) Optimization (Hours 10.5 – 14.0)

- [ ] **4.1** Verify `serve.py` loads all 4 ONNX models + `ensemble_weights.json` via CPU `onnxruntime` (`<= 6 GB RAM`, `<= 4 CPU cores`, `< 5.0 s`).
- [ ] **4.2** Start server & run 8-call automated stress verification:
  ```bash
  uvicorn serve:app --port 8000 &
  python -m twin.cli verify --url http://localhost:8000 --calls 8 --days 10 --max-latency 5.0
  ```
- [ ] **Gate 4 Check:** Pass Person B's automated `schema_validator.py`, latency (`< 1.0 s`), and `dashboard/` live integration tests.

---

## Phase 5 — Final Model Sign-off & Technical Defense Prep (Hours 14.0 – 20.0)

- [ ] **5.1** Provide Person B with exact final Val (`2019–2021`) + Fold-15 (`2016–2018`) metrics tables, `MonsoonLoss v2` (Soft-CSI + physical Day 5–10 variance), and 4-family architecture summary for the **2-Page Project Brief PDF**.
- [ ] **5.2** Co-verify `dashboard/` live shock behavior (`warm_ocean`, `monsoon_stall`, `null`) against `serve.py`.
- [ ] **5.3** Desk Review defense points:
  1. Why 4-harmonic climatology + exact 0-indexed `doy_last` + residual tendency eliminates seasonal drift.
  2. How bounded RF (`<= ±7 cells`), physical-wind RK2 advection, replicate-padded `LocalSwin`, and `LocalPDENet` mathematically guarantee Day-1 locality (`near_frac_day1 >= 0.60`) and plume propagation (`<= 8 cells/day`).
  3. How dual temporal holdout (`2016–2018` + `2019–2021`) proves zero overfitting prior to `2005–2021` final retrain.
- [ ] **Gate 5 Check:** Joint sign-off with Person B on clean `git clone` of `main` before Hour 20.0 submission.
