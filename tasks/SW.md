# Software, Validation & Product Workflow (`SW.md`)

**Owner:** Person B (SW / Product Lead) — *supporting parallel `advect_resnet` training on A100 #2 in Hours 1–10, then owning Verification Harness, React + Vite Digital Twin Dashboard (`dashboard/`), and 2-Page Project Brief PDF.*  
**Repo:** `https://github.com/SinisterDeveloper/twin.git` (`main`)

---

## Sync Gates Overview (SW ↔ ML)

| Gate | Target Time | Shared Milestone | Blocking Condition |
| :--- | :--- | :--- | :--- |
| **Gate 0** | **Hour 0.5** | Colab A100s ready & smoke test green | `python -m twin.cli smoke-test` passes on both A100s |
| **Gate 1** | **Hour 2.0** | Data downloaded & `artifacts/` locked | Identical `train-end 2018-12-31` artifacts on both A100s |
| **Gate 2** | **Hour 6.0** | Baselines trained & scored on Val (2019–2021) | Both `convnext_unet` and `advect_resnet` beat persistence baseline |
| **Gate 3** | **Hour 10.0** | Hyperparameters frozen + Final ONNX weights in LFS | `artifacts/*.onnx` pushed to `main`; Person B pivots 100% to SW/UI/PDF |
| **Gate 4** | **Hour 14.0** | `serve.py` live & contract-verified + UI scaffolded | `schema_validator.py` passes `< 5.0 s`; `dashboard/` renders live `/simulate` |
| **Gate 5** | **Hour 18.0** | Full end-to-end freeze | Clean clone passes Round-1 checks, `dashboard/` demo works, 2-Page PDF exported |

---

## Phase 0 — Environment & Parallel A100 Setup (Hours 0.0 – 0.5)

- [ ] **0.1** Connect to Colab High-RAM A100 #2.
- [ ] **0.2** Clone repo & install dependencies:
  ```bash
  git clone https://github.com/SinisterDeveloper/twin.git
  cd twin
  pip install -r requirements.txt
  ```
- [ ] **0.3** Run `python -m twin.cli smoke-test` and verify `git lfs install` is active.
- [ ] **Gate 0 Check:** Confirm `smoke-test` passes and sync with Person A (`ML.md`).

---

## Phase 1 — Data Sync & Verification Harness Prep (Hours 0.5 – 2.0)

- [ ] **1.1** Download dataset on Colab A100 #2 (`python -m twin.cli download-data --data-dir data`).
- [ ] **1.2** Wait for Person A's `prepare-artifacts` verification or run the identical command:
  ```bash
  python -m twin.cli prepare-artifacts --nc data/india_climate_train.nc --artifacts-dir artifacts --train-end 2018-12-31
  ```
- [ ] **1.3** Verify `monsoon_r1/schema_validator.py` and `monsoon_r1/client.py` work locally/on Colab against a dummy `uvicorn serve:app --port 8000` instance so the test harness is ready early.
- [ ] **Gate 1 Check:** Confirm identical `artifacts/` stats with Person A before starting training.

---

## Phase 2 — Parallel Family B (`advect_resnet`) Training (Hours 2.0 – 6.0)

> Person B runs Family B (`advect_resnet` — differentiable `grid_sample` wind advection + local FiLM ResBlocks) on A100 #2 in parallel with Person A's `convnext_unet`.

- [ ] **2.1** Train baseline `advect_resnet`:
  ```bash
  python -m twin.cli train --model advect_resnet --epochs-s1 25 --epochs-s2 25 --rollout-schedule 4,8,10 --batch-size 64 --lr 1e-3 --seed 42
  ```
- [ ] **2.2** Evaluate on 2019–2021 Val split:
  ```bash
  python -m twin.cli eval --ckpts checkpoints/advect_resnet_s42/best.pt --val-start 2019-01-01 --out-json results/advect_s42.json
  ```
- [ ] **Gate 2 Check (Hour 6.0):** Share `results/advect_s42.json` with Person A. Compare `SS_T2M`, `SS_TP`, `CSI_10`, `blur_ratio`, and shock locality against `convnext_unet`.

---

## Phase 3 — `advect_resnet` Ablations & Handoff (Hours 6.0 – 10.0)

- [ ] **3.1** Run 2 targeted ablations on `advect_resnet` (e.g., `--lr 5e-4`, `--w-var 0.08`, or `--rollout-schedule 4,8,12`) on Val 2019–2021.
- [ ] **3.2** Send best `advect_resnet` checkpoint + config to Person A for ensemble evaluation and final 2005–2021 retrain.
- [ ] **Gate 3 Check (Hour 10.0):** Pull Person A's frozen `artifacts/*.onnx` + `artifacts/*.npz` via `git pull`. **Person B now pivots 100% to SW, Dashboard, and PDF Brief.**

---

## Phase 4 — Contract Verification & React Digital Twin Dashboard (Hours 10.0 – 14.0)

> **Strict Separation Rule:** `dashboard/` is a standalone frontend under `dashboard/` calling `serve.py` via HTTP (`POST /simulate`, `GET /health`). Never couple UI code into `serve.py` or add UI dependencies to root `requirements.txt`.

- [ ] **4.1** Start `serve.py` (`uvicorn serve:app --host 0.0.0.0 --port 8000`) and run automated Round-1 verification:
  - `python -m monsoon_r1.schema_validator --url http://localhost:8000`
  - Latency test: 8 sequential `/simulate` calls (`10`-day rollouts), all `< 5.0 s`, RAM `< 6 GB`.
  - Anti-cheat regex scan: confirm zero `open-meteo|openweathermap|weatherbench2|arco-era5|cdsapi|imdlib` matches and zero `.nc` files tracked by git.
- [ ] **4.2** Scaffold `dashboard/` (React + Vite + TypeScript / Tailwind) with:
  - **India 32×36 Interactive Grid Map** (canvas/SVG heatmap over `lat`/`lon` with coastline/orography overlay from `static.npy`).
  - **Variable Selector:** `t2m`, `tp`, `u10`, `v10`, `msl`, `tcwv` + wind vector barbs/arrows (`u10`, `v10`).
  - **Day Scrubber (Day 0 → Day 10):** Play/pause rollout animation.
  - **Shock Builder Panel:** Presets (`warm_ocean`, `monsoon_stall`, `null`) + custom lat/lon/radius/intensity controls, showing **Baseline vs. Shocked vs. Delta (`Shocked - Baseline`)** side-by-side!
  - **Telemetry & Physics Guardrails Bar:** Live `/simulate` latency (ms), max anomaly std, shock plume spread check.
- [ ] **Gate 4 Check (Hour 14.0):** Both `/simulate` contract validation AND `dashboard/` live playback verified together with Person A.

---

## Phase 5 — 2-Page Project Brief PDF & Final Submission Freeze (Hours 14.0 – 20.0)

- [ ] **5.1** Draft **2-Page Project Brief (PDF, max 2 pages)** using Person A's final Val 2019–2021 metrics:
  - **USP:** Physics-aware 2-family ensemble (`ConvNeXt-UNet` + `Advect-ResNet` with differentiable `grid_sample` wind advection) on 4-harmonic climatology + `log1p(tp)` residual dynamics.
  - **Architecture Diagram:** Clear schematic of sensory encoding, causal local receptive fields, and ONNX CPU inference path.
  - **Validation & Anti-Overfitting Proof:** 2019–2021 temporal holdout table + 3-shock causality metrics (`null <= 1e-3`, plume speed `<= 8 cells/day`).
  - **Product Demo:** Screenshot of `dashboard/` showing `warm_ocean` / `monsoon_stall` delta rollout.
- [ ] **5.2** Export PDF and verify **exact page count <= 2 pages**.
- [ ] **5.3** Final Repo Hygiene & Clean-Clone Verification:
  - `git status` clean, no `.nc` files, no `checkpoints/` bloat > 1 GB.
  - `git lfs ls-files` lists final `.onnx` weights.
  - Fresh `git clone` in a clean folder + `uvicorn serve:app` + `schema_validator.py` passes 100%.
- [ ] **Gate 5 Check (Hour 18.0 – 19.0):** Final sign-off between Person A & Person B. Submit **Public GitHub Repo URL** + **2-Page PDF** before Hour 20.0.
