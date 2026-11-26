# Software, Validation & Product Workflow (`SW.md`)

**Owner:** Person B (Software & Product Lead) — *100% software engineering: Round-1 contract/latency/anti-cheat verification harness, React + Vite Digital Twin Dashboard (`dashboard/`), repo/Git LFS hygiene, and 2-Page Project Brief PDF.*  
**Repo:** `https://github.com/SinisterDeveloper/twin.git` (`main`)

---

## Sync Gates Overview (SW ↔ ML)

| Gate | Target Time | Shared Milestone | Blocking Condition |
| :--- | :--- | :--- | :--- |
| **Gate 0** | **Hour 0.5** | Dev env & SW test harness ready | Repo cloned; `uvicorn serve:app` + `monsoon_r1/schema_validator.py` verified locally |
| **Gate 1** | **Hour 2.0** | Static grid & schema contract locked | `artifacts/static.npy` pulled from ML; `dashboard/` grid coordinates `(32, 36)` matched |
| **Gate 2** | **Hour 6.0** | Dashboard UI shell + mock/live `/simulate` client working | React + Vite `dashboard/` renders 32×36 heatmap, day scrubber, and shock controls |
| **Gate 3** | **Hour 10.0** | Final ONNX weights in Git LFS | `artifacts/*.onnx` pulled from `main`; full Round-1 verification suite runs on real weights |
| **Gate 4** | **Hour 14.0** | End-to-end live integration green | `serve.py` + `schema_validator.py` (`< 5.0 s`) + `dashboard/` live playback verified together |
| **Gate 5** | **Hour 18.0** | Full end-to-end freeze | Clean clone passes Round-1 checks, `dashboard/` demo works, 2-Page PDF exported (`<= 2` pages) |

---

## Phase 0 — Local Software Environment & Contract Harness (Hours 0.0 – 0.5)

- [ ] **0.1** Clone repo and verify local tooling (Python, Node.js/npm, Git LFS):
  ```bash
  git clone https://github.com/SinisterDeveloper/twin.git
  cd twin
  git lfs install
  ```
- [ ] **0.2** Read `docs/R1_CONTRACT.md`, `monsoon_r1/contract.py`, and `monsoon_r1/schema_validator.py` to lock exact `/health` and `/simulate` JSON schemas.
- [ ] **0.3** Verify `uvicorn serve:app --port 8000` starts cleanly (or with placeholder/smoke weights) and `monsoon_r1/schema_validator.py` executes without import errors.
- [ ] **Gate 0 Check:** Signal Person A (`ML.md`) that local SW/verification environment is ready.

---

## Phase 1 — Automated Verification & Anti-Cheat Suite (Hours 0.5 – 2.0)

- [ ] **1.1** Build a single repeatable verification script (e.g. `scripts/verify_submission.py` or `.sh`/`.ps1` — keep clean & outside `serve.py`) that checks:
  - `GET /health` returns `200` and expected keys.
  - `POST /simulate` passes `monsoon_r1/schema_validator.py` for `warm_ocean`, `monsoon_stall`, and `null` shocks.
  - Latency across **8 consecutive 10-day rollout requests** stays `< 5.0 s` per call, with process RSS `< 6 GB`.
  - Anti-cheat regex scan: zero matches for `open-meteo|openweathermap|weatherbench2|arco-era5|cdsapi|imdlib`.
  - Repo hygiene: zero `.nc` files tracked by git, total repo size `< 1 GB`, `git lfs ls-files` tracks `*.onnx`.
- [ ] **Gate 1 Check (Hour 2.0):** Pull Person A's `artifacts/static.npy` & metadata from `main` to lock exact `(32, 36)` lat/lon grid, land-sea mask, and orography for the frontend.

---

## Phase 2 — React + Vite Digital Twin Dashboard Core (Hours 2.0 – 6.0)

> **Strict Separation Rule:** Build the UI strictly inside `dashboard/` as a standalone React + Vite app calling `serve.py` via HTTP (`GET /health`, `POST /simulate`). Never modify `serve.py` for UI concerns or add frontend dependencies to root `requirements.txt`.

- [ ] **2.1** Scaffold `dashboard/` (React + Vite + TypeScript + Tailwind CSS) and configure Vite dev proxy (`/api` -> `http://localhost:8000`) + CORS check if needed.
- [ ] **2.2** Build **India 32×36 Interactive Grid Map** (Canvas/SVG heatmap over `lat`/`lon`, with coastline & orography contour overlay from `static.npy` / `/health` or bundled JSON export).
- [ ] **2.3** Build **Variable & Vector Controls:**
  - Channel selector: `t2m` (°C / K), `tp` (mm/day), `u10` / `v10` (m/s), `msl` (hPa), `tcwv` (kg/m²).
  - Optional wind vector arrows/barbs overlay from `(u10, v10)`.
- [ ] **2.4** Build **Day Scrubber (Day 0 → Day 10):** Play / pause / step rollout timeline.
- [ ] **Gate 2 Check (Hour 6.0):** Dashboard renders clean 32×36 maps and timeline scrubbing against mock/test `/simulate` payloads; sync with Person A on baseline Val metrics.

---

## Phase 3 — Shock Builder, Delta Comparison & Telemetry UI (Hours 6.0 – 10.0)

- [ ] **3.1** Build **Shock Builder Panel:**
  - One-click presets matching Round-1 regimes: `warm_ocean`, `monsoon_stall`, `null`.
  - Interactive sliders/click-on-map for custom shock center `(lat, lon)`, radius, channel, and intensity.
- [ ] **3.2** Build **Side-by-Side Comparison View:**
  - **Baseline** vs. **Shocked** vs. **Delta (`Shocked - Baseline`)** panels so judges immediately see localized physical propagation!
- [ ] **3.3** Build **Telemetry & Physics Guardrails Bar:**
  - Live `/simulate` latency (ms vs `5000 ms` budget).
  - Null-shock identity check (`max |delta|` vs `1e-3`).
  - Plume spread / stability indicator (`STABLE_MAX_STD <= 5.0`).
- [ ] **Gate 3 Check (Hour 10.0):** Pull Person A's final `artifacts/*.onnx` + `artifacts/*.npz` via `git pull` (`git lfs pull`).

---

## Phase 4 — Full Live Integration & Stress Testing (Hours 10.0 – 14.0)

- [ ] **4.1** Run the full verification suite (`scripts/verify_submission.py`) against `uvicorn serve:app --port 8000` loaded with Person A's final ONNX ensemble.
- [ ] **4.2** Test `dashboard/` end-to-end against live `serve.py`:
  - Verify `warm_ocean`, `monsoon_stall`, and `null` presets trigger real `/simulate` calls and render crisp deltas.
  - Fix any UI state bugs, scaling/colorbar glitches, or CORS/proxy issues.
- [ ] **4.3** Add `dashboard/README.md` (how to run `npm install && npm run dev`) without touching root `serve.py` / `requirements.txt`.
- [ ] **Gate 4 Check (Hour 14.0):** Joint sign-off with Person A that `serve.py`, `schema_validator.py` (`< 5.0 s`), and `dashboard/` all pass live.

---

## Phase 5 — 2-Page Project Brief PDF & Final Submission Freeze (Hours 14.0 – 20.0)

- [ ] **5.1** Author **2-Page Project Brief (PDF, max 2 pages)** incorporating Person A's final Val (`2019–2021`) numbers:
  - **USP:** Physics-aware 2-family ensemble (`ConvNeXt-UNet` + `Advect-ResNet` with differentiable `grid_sample` wind advection) on 4-harmonic climatology + `log1p(tp)` residual dynamics.
  - **System Architecture Diagram:** End-to-end schematic (Ingestion -> 4-Harmonic + Residual State -> 2-Family ONNX Ensemble -> FastAPI `/simulate` + React Digital Twin).
  - **Validation & Anti-Overfitting Proof:** Strict `2019–2021` temporal holdout table + 3-shock causality metrics (`null <= 1e-3`, plume speed `<= 8 cells/day`).
  - **Product Screenshot:** High-res capture of `dashboard/` showing `warm_ocean` / `monsoon_stall` delta rollout.
- [ ] **5.2** Export PDF and verify **exact page count <= 2 pages**.
- [ ] **5.3** **Clean-Clone Final Gate:**
  - Clone `https://github.com/SinisterDeveloper/twin.git` into a brand-new temp directory.
  - Confirm `git lfs ls-files` includes final `.onnx` weights, zero `.nc` files exist, and total repo size `< 1 GB`.
  - Run `uvicorn serve:app --port 8000` + `schema_validator.py` + 8-call latency test.
- [ ] **Gate 5 Check (Hour 18.0 – 19.0):** Final sign-off with Person A. Submit **Public GitHub Repo URL** + **2-Page PDF** before Hour 20.0.
