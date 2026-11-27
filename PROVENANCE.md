# Model Provenance, Data Split, and Shock Characterisation

## P1 & P2: Frozen Checkpoints & ONNX Mapping

All three active ensemble members are frozen from pre-2023 training (`<= 2022-12-31`) and tracked via Git LFS (`*.onnx`, `*.pt` in `.gitattributes`). `local_swin.onnx` remains on disk with `0.0` weight in `artifacts/ensemble_weights.json`.

| Served ONNX | Tracked PyTorch Checkpoint | Training Stage | Train/Val Cutoff | Pre-Reexport `pt` vs `onnx` Max Diff | Dynamic-Batch (`batch`) Re-export Max Diff |
| :--- | :--- | :--- | :--- | :---: | :---: |
| `artifacts/advect_resnet.onnx` | `artifacts/pt/advect_resnet_best.pt` | Stage A (`stageA_advect_resnet`, `2026-11-27`) | `2005–2018` train, `2019–2021` val (`<= 2022-12-31`) | `9.090e-07` | `8.34e-07` |
| `artifacts/convnext_unet.onnx` | `artifacts/pt/convnext_unet_best.pt` | Stage A (`stageA_convnext_unet`, `2026-11-27`) | `2005–2018` train, `2019–2021` val (`<= 2022-12-31`) | `1.848e-06` | `3.10e-06` |
| `artifacts/pde_net.onnx` | `artifacts/pt/pde_net_best.pt` | Stage B22 (`stageB22_pde_net`, `2026-11-27`) | `2005–2021` train, `2022` gate (`<= 2022-12-31`) | `1.013e-06` | `1.19e-06` |

- **Ensemble weights:** `artifacts/ensemble_weights.json` (fitted strictly on `2019–2021` validation, `local_swin = 0.0`).
- **Dynamic batching (P4):** All three ONNX models export `z_hist`, `doy_sc`, and `z_next` with dynamic axis `0` (`batch`), verified for `B=1, 2, 3` (`max_diff <= 1e-4`), and preserve the blind `2023–2024` `eval-live` score of **`68.44 / 70`** (`worst_max_z = 4.0316`, `tp_blur = 0.7379`, `schema_validator` pass).

## P3 & P5: Differentiable PyTorch Rollout & Per-Family Outputs

- `twin/rollout.py` (`load_ensemble_torch`, `rollout_ensemble_torch`) reproduces `serve.simulate` end-to-end in PyTorch (normalization, `log1p`/`expm1` `tp` transform capped at `TP_LOG1P_MAX = 4.15`, `t2m` `[233.15, 333.15]` clamp, DOY `(sin, cos)` rotation, and per-channel blend).
  - Verified max absolute difference vs `serve.simulate` on real `2023–2024` windows (`2023-01-23`, `2023-06-08`, `2024-06-30`): **`0.015625`** (1 ULP of `float32` at `~101,000 Pa` `msl`).
- `serve.py` exposes `simulate_with_families(...)` returning `(ens, family_rollouts)` while `simulate(...)` and `POST /simulate` remain identical.

## P6: Data Split for USP 2 (Analog Library)

- **In-sample library (`2005–2018`):** `india_climate_train.nc` (`https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/download/v1.0.0/india_climate_train.nc`, pulled on Colab; not committed to git).
- **Out-of-sample evaluation (`2019+`):** `2019–2022` validation windows (from `india_climate_train.nc`) and `2023–2024` blind test windows (`data/post2023_eval.nc`, local gitignored).

## P7: Shock Behaviour & USP Demo Cases

| Window Date | Season | Shock | Passes All Checks | Propagates | `max_z` (Channel) |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `2023-01-23` | non-monsoon | `warm_ocean` | `False` | `False` | `1.2258` (`tp`) |
| `2023-01-23` | non-monsoon | `monsoon_stall` | `False` | `False` | `0.8219` (`v10`) |
| **`2023-06-08`** | **monsoon** | **`warm_ocean`** | **`True`** | **`True`** | **`1.0841` (`tp`)** |
| `2023-06-08` | monsoon | `monsoon_stall` | `False` | `False` | `4.0316` (`tp`) |
| `2023-12-01` | non-monsoon | `warm_ocean` | `True` | `True` | `2.0339` (`tp`) |
| `2023-12-01` | non-monsoon | `monsoon_stall` | `True` | `True` | `0.6251` (`v10`) |
| `2024-05-02` | monsoon | `warm_ocean` | `True` | `True` | `0.8868` (`u10`) |
| `2024-05-02` | monsoon | `monsoon_stall` | `False` | `False` | `0.5402` (`v10`) |
| **`2024-06-30`** | **monsoon** | **`warm_ocean`** | **`True`** | **`True`** | **`1.7343` (`tp`)** |
| `2024-06-30` | monsoon | `monsoon_stall` | `True` | `True` | `1.5672` (`tp`) |
| `2024-12-11` | non-monsoon | `warm_ocean` | `True` | `True` | `2.5679` (`tp`) |
| `2024-12-11` | non-monsoon | `monsoon_stall` | `False` | `False` | `0.6497` (`v10`) |
| `2022-05-08`..`2022-09-10` (5 windows) | monsoon | `warm_ocean` & `monsoon_stall` | `True` (10/10) | `True` (10/10) | `0.7861`–`3.0915` |

**Selected USP Demo Cases:** Monsoon-season `warm_ocean` windows **`2023-06-08`** and **`2024-06-30`** (both pass all locality, non-teleportation, stability, and propagation checks cleanly).
