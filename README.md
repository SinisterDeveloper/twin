# Multiverse of Monsoon — Climate Digital Twin

Physics-aware spatiotemporal Deep Learning Digital Twin for India's weather (`32 × 36` 1° grid, 6 channels: `t2m`, `tp`, `u10`, `v10`, `msl`, `tcwv`) built for **Yugaantar DevForge Track 2**.

## Architecture
- **State Formulation:** Smooth 4-harmonic 365-day climatology + `log1p(tp)` precipitation transform (guarantees `tp >= 0`) + residual tendency (`a_{t+1} = a_t + Δa`).
- **Complementary Model Families (Causality-Safe, Bounded Receptive Field):**
  - `convnext_unet` (Family A): Shallow 2-scale ConvNeXt-UNet with `GroupNorm` and `FiLM(doy_sincos)` conditioning.
  - `advect_resnet` (Family B): Physics-informed semi-Lagrangian wind advection (`grid_sample` driven by `u10/v10`) + local residual reaction-diffusion blocks.
- **2-Stage A100 Curriculum:** 1-step pretraining → unrolled autoregressive BPTT (`4 → 8 → 10` days) with pushforward noise, latitude-weighted Huber, heavy-rain (`>10 mm/day`) emphasis, and spatial gradient + variance anti-blur losses.
- **Inference:** Deterministic ONNX Runtime ensemble in `serve.py` (`<500 ms` per 10-day rollout on 4 CPU cores).

## Colab A100 Quickstart (CLI-First)
```bash
git clone https://github.com/SinisterDeveloper/twin.git && cd twin
pip install -r requirements.txt

# 1. Verify model architectures
python -m twin.cli smoke-test

# 2. Download dataset in cloud & prepare climatology/static artifacts (2005-2018 train split)
python -m twin.cli download-data
python -m twin.cli prepare-artifacts --train-end 2018-12-31

# 3. Train Family A (Person A) & Family B (Person B)
python -m twin.cli train --model convnext_unet --epochs-s1 25 --epochs-s2 25 --out-dir runs/a
python -m twin.cli train --model advect_resnet --epochs-s1 25 --epochs-s2 25 --out-dir runs/b

# 4. Evaluate on 25-window 5-regime + 3-shock Val benchmark (2019-2021)
python -m twin.cli eval --ckpts runs/a/convnext_unet_best.pt runs/b/advect_resnet_best.pt

# 5. Export ONNX ensemble & run offline server
python -m twin.cli export-onnx --ckpt runs/a/convnext_unet_best.pt --out artifacts/convnext_unet.onnx
python -m twin.cli export-onnx --ckpt runs/b/advect_resnet_best.pt --out artifacts/advect_resnet.onnx
python serve.py --port 8000
```

## Verification
```bash
python -m monsoon_r1.schema_validator http://localhost:8000
python -c "from monsoon_r1.repo_scan import scan; import json; print(json.dumps(scan('.'), indent=2))"
```
