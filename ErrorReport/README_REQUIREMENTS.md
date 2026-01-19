# Installation notes for the pinned `requirements.txt`

This document explains how to install the packages pinned in `requirements.txt` and best practices for getting the environment to match the one used to run `test.py` successfully.

1) Recommended approach (conda + pip):

- Create a fresh conda environment (example uses Python 3.12):

```powershell
conda create -n e2fgvi_py312 python=3.12 -y
conda activate e2fgvi_py312
```

- Install PyTorch with CUDA support that matches your GPU driver/CUDA runtime. Example for CUDA 12.x (choose the correct channel and version from PyTorch site):

```powershell
# using conda-forge/pytorch channel (example) - adjust for your CUDA version
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c conda-forge -y
```

- Install the remaining pip packages from `requirements.txt`:

```powershell
python -m pip install -r requirements.txt
```

2) Notes about `mmcv` vs `mmcv-full`:

- In this repository run we used `mmcv==1.7.2` (pure-Python mmcv). That is sufficient because the code was adjusted to fall back to a compatibility implementation for deformable conv ops.
- If you want the native (fast) C++/CUDA ops, you must install `mmcv-full` compiled for your exact PyTorch and CUDA versions. Example link format from OpenMMLab:

```powershell
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0.1/index.html
```

Replace the `cu118` / `torch2.0.1` tags with the wheel matching your environment. If a prebuilt wheel is not available for your configuration, mmcv-full will try building from source (requires Visual Studio build tools on Windows and a compatible CUDA toolkit).

3) Troubleshooting tips:

- If you see `ImportError: DLL load failed while importing _ext` or `No module named 'mmcv._ext'`, you are using a mmcv-full binary that doesn't match your CUDA/PyTorch/Python ABI. Two options:
  - Install / rebuild `mmcv-full` for your exact CUDA/PyTorch versions.
  - Use the provided compatibility fallback (recommended quick fix): keep `mmcv` (pure Python) and ensure `model/modules/deform_conv_compat.py` and modified imports in `model/modules/feat_prop.py` are present.

- If you prefer pip-only installation in a venv (no conda):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# install PyTorch with the correct wheel (see https://pytorch.org/get-started/locally/)
```

4) Optional: export environment.yml

If you want a conda `environment.yml` instead, create one after setting up the environment:

```powershell
conda env export --no-builds > environment_exported.yml
```

This repo already includes `environment_fixed.yml` as a starting point for CUDA 12 setups — you may adjust it to match your local runtime.

---

If you希望我把 `requirements.txt` 中的版本作微调（比如把 torch 固定为 2.5.x 或降级到 2.4.x，或使用特定 CUDA 版本的 PyTorch wheel），告诉我你的目标 CUDA 和 PyTorch 版本，我可以生成更精确的安装命令。