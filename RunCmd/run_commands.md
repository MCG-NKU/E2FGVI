# E2FGVI 运行命令清单

## 环境激活命令

### 方法1: 使用 conda hook（推荐）
```powershell
# 每次打开新的 PowerShell 窗口后先运行这个
& "C:\Users\Aiur\miniconda3\shell\condabin\conda-hook.ps1"

# 然后激活环境
conda activate e2fgvi-project
```

### 方法2: 直接使用 activate.bat
```powershell
& "C:\Users\Aiur\miniconda3\Scripts\activate.bat" e2fgvi-project
```

### 验证环境
```powershell
# 检查 Python 版本和路径
python --version
python -c "import sys; print(sys.executable)"

# 检查 GPU 支持
python check_cuda.py
```

---

## 测试命令 (TEST)

### 1. 标准模型测试 - Tennis 视频
```powershell
python test.py --model e2fgvi --video examples\tennis --mask examples\tennis_mask --ckpt release_model\E2FGVI-CVPR22.pth
```

### 2. 标准模型测试 - Schoolgirls 视频
```powershell
python test.py --model e2fgvi --video examples/schoolgirls.mp4 --mask examples/schoolgirls_mask --ckpt release_model/E2FGVI-CVPR22.pth
```

### 3. HQ 高质量模型测试 - Tennis 视频
```powershell
python test.py --model e2fgvi_hq --video examples\tennis --mask examples\tennis_mask --ckpt release_model\E2FGVI-HQ-CVPR22.pth
```

### 4. HQ 高质量模型测试 - Schoolgirls 视频
```powershell
python test.py --model e2fgvi_hq --video examples/schoolgirls.mp4 --mask examples/schoolgirls_mask --ckpt release_model/E2FGVI-HQ-CVPR22.pth
```

**测试结果保存位置**: `results/examples/`

---

## 训练命令 (TRAIN)

### 1. 小规模训练（快速验证，推荐首次运行）
```powershell
python train.py -c configs/train_e2fgvi_small.json
```
- **数据集**: 3个视频（train_small.json）
- **迭代次数**: 100 iterations
- **batch_size**: 2
- **预计时间**: ~10-15分钟
- **日志位置**: `logs/e2fgvi_train_e2fgvi_small.log`
- **检查点**: `checkpoints/e2fgvi_train_e2fgvi_small/`

### 2. 标准训练
```powershell
python train.py -c configs/train_e2fgvi.json
```
- **数据集**: 完整 youtube-vos 数据集
- **迭代次数**: 300,000 iterations
- **batch_size**: 8
- **预计时间**: 数天
- **日志位置**: `logs/e2fgvi_train_e2fgvi.log`
- **检查点**: `checkpoints/e2fgvi_train_e2fgvi/`

### 3. HQ 高质量训练
```powershell
python train.py -c configs/train_e2fgvi_hq.json
```
- **数据集**: 完整 youtube-vos 数据集
- **迭代次数**: 300,000 iterations
- **batch_size**: 4（更大的模型，需要更多显存）
- **日志位置**: `logs/e2fgvi_train_e2fgvi_hq.log`
- **检查点**: `checkpoints/e2fgvi_train_e2fgvi_hq/`

---

## 日志监控命令

### 实时监控训练日志
```powershell
# 查看最新 100 行日志
python view_training_log.py logs\e2fgvi_train_e2fgvi_small.log -n 100

# 显示训练统计摘要
python view_training_log.py logs\e2fgvi_train_e2fgvi_small.log -s

# 实时监控模式（类似 tail -f）
python view_training_log.py logs\e2fgvi_train_e2fgvi_small.log -w
```

### 使用 PowerShell 原生命令查看日志
```powershell
# 查看最后 50 行
Get-Content logs\e2fgvi_train_e2fgvi_small.log -Tail 50

# 实时监控（PowerShell 等效 tail -f）
Get-Content logs\e2fgvi_train_e2fgvi_small.log -Wait -Tail 10
```

---

## 评估命令

### 评估训练好的模型
```powershell
python evaluate.py
```

---

## 完整工作流示例

### 首次运行完整流程
```powershell
# 1. 激活环境
& "C:\Users\Aiur\miniconda3\shell\condabin\conda-hook.ps1"
conda activate e2fgvi-project

# 2. 验证 GPU
python check_cuda.py

# 3. 运行测试（验证环境）
python test.py --model e2fgvi --video examples\tennis --mask examples\tennis_mask --ckpt release_model\E2FGVI-CVPR22.pth

# 4. 小规模训练（验证数据和训练流程）
python train.py -c configs/train_e2fgvi_small.json

# 5. 监控日志
python view_training_log.py logs\e2fgvi_train_e2fgvi_small.log -s
```

---

## 常见问题解决

### 问题1: conda 命令不识别
**解决方案**:
```powershell
# 运行 conda hook
& "C:\Users\Aiur\miniconda3\shell\condabin\conda-hook.ps1"

# 或者初始化 PowerShell（只需运行一次）
& "C:\Users\Aiur\miniconda3\Scripts\conda.exe" init powershell
# 然后重启 PowerShell
```

### 问题2: ModuleNotFoundError
**解决方案**:
```powershell
# 确保在正确的环境中
conda activate e2fgvi-project

# 重新安装依赖
pip install -r requirements.txt
```

### 问题3: CUDA out of memory
**解决方案**:
- 减小 batch_size（在 config 文件中）
- 使用小规模配置：`train_e2fgvi_small.json`
- 关闭其他占用显存的程序

### 问题4: 检查当前环境
```powershell
# 查看所有 conda 环境
conda env list

# 查看当前环境中的包
pip list

# 查看环境详情
conda info
```

---

## 文件路径说明

| 文件/目录 | 说明 |
|----------|------|
| `configs/` | 训练配置文件 |
| `datasets/youtube-vos/` | 训练数据集 |
| `examples/` | 测试样例视频和蒙版 |
| `release_model/` | 预训练模型权重 |
| `checkpoints/` | 训练过程保存的检查点 |
| `logs/` | 训练日志文件 |
| `results/` | 测试结果输出 |
| `scores/` | 评估分数 |

---

## 环境信息

- **环境名称**: e2fgvi-project
- **Python 版本**: 3.12.4
- **PyTorch 版本**: 2.6.0+cu124
- **CUDA 版本**: 12.4
- **GPU**: NVIDIA GeForce RTX 4070 Laptop GPU
- **环境路径**: `C:\Users\Aiur\miniconda3\envs\e2fgvi-project`

---

*最后更新: 2025-11-17*
