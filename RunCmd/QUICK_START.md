# E2FGVI 快速启动指南

## 环境信息
- **Conda 环境**: `e2fgvi-project`
- **Python**: 3.12.4
- **PyTorch**: 2.6.0+cu124 (GPU)
- **CUDA**: 12.4
- **GPU**: NVIDIA RTX 4070 Laptop

---

## 1. 激活 Conda 环境

每次使用前需要激活环境：

```powershell
conda activate e2fgvi-project
```

**如果 conda 命令不可用，先初始化 PowerShell：**
```powershell
& "C:\Users\Aiur\miniconda3\shell\condabin\conda-hook.ps1"
conda init powershell
```

然后**重启 PowerShell**，再运行：
```powershell
conda activate e2fgvi-project
```

---

## 2. 验证环境

检查 GPU 和 CUDA 是否正常：

```powershell
conda activate e2fgvi-project
python check_cuda.py
```

**预期输出：**
```
CUDA available: True
CUDA device: NVIDIA GeForce RTX 4070 Laptop GPU
CUDA version: 12.4
Testing PyTorch core dependencies...
2.6.0+cu124
```

---

## 3. 运行测试（视频修复推理）

### 3.1 标准模型 - Tennis 视频

```powershell
conda activate e2fgvi-project
python test.py --model e2fgvi --video examples\tennis --mask examples\tennis_mask --ckpt release_model\E2FGVI-CVPR22.pth
```

**输出位置**: `results\examples\tennis_results.mp4`

### 3.2 高质量模型 (HQ) - Tennis 视频

```powershell
conda activate e2fgvi-project
python test.py --model e2fgvi_hq --video examples\tennis --mask examples\tennis_mask --ckpt release_model\E2FGVI-HQ-CVPR22.pth
```

**输出位置**: `results\examples\tennis_results.mp4`

### 3.3 标准模型 - Schoolgirls 视频

```powershell
conda activate e2fgvi-project
python test.py --model e2fgvi --video examples/schoolgirls.mp4 --mask examples/schoolgirls_mask --ckpt release_model/E2FGVI-CVPR22.pth
```

**输出位置**: `results\examples\schoolgirls_results.mp4`

### 3.4 高质量模型 - Schoolgirls 视频

```powershell
conda activate e2fgvi-project
python test.py --model e2fgvi_hq --video examples/schoolgirls.mp4 --mask examples/schoolgirls_mask --ckpt release_model/E2FGVI-HQ-CVPR22.pth
```

**输出位置**: `results\examples\schoolgirls_results.mp4`

---

## 4. 运行训练

### 4.1 小规模训练（推荐用于测试）

**配置**: 3个视频，100次迭代，batch_size=2

```powershell
conda activate e2fgvi-project
python train.py -c configs/train_e2fgvi_small.json
```

**输出位置**:
- 日志: `logs/e2fgvi_train_e2fgvi_small.log`
- 检查点: `checkpoints/e2fgvi_train_e2fgvi_small/`
- TensorBoard: `checkpoints/e2fgvi_train_e2fgvi_small/gen/` 和 `dis/`

### 4.2 标准训练（完整数据集）

**配置**: YouTube-VOS 完整数据集，150,000次迭代

```powershell
conda activate e2fgvi-project
python train.py -c configs/train_e2fgvi.json
```

**输出位置**:
- 日志: `logs/e2fgvi_train_e2fgvi.log`
- 检查点: `checkpoints/e2fgvi_train_e2fgvi/`

### 4.3 高质量训练

**配置**: YouTube-VOS 完整数据集，更高分辨率

```powershell
conda activate e2fgvi-project
python train.py -c configs/train_e2fgvi_hq.json
```

**输出位置**:
- 日志: `logs/e2fgvi_train_e2fgvi_hq.log`
- 检查点: `checkpoints/e2fgvi_train_e2fgvi_hq/`

---

## 5. 监控训练日志

### 5.1 实时查看日志（自动更新）

```powershell
conda activate e2fgvi-project
python view_training_log.py logs/e2fgvi_train_e2fgvi_small.log -w
```

**参数说明**:
- `-w` / `--watch`: 实时监控模式，每2秒刷新
- `-s` / `--summary`: 显示统计摘要

### 5.2 查看训练摘要

```powershell
conda activate e2fgvi-project
python view_training_log.py logs/e2fgvi_train_e2fgvi_small.log -s
```

**输出示例**:
```
Training Summary:
Total iterations: 74
Losses statistics:
  flow: min=0.2134, max=1.0876, avg=0.5234
  d: min=0.9234, max=1.0123, avg=0.9876
  hole: min=0.1234, max=0.4567, avg=0.2345
  valid: min=0.1123, max=0.3456, avg=0.1987
```

### 5.3 使用 TensorBoard 可视化

```powershell
conda activate e2fgvi-project
tensorboard --logdir checkpoints/e2fgvi_train_e2fgvi_small
```

然后在浏览器打开: http://localhost:6006

---

## 6. 评估模型

```powershell
conda activate e2fgvi-project
python evaluate.py
```

---

## 7. 常用命令速查

| 操作 | 命令 |
|------|------|
| **激活环境** | `conda activate e2fgvi-project` |
| **检查 GPU** | `python check_cuda.py` |
| **快速测试** | `python test.py --model e2fgvi --video examples\tennis --mask examples\tennis_mask --ckpt release_model\E2FGVI-CVPR22.pth` |
| **小规模训练** | `python train.py -c configs/train_e2fgvi_small.json` |
| **监控日志** | `python view_training_log.py logs/e2fgvi_train_e2fgvi_small.log -w` |
| **启动 TensorBoard** | `tensorboard --logdir checkpoints/e2fgvi_train_e2fgvi_small` |

---

## 8. 故障排查

### 8.1 conda 命令找不到

**问题**: `conda : 无法将"conda"项识别为 cmdlet、函数...`

**解决方案**:
```powershell
& "C:\Users\Aiur\miniconda3\shell\condabin\conda-hook.ps1"
conda init powershell
```
然后**重启 PowerShell**。

### 8.2 GPU 不可用

**问题**: `CUDA available: False`

**解决方案**:
1. 检查 NVIDIA 驱动: `nvidia-smi`
2. 重新安装 PyTorch GPU 版本:
```powershell
conda activate e2fgvi-project
pip install --force-reinstall torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

### 8.3 模块缺失错误

**问题**: `ModuleNotFoundError: No module named 'cv2'`

**解决方案**:
```powershell
conda activate e2fgvi-project
pip install -r requirements.txt
```

### 8.4 MMCV 警告

**警告**: `[WARNING] Using deformable convolution compatibility layer`

**说明**: 这是正常的。项目使用了纯 Python 兼容层来替代 MMCV 的 C++ 扩展，不影响功能。

---

## 9. 文件结构说明

```
E2FGVI/
├── configs/                     # 训练配置文件
│   ├── train_e2fgvi_small.json  # 小规模训练 (100 iters)
│   ├── train_e2fgvi.json        # 标准训练 (150k iters)
│   └── train_e2fgvi_hq.json     # 高质量训练
├── datasets/                    # 数据集
│   └── youtube-vos/
│       ├── train_small.json     # 小数据集 (3 videos)
│       └── train.json           # 完整数据集
├── examples/                    # 测试样例
│   ├── tennis/                  # 网球视频帧
│   ├── tennis_mask/             # 对应mask
│   ├── schoolgirls.mp4          # 女学生视频
│   └── schoolgirls_mask/        # 对应mask
├── release_model/               # 预训练模型
│   ├── E2FGVI-CVPR22.pth       # 标准模型
│   └── E2FGVI-HQ-CVPR22.pth    # 高质量模型
├── checkpoints/                 # 训练输出（自动生成）
├── logs/                        # 训练日志（自动生成）
├── results/                     # 测试结果（自动生成）
├── test.py                      # 测试脚本
├── train.py                     # 训练脚本
├── evaluate.py                  # 评估脚本
├── check_cuda.py                # GPU 验证脚本
├── view_training_log.py         # 日志查看工具
├── requirements.txt             # Python 依赖
└── environment_e2fgvi_conda.yml # Conda 环境配置
```

---

## 10. 下一步

1. ✅ **环境已设置**: conda 环境 `e2fgvi-project` 已完全配置
2. ✅ **测试已验证**: tennis 视频修复成功
3. ✅ **训练可运行**: 小规模训练正在进行中
4. 📝 **建议操作**:
   - 等待小规模训练完成（100 iterations）
   - 使用 TensorBoard 查看训练曲线
   - 尝试在更多视频上测试
   - 准备完整数据集进行全规模训练

---

## 附录：完整的一键启动脚本

创建 `start_training.ps1`:

```powershell
# 激活环境
conda activate e2fgvi-project

# 验证 GPU
Write-Host "=== 检查 GPU ===" -ForegroundColor Green
python check_cuda.py

# 启动训练
Write-Host "`n=== 开始训练 ===" -ForegroundColor Green
python train.py -c configs/train_e2fgvi_small.json
```

运行:
```powershell
.\start_training.ps1
```

---

**最后更新**: 2025-11-17  
**环境版本**: e2fgvi-project (Python 3.12.4, PyTorch 2.6.0+cu124)
