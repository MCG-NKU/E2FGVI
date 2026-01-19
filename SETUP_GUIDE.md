# E2FGVI 环境配置与使用指南

**最后更新**: 2025-11-14  
**系统**: Windows 11 + RTX 4070 Laptop GPU + CUDA 12.7  
**状态**: ✅ 已验证可正常运行

---

## 快速开始

### 1. 环境配置（一次性）

#### 选项 A: 使用 Conda（推荐）
```bash
# 创建环境
conda env create -f environment_e2fgvi.yml

# 激活环境
conda activate e2fgvi
```

#### 选项 B: 使用 pip 和现有 Python
```bash
# 确保 Python 3.12.4
python --version

# 安装依赖
pip install -r requirements.txt

# 验证 GPU 支持
python check_cuda.py
```

### 2. 准备数据集

数据集需要以 **zip 格式** 存储，结构如下：

```
datasets/
  ├── youtube-vos/
  │   ├── JPEGImages/
  │   │   ├── 00a23ccf53.zip  # 视频 ID 作为 zip 文件名
  │   │   ├── 00ad5016a4.zip
  │   │   └── ...
  │   ├── test_masks/          # 测试掩码
  │   │   ├── 00a23ccf53/
  │   │   │   ├── 00000.png
  │   │   │   └── ...
  │   │   └── ...
  │   ├── train.json           # 训练数据映射
  │   ├── train_small.json     # 小规模测试用
  │   └── test.json
```

**压缩视频文件夹**（如果有原始文件夹）：
```powershell
# Windows PowerShell
Get-ChildItem datasets\youtube-vos\JPEGImages -Directory | ForEach-Object {
    Write-Host "Compressing $($_.Name)..."
    $zipName = "$($_.FullName).zip"
    Compress-Archive -Path $_.FullName -DestinationPath $zipName -Force
    Remove-Item $_.FullName -Recurse -Force
}
```

### 3. 验证环境

```bash
# 检查 CUDA 支持
python check_cuda.py

# 预期输出:
# CUDA available: True
# CUDA device: NVIDIA GeForce RTX 4070 Laptop GPU
# CUDA version: 12.4
```

---

## 训练

### 快速测试（小数据集，~15 分钟）

用于快速验证训练流程是否正常运行：

```bash
python train.py -c configs/train_e2fgvi_small.json
```

**配置说明**（`configs/train_e2fgvi_small.json`）：
- 数据集：3 个视频（youtube-vos 的小子集）
- 迭代次数：100
- Batch size：2
- 日志输出：每 1 步记录一次

### 完整训练（标准配置）

```bash
python train.py -c configs/train_e2fgvi.json
```

**配置说明**（`configs/train_e2fgvi.json`）：
- 数据集：完整 youtube-vos 训练集
- 迭代次数：500,000
- Batch size：8
- 日志输出：每 100 步记录一次

### 高质量训练（HQ 模型）

```bash
python train.py -c configs/train_e2fgvi_hq.json
```

---

## 监测训练进度

### 实时查看训练日志

```bash
# 显示最近的迭代信息（50 行）
python view_training_log.py logs/e2fgvi_train_e2fgvi_small.log

# 显示摘要统计（损失均值、最大值等）
python view_training_log.py logs/e2fgvi_train_e2fgvi_small.log -s

# 实时监视日志（类似 tail -f）
python view_training_log.py logs/e2fgvi_train_e2fgvi_small.log -w
```

### 输出示例

```
✓ [Iter 74] flow: 0.6701 d: 0.9977 hole: 0.1712 valid: 0.1664 @15:46:36

📈 最新: 第 74 步 | flow: 0.6701 | d: 0.9977 | hole: 0.1712 | valid: 0.1664
```

**损失项说明**：
- `flow`：光流估计损失
- `d`：判别器损失（GAN）
- `hole`：空洞区域修复损失
- `valid`：有效区域一致性损失

---

## 关键改进与兼容性处理

### 1. MMCV DLL 兼容性问题修复

**问题**：原生 `mmcv-full` 1.7.2 的 C++ 扩展编译为特定的 CUDA/Python ABI，在不同环境中加载失败。

**解决方案**：
- 创建 `model/modules/deform_conv_compat.py`：提供纯 PyTorch 实现的可变形卷积算子
- 修改 `model/modules/feat_prop.py`：添加 try-except 导入逻辑，自动降级到兼容层

**相关文件**：
```
model/modules/
  ├── deform_conv_compat.py      # 新增：兼容层实现
  └── feat_prop.py               # 已修改：添加降级逻辑
```

### 2. 配置字段补全

添加了原代码缺失的必要字段到 `configs/train_*.json`：
```json
{
  "distributed": false,      // 单机训练
  "world_size": 1,           // 进程数
  "local_rank": 0,           // 本地进程 ID
  "global_rank": 0,          // 全局进程 ID
  "device": "cuda"           // 计算设备
}
```

### 3. 数据集加载灵活性

修改 `core/dataset.py` 支持自定义训练数据文件名：
```json
{
  "train_data_loader": {
    "train_file": "train_small.json"  // 可指定不同的数据集
  }
}
```

---

## 文件结构

### 核心训练相关

| 文件 | 用途 |
|-----|-----|
| `train.py` | 主训练脚本 |
| `configs/train_e2fgvi.json` | 标准训练配置 |
| `configs/train_e2fgvi_small.json` | 快速测试配置 |
| `core/trainer.py` | 训练器主类 |
| `core/dataset.py` | 数据集加载 |
| `core/loss.py` | 损失函数定义 |

### 新增文件

| 文件 | 用途 |
|-----|-----|
| `model/modules/deform_conv_compat.py` | MMCV 兼容性层 |
| `view_training_log.py` | 日志浏览工具 |
| `check_cuda.py` | GPU 验证工具 |
| `debug_train.py` | 训练调试工具 |
| `environment_e2fgvi.yml` | Conda 环境文件 |
| `requirements.txt` | pip 依赖文件 |

### 输出目录

```
checkpoints/
  └── e2fgvi_train_e2fgvi_small/
      ├── gen_*.pth              # 生成器权重检查点
      ├── dis_*.pth              # 判别器权重检查点
      ├── opt_*.pth              # 优化器状态
      └── latest.ckpt            # 最新检查点
logs/
  └── e2fgvi_train_e2fgvi_small.log  # 训练日志
```

---

## 环境信息

### 已验证的依赖版本

| 包 | 版本 | 备注 |
|----|-----|------|
| Python | 3.12.4 | - |
| PyTorch | 2.6.0+cu124 | GPU 版本，CUDA 12.4 |
| torchvision | 0.21.0+cu124 | - |
| torchaudio | 2.6.0+cu124 | - |
| MMCV | 1.7.2 | 纯 Python 模式（无 C++ ops） |
| CUDA | 12.4 (wheels) / 12.7 (runtime) | 兼容 |
| cuDNN | 9.x | 自动包含在 torch wheels 中 |
| NumPy | 1.26.4 | - |
| OpenCV | 4.10.0.84 | - |

### GPU 硬件

- **模型**：NVIDIA GeForce RTX 4070 Laptop GPU
- **显存**：8GB
- **CUDA Compute Capability**：8.9

---

## 常见问题与解决方案

### Q1: ImportError: DLL load failed while importing _ext

**原因**：mmcv 的 C++ 扩展与当前环境不兼容。

**解决**：已通过 `deform_conv_compat.py` 自动处理。如果仍出现警告，属于正常情况（会自动降级）。

```python
[WARNING] Using deformable convolution compatibility layer
```

### Q2: CUDA 计算能力不足

**错误**：`RuntimeError: CUDA out of memory`

**解决**：减少 `batch_size` 或图像分辨率（`w`, `h`）：
```json
{
  "trainer": {
    "batch_size": 4,  // 从 8 降低到 4
    "w": 216,         // 从 432 降低到 216
    "h": 120          // 从 240 降低到 120
  }
}
```

### Q3: 训练速度慢

**原因**：数据加载是瓶颈。

**优化**：
1. 确保视频文件已压缩为 zip 格式
2. 增加 `num_workers`（数据加载线程数）
3. 增加 `batch_size`（如显存允许）

### Q4: 找不到数据集文件

**错误**：`FileNotFoundError: [Errno 2] No such file or directory: 'datasets\youtube-vos\JPEGImages\*.zip'`

**检查**：
1. 确认数据集结构正确（见上面的数据准备部分）
2. 视频文件夹已转换为 zip 格式
3. 配置文件中的 `data_root` 和 `name` 字段正确

---

## Git 提交记录

所有改进已提交到分支 `fix/mmcv-dll-compat`：

```bash
git log --oneline fix/mmcv-dll-compat | head -5
# 包含以下改动：
# - 添加 MMCV 兼容性层
# - 修复配置缺失字段
# - 添加数据集灵活加载
# - 完整的环境文档
```

---

## 下一步建议

1. **扩大数据集**：用完整的 youtube-vos 训练集运行 `train_e2fgvi.json`
2. **模型评估**：使用 `evaluate.py` 在 DAVIS 等基准数据集上评估模型
3. **推理部署**：使用 `test.py` 对新视频进行补帧/修复
4. **超参数调优**：根据 GPU 显存调整 batch_size、学习率等

---

## 联系方式

如遇问题，请检查：
1. 本指南的"常见问题"部分
2. `logs/` 目录中的详细训练日志
3. GitHub Issues 或提交 bug report

**验证命令集**（快速自检）：

```bash
# 1. 检查 Python 版本
python --version

# 2. 检查 CUDA 支持
python check_cuda.py

# 3. 运行小规模训练测试（~15 分钟）
python train.py -c configs/train_e2fgvi_small.json

# 4. 查看训练结果
python view_training_log.py logs/e2fgvi_train_e2fgvi_small.log -s
```

所有步骤成功完成 ✅ 表示环境已就绪，可以开始大规模训练。

## 更加详细的包信息:
```
absl-py==2.1.0
addict==2.4.0
anyio==4.8.0
argon2-cffi==23.1.0
argon2-cffi-bindings==21.2.0
arrow==1.3.0
asttokens==3.0.0
astunparse==1.6.3
async-lru==2.0.4
attrs==25.1.0
babel==2.17.0
beautifulsoup4==4.12.3
bleach==6.2.0
certifi==2024.7.4
cffi==1.17.1
chardet==3.0.4
charset-normalizer==3.3.2
click==8.1.7
colorama==0.4.6
comm==0.2.2
contourpy==1.2.1
cycler==0.12.1
debugpy==1.8.12
decorator==5.1.1
defusedxml==0.7.1
einops==0.8.0
et_xmlfile==2.0.0
executing==2.2.0
fairscale==0.4.13
fastjsonschema==2.21.1
filelock==3.15.4
flatbuffers==24.3.25
fonttools==4.53.1
fqdn==1.5.1
fsspec==2024.6.1
gast==0.6.0
gitdb==4.0.11
GitPython==3.1.43
google-pasta==0.2.0
googletrans==4.0.0rc1
grpcio==1.65.1
h11==0.14.0
h2==3.2.0
h5py==3.11.0
hpack==3.0.0
hstspreload==2024.7.1
httpcore==1.0.7
httpx==0.28.1
hyperframe==5.2.0
idna==2.10
idx2numpy==1.2.3
imageio==2.34.2
ipykernel==6.29.5
ipython==8.32.0
ipywidgets==8.1.5
isoduration==20.11.0
jedi==0.19.2
Jinja2==3.1.4
joblib==1.4.2
json5==0.10.0
jsonpointer==3.0.0
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
jupyter_client==8.6.3
jupyter_core==5.7.2
jupyter_server_terminals==0.5.3
jupyter_server==2.15.0
jupyter==1.1.1
jupyter-console==6.6.3
jupyter-events==0.12.0
jupyterlab_pygments==0.3.0
jupyterlab_server==2.27.3
jupyterlab_widgets==3.0.13
jupyterlab==4.3.5
jupyter-lsp==2.2.5
keras==3.4.1
kiwisolver==1.4.5
labml==0.4.168
labml-helpers==0.4.89
labml-nn==0.4.136
lazy_loader==0.4
libclang==18.1.1
Markdown==3.6
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.9.1
matplotlib-inline==0.1.7
mdurl==0.1.2
mistune==3.1.1
ml-dtypes==0.4.0
mmcv==1.7.2
mpmath==1.3.0
namex==0.0.8
nbclient==0.10.2
nbconvert==7.16.6
nbformat==5.10.4
nest-asyncio==1.6.0
networkx==3.3
nltk==3.8.1
notebook_shim==0.2.4
notebook==7.3.2
numpy==1.26.4
opencv-python==4.10.0.84
openpyxl==3.1.5
opt-einsum==3.3.0
optree==0.12.1
overrides==7.7.0
packaging==24.1
pandas==2.2.2
pandocfilters==1.5.1
parso==0.8.4
pillow==10.4.0
pip==25.0
platformdirs==4.3.6
prometheus_client==0.21.1
prompt_toolkit==3.0.50
protobuf==4.25.4
psutil==6.1.1
pure_eval==0.2.3
pycparser==2.22
Pygments==2.18.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
python-json-logger==3.2.1
pytz==2024.1
pywebio==1.8.3
pywin32==308
pywinpty==2.0.15
PyYAML==6.0.1
pyzmq==26.2.1
referencing==0.36.2
regex==2024.7.24
requests==2.32.3
rfc3339-validator==0.1.4
rfc3986==1.5.0
rfc3986-validator==0.1.1
rich==13.7.1
rpds-py==0.22.3
scikit-image==0.24.0
scikit-learn==1.5.1
scipy==1.14.0
seaborn==0.13.2
Send2Trash==1.8.3
setuptools==71.1.0
six==1.16.0
smmap==5.0.1
sniffio==1.3.1
soupsieve==2.5
stack-data==0.6.3
sympy==1.13.1
tensorboard==2.17.0
tensorboard-data-server==0.7.2
tensorflow==2.17.0
tensorflow-intel==2.17.0
termcolor==2.4.0
terminado==0.18.1
threadpoolctl==3.5.0
tifffile==2024.7.24
tinycss2==1.4.0
torch==2.6.0+cu124
torchaudio==2.6.0+cu124
torchtext==0.18.0
torchvision==0.21.0+cu124
tornado==6.4.1
tqdm==4.66.4
traitlets==5.14.3
types-python-dateutil==2.9.0.20241206
typing_extensions==4.12.2
tzdata==2024.1
ua-parser==0.18.0
uri-template==1.3.0
ua-parser==0.18.0
uri-template==1.3.0
urllib3==2.2.2
user-agents==2.2.0
wcwidth==0.2.13
webcolors==24.11.1
webencodings==0.5.1
websocket-client==1.8.0
ua-parser==0.18.0
uri-template==1.3.0
urllib3==2.2.2
user-agents==2.2.0
wcwidth==0.2.13
webcolors==24.11.1
webencodings==0.5.1
ua-parser==0.18.0
uri-template==1.3.0
urllib3==2.2.2
user-agents==2.2.0
wcwidth==0.2.13
ua-parser==0.18.0
uri-template==1.3.0
urllib3==2.2.2
user-agents==2.2.0
wcwidth==0.2.13
uri-template==1.3.0
urllib3==2.2.2
user-agents==2.2.0
urllib3==2.2.2
user-agents==2.2.0
user-agents==2.2.0
wcwidth==0.2.13
wcwidth==0.2.13
webcolors==24.11.1
webcolors==24.11.1
webencodings==0.5.1
webencodings==0.5.1
websocket-client==1.8.0
websocket-client==1.8.0
Werkzeug==3.0.3
Werkzeug==3.0.3
wheel==0.43.0
wheel==0.43.0
widgetsnbextension==4.0.13
wrapt==1.16.0
xlrd==2.0.1
yapf==0.43.0
```
