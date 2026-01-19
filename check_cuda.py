import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("CUDA not available - using CPU mode")

print("Testing PyTorch core dependencies...")
print(torch.__version__)
# 如果没有报错并能打印版本号 (1.5.1)，则 PyTorch 核心依赖加载成功。

# 退出 Python 解释器
exit()