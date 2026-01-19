import sys
import json
import traceback
import os

print(f"[DEBUG] Current directory: {os.getcwd()}")
print(f"[DEBUG] Python executable: {sys.executable}")
print(f"[DEBUG] Python version: {sys.version}")

try:
    print("[DEBUG] Loading config...")
    config = json.load(open('configs/train_e2fgvi.json'))
    print(f"[DEBUG] Config keys: {list(config.keys())}")
    print(f"[DEBUG] train_data_loader: {config.get('train_data_loader', {})}")
    
    print("[DEBUG] Checking dataset path...")
    data_root = config['train_data_loader'].get('data_root', 'datasets')
    dataset_name = config['train_data_loader'].get('name', 'youtube-vos')
    dataset_path = os.path.join(data_root, dataset_name)
    print(f"[DEBUG] Dataset path: {dataset_path}")
    print(f"[DEBUG] Dataset exists: {os.path.isdir(dataset_path)}")
    print(f"[DEBUG] JPEGImages exists: {os.path.isdir(os.path.join(dataset_path, 'JPEGImages'))}")
    print(f"[DEBUG] train.json exists: {os.path.isfile(os.path.join(dataset_path, 'train.json'))}")
    
    print("[DEBUG] Importing Trainer...")
    from core.trainer import Trainer
    print("[DEBUG] Trainer imported successfully")
    
    print("[DEBUG] Attempting to initialize Trainer...")
    trainer = Trainer(config)
    print("[DEBUG] Trainer initialized successfully")
    print("[DEBUG] All checks passed!")
    
except Exception as e:
    print(f"[ERROR] {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)
