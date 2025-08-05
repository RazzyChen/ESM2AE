#!/usr/bin/env python3
"""
简化的ESM2AE训练脚本
支持直接指定配置文件启动训练

使用方法:
python train.py --config train_config/train_config_2080super.yaml
python train.py --config train_config/train_config_3080.yaml
"""

import argparse
import os
import yaml
import torch
from pathlib import Path

# 根据配置选择训练方式
def main():
    parser = argparse.ArgumentParser(description="ESM2AE训练脚本")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="训练配置文件路径"
    )
    
    args = parser.parse_args()
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return
    
    # 读取配置文件
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"🚀 使用配置文件: {config_path}")
    
    # 根据配置中的ray设置决定使用哪种训练方式
    if 'ray' in config and config['ray'].get('num_workers', 1) > 1:
        print("📡 检测到多GPU配置，使用Ray分布式训练")
        launch_ray_training(config_path)
    else:
        print("🔧 使用标准训练模式")
        launch_standard_training(config_path)

def launch_ray_training(config_path):
    """启动Ray分布式训练"""
    # 设置配置文件为当前配置
    config_dir = Path("train_config")
    target_config = config_dir / "trainer_config.yaml"
    
    # 复制配置文件
    import shutil
    shutil.copy2(config_path, target_config)
    
    # 同时复制对应的DeepSpeed配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if 'deepspeed' in config and 'config_path' in config['deepspeed']:
        deepspeed_src = Path(config['deepspeed']['config_path'])
        deepspeed_dst = config_dir / "ZERO2_optimized.yaml"
        if deepspeed_src.exists():
            shutil.copy2(deepspeed_src, deepspeed_dst)
            print(f"📋 复制DeepSpeed配置: {deepspeed_src.name}")
    
    print(f"📋 配置文件已设置: {target_config}")
    
    # 启动Ray训练
    os.system("python3 train_ray.py")

def launch_standard_training(config_path):
    """启动标准训练"""
    # 设置配置文件为当前配置
    config_dir = Path("train_config")
    target_config = config_dir / "trainer_config.yaml"
    
    # 复制配置文件
    import shutil
    shutil.copy2(config_path, target_config)
    
    print(f"📋 配置文件已设置: {target_config}")
    
    # 启动标准训练
    os.system("python3 train_bak.py")

if __name__ == "__main__":
    main()