#!/usr/bin/env python3
"""
配置切换脚本 - ESM2AE训练
支持在不同硬件配置间快速切换训练配置

使用方法:
python train_config/switch_config.py --hardware 2080super
python train_config/switch_config.py --hardware 3080
"""

import argparse
import os
import shutil
import sys
from pathlib import Path

# 配置映射
CONFIG_MAPPING = {
    "2080super": {
        "train_config": "train_config_2080super.yaml",
        "deepspeed_config": "ZERO2_2080super.yaml",
        "description": "单卡RTX 2080 Super 8G VRAM + 64G RAM + 12核CPU",
        "features": [
            "FP16混合精度",
            "激进内存优化",
            "CPU参数卸载",
            "利用64G RAM优势",
            "12核CPU优化"
        ]
    },
    "3080": {
        "train_config": "train_config_3080.yaml", 
        "deepspeed_config": "ZERO2_3080.yaml",
        "description": "单卡RTX 3080 + 12核CPU (调试配置)",
        "features": [
            "TF32 + BF16混合精度",
            "Flash Attention 2",
            "Fused优化器",
            "调试友好配置",
            "12核CPU优化"
        ]
    }
}

def print_banner():
    """打印横幅"""
    print("=" * 60)
    print("  ESM2AE 训练配置切换工具")
    print("  支持硬件: 2080super, 3080")
    print("=" * 60)

def list_configs():
    """列出所有可用配置"""
    print("\n可用配置:")
    print("-" * 40)
    
    for hardware, config in CONFIG_MAPPING.items():
        print(f"\n硬件配置: {hardware}")
        print(f"描述: {config['description']}")
        print("特性:")
        for feature in config['features']:
            print(f"  • {feature}")
        print(f"训练配置: {config['train_config']}")
        print(f"DeepSpeed配置: {config['deepspeed_config']}")

def validate_config_files():
    """验证配置文件是否存在"""
    config_dir = Path(__file__).parent
    missing_files = []
    
    for hardware, config in CONFIG_MAPPING.items():
        train_config_path = config_dir / config['train_config']
        deepspeed_config_path = config_dir / config['deepspeed_config']
        
        if not train_config_path.exists():
            missing_files.append(str(train_config_path))
        if not deepspeed_config_path.exists():
            missing_files.append(str(deepspeed_config_path))
    
    if missing_files:
        print(f"❌ 缺少配置文件:")
        for file in missing_files:
            print(f"   {file}")
        return False
    
    print("✅ 所有配置文件验证通过")
    return True

def switch_config(hardware: str, dry_run: bool = False):
    """切换到指定硬件配置"""
    if hardware not in CONFIG_MAPPING:
        print(f"❌ 不支持的硬件配置: {hardware}")
        print(f"支持的配置: {', '.join(CONFIG_MAPPING.keys())}")
        return False
    
    config = CONFIG_MAPPING[hardware]
    config_dir = Path(__file__).parent
    
    # 源文件路径
    src_train_config = config_dir / config['train_config']
    src_deepspeed_config = config_dir / config['deepspeed_config']
    
    # 目标文件路径
    dst_train_config = config_dir / "trainer_config.yaml"
    dst_deepspeed_config = config_dir / "ZERO2_optimized.yaml"
    
    print(f"\n🔄 切换到硬件配置: {hardware}")
    print(f"描述: {config['description']}")
    
    if dry_run:
        print("\n[DRY RUN] 将执行以下操作:")
        print(f"  复制: {src_train_config.name} -> {dst_train_config.name}")
        print(f"  复制: {src_deepspeed_config.name} -> {dst_deepspeed_config.name}")
        return True
    
    try:
        # 备份现有配置
        if dst_train_config.exists():
            backup_path = dst_train_config.with_suffix('.yaml.backup')
            shutil.copy2(dst_train_config, backup_path)
            print(f"📁 备份训练配置到: {backup_path.name}")
        
        if dst_deepspeed_config.exists():
            backup_path = dst_deepspeed_config.with_suffix('.yaml.backup')
            shutil.copy2(dst_deepspeed_config, backup_path)
            print(f"📁 备份DeepSpeed配置到: {backup_path.name}")
        
        # 复制新配置
        shutil.copy2(src_train_config, dst_train_config)
        shutil.copy2(src_deepspeed_config, dst_deepspeed_config)
        
        print(f"✅ 成功切换到 {hardware} 配置")
        print("\n启用的特性:")
        for feature in config['features']:
            print(f"  • {feature}")
        
        print(f"\n🚀 现在可以使用以下命令开始训练:")
        print(f"   python train_ray.py")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置切换失败: {e}")
        return False

def show_current_config():
    """显示当前配置信息"""
    config_dir = Path(__file__).parent
    current_train_config = config_dir / "trainer_config.yaml"
    
    if not current_train_config.exists():
        print("❌ 未找到当前训练配置文件")
        return
    
    print("\n📋 当前配置信息:")
    
    # 尝试识别当前使用的配置
    current_hardware = None
    for hardware, config in CONFIG_MAPPING.items():
        src_config = config_dir / config['train_config']
        if src_config.exists():
            # 简单的文件大小比较来识别配置
            if abs(current_train_config.stat().st_size - src_config.stat().st_size) < 100:
                current_hardware = hardware
                break
    
    if current_hardware:
        config = CONFIG_MAPPING[current_hardware]
        print(f"硬件配置: {current_hardware}")
        print(f"描述: {config['description']}")
        print("特性:")
        for feature in config['features']:
            print(f"  • {feature}")
    else:
        print("无法识别当前配置，可能是自定义配置")
    
    print(f"配置文件: {current_train_config}")

def main():
    parser = argparse.ArgumentParser(
        description="ESM2AE训练配置切换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --list                    # 列出所有可用配置
  %(prog)s --hardware 2080super      # 切换到2080 Super配置
  %(prog)s --hardware 3080           # 切换到3080配置
  %(prog)s --current                 # 显示当前配置
  %(prog)s --validate                # 验证配置文件
        """
    )
    
    parser.add_argument(
        "--hardware",
        choices=list(CONFIG_MAPPING.keys()),
        help="目标硬件配置"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="列出所有可用配置"
    )
    
    parser.add_argument(
        "--current",
        action="store_true", 
        help="显示当前配置信息"
    )
    
    parser.add_argument(
        "--validate",
        action="store_true",
        help="验证配置文件是否存在"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要执行的操作，不实际执行"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    if args.list:
        list_configs()
    elif args.current:
        show_current_config()
    elif args.validate:
        validate_config_files()
    elif args.hardware:
        if validate_config_files():
            switch_config(args.hardware, args.dry_run)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()