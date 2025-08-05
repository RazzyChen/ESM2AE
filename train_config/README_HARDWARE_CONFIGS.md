# ESM2AE 硬件优化配置指南

本文档说明如何在不同硬件配置间切换训练配置，以获得最佳性能。

## 🔬 ESM-2论文训练参数

根据ESM-2论文，我们的650M参数模型使用以下关键设置：
- **批次大小**: 2M tokens (通过梯度累积实现)
- **学习率**: 4e-4 (峰值)，2000步warmup，然后线性衰减到峰值的1/10
- **优化器**: Adam (β1=0.9, β2=0.98, ε=1e-8, weight_decay=0.01)
- **训练步数**: 500K steps
- **序列长度**: 512 tokens (论文中提到1024，但我们使用512以适应显存)
- **分布式策略**: 650M模型使用标准DDP (不需要FSDP)

## 支持的硬件配置

### 1. RTX 2080 Super 单卡配置 (生产训练)
- **硬件规格**: 1x RTX 2080 Super 8G VRAM + 64G RAM + 12核CPU
- **配置文件**: `train_config_2080super.yaml` + `ZERO2_2080super.yaml`
- **优化重点**: 内存效率、大RAM利用、FP16混合精度

#### 主要特性
- ✅ **FP16混合精度**: 2080 Super不支持BF16，使用FP16
- ✅ **激进内存优化**: 针对8G显存限制
- ✅ **CPU参数卸载**: 利用64G RAM优势
- ✅ **大RAM利用**: 12个数据加载worker，8倍预取因子
- ✅ **12核CPU优化**: 充分利用多核CPU资源

#### 性能参数 (遵循ESM-2论文设置)
```yaml
per_device_train_batch_size: 6      # 512序列长度下8G显存安全批次
gradient_accumulation_steps: 651    # 接近ESM-2的2M tokens批次大小
max_length: 512                     # 固定序列长度
learning_rate: 4e-4                 # ESM-2论文峰值学习率
max_steps: 500000                   # ESM-2论文训练步数
adam_beta2: 0.98                    # ESM-2论文优化器设置
weight_decay: 0.01                  # ESM-2论文权重衰减
```

### 2. RTX 3080 单卡配置 (调试开发)
- **硬件规格**: 1x RTX 3080 10/12G VRAM + 12核CPU
- **配置文件**: `train_config_3080.yaml` + `ZERO2_3080.yaml`
- **优化重点**: 现代GPU特性、快速迭代

#### 主要特性
- ✅ **TF32 + BF16**: 利用3080现代特性
- ✅ **Flash Attention 2**: 高效注意力计算
- ✅ **Fused优化器**: 更快的参数更新
- ✅ **调试友好**: 频繁日志、性能分析
- ✅ **无CPU卸载**: 单卡无需复杂内存管理

#### 性能参数 (遵循ESM-2论文设置)
```yaml
per_device_train_batch_size: 20     # 3080支持更大批次
gradient_accumulation_steps: 195    # 接近ESM-2的2M tokens批次大小
max_length: 512                     # 固定序列长度
learning_rate: 4e-4                 # ESM-2论文峰值学习率
tf32: true                          # 启用TF32加速
bf16: true                          # 更稳定的混合精度
```

## 快速切换配置

### 使用配置切换脚本

```bash
# 查看所有可用配置
python train_config/switch_config.py --list

# 切换到2080 Super配置 (生产训练)
python train_config/switch_config.py --hardware 2080super

# 切换到3080配置 (调试开发)
python train_config/switch_config.py --hardware 3080

# 查看当前配置
python train_config/switch_config.py --current

# 验证配置文件
python train_config/switch_config.py --validate
```

### 手动切换配置

如果不使用脚本，可以手动复制配置文件：

```bash
# 切换到2080 Super配置
cp train_config/train_config_2080super.yaml train_config/trainer_config.yaml
cp train_config/ZERO2_2080super.yaml train_config/ZERO2_optimized.yaml

# 切换到3080配置
cp train_config/train_config_3080.yaml train_config/trainer_config.yaml
cp train_config/ZERO2_3080.yaml train_config/ZERO2_optimized.yaml
```

## 性能对比与建议

### RTX 2080 Super vs RTX 3080 单卡对比

| 指标 | 2080 Super | 3080 | 说明 |
|------|------------|------|------|
| **显存** | 8GB | 10-12GB | 3080显存更大 |
| **内存带宽** | 448 GB/s | 760 GB/s | 3080带宽更高 |
| **计算能力** | 11.15 TFLOPS | 29.77 TFLOPS | 3080计算能力更强 |
| **现代特性** | ❌ | ✅ TF32/BF16/FA2 | 3080支持更多优化 |
| **系统RAM** | 64GB | 标准配置 | 2080配置RAM更大 |

### 使用建议

#### 选择2080 Super配置的场景
- 🎯 **生产训练**: 长时间稳定训练
- 🎯 **大RAM优势**: 利用64G系统内存
- 🎯 **内存密集**: 大批次数据处理
- 🎯 **成本敏感**: 充分利用现有硬件

#### 选择3080配置的场景
- 🎯 **快速原型**: 快速验证想法
- 🎯 **调试开发**: 需要详细性能分析
- 🎯 **实验测试**: 频繁修改模型结构
- 🎯 **现代特性**: 需要最新GPU特性

## 性能调优建议

### 2080 Super 双卡优化

#### 内存优化
```bash
# 监控显存使用
nvidia-smi -l 1

# 如果显存不足，可以调整以下参数:
per_device_train_batch_size: 6      # 减少到6
max_length: 320                     # 进一步减少序列长度
gradient_accumulation_steps: 6      # 相应增加累积步数
```

#### 通信优化
```bash
# 确保NCCL环境变量设置正确
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_IB_DISABLE=1            # 禁用InfiniBand
```

### 3080 单卡优化

#### 现代特性启用
```bash
# 确保Flash Attention安装
pip install flash-attn --no-build-isolation

# 验证TF32支持
python -c "import torch; print(torch.backends.cuda.matmul.allow_tf32)"
```

#### 调试模式
```yaml
# 启用快速开发模式 (在配置文件中)
debug:
  fast_dev_run: true                # 只运行几个batch测试
  overfit_batches: 10               # 过拟合少量数据验证
```

## 故障排除

### 常见问题

#### 1. 显存不足 (OOM)
```bash
# 2080 Super: 减少批次大小
per_device_train_batch_size: 4

# 或增加梯度累积
gradient_accumulation_steps: 8

# 或减少序列长度
max_length: 256
```

#### 2. 跨卡通信失败
```bash
# 检查GPU拓扑
nvidia-smi topo -m

# 设置NCCL调试
export NCCL_DEBUG=WARN
export NCCL_TREE_THRESHOLD=0
```

#### 3. Flash Attention错误
```bash
# 3080: 降级到标准attention
attn_implementation: "eager"

# 或更新Flash Attention
pip install flash-attn --upgrade --no-build-isolation
```

### 性能监控

#### 推荐监控工具
```bash
# GPU监控
nvidia-smi -l 1

# 系统监控  
htop

# 网络监控 (双卡通信)
iftop

# 训练监控
wandb
```

#### 性能基准测试
```bash
# 启动训练并监控性能
python3 train.py --config train_config/train_config_2080super.yaml

# 启用基准测试模式
# 在配置文件中设置: performance.benchmark_mode: true

# 预期性能指标 (序列长度512):
# 2080 Super x2: ~800 tokens/sec
# 3080 x1: ~900 tokens/sec (更高效的架构)
```

## 配置文件详解

### 关键差异对比

| 配置项 | 2080 Super | 3080 | 说明 |
|--------|------------|------|------|
| `tf32` | false | true | 3080支持TF32 |
| `fp16` | true | false | 2080用FP16，3080用BF16 |
| `bf16` | false | true | BF16数值更稳定 |
| `attn_implementation` | "eager" | "flash_attention_2" | 3080支持FA2 |
| `optim` | "adamw_torch" | "adamw_torch_fused" | 3080用fused优化器 |
| `CPU offload` | 启用 | 禁用 | 2080需要CPU卸载 |

### 自定义配置

如需创建自定义配置，可以基于现有配置修改：

```bash
# 复制基础配置
cp train_config/train_config_3080.yaml train_config/my_custom_config.yaml

# 编辑配置
vim train_config/my_custom_config.yaml

# 手动切换
cp train_config/my_custom_config.yaml train_config/trainer_config.yaml
```

---

## 总结

通过使用针对不同硬件优化的配置文件，可以：

1. **最大化硬件利用率**: 每种配置都针对特定硬件特性优化
2. **简化切换流程**: 一键切换不同环境配置
3. **提升训练效率**: 避免通用配置的性能损失
4. **降低调试成本**: 调试配置提供更多诊断信息

建议在开发阶段使用3080配置快速迭代，在生产训练时切换到2080 Super配置以获得更好的成本效益。