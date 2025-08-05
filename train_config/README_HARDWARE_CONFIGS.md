# ESM2AE 硬件优化配置指南

本文档说明如何在不同硬件配置间切换训练配置，以获得最佳性能。

## 🔬 当前训练配置

当前配置针对快速测试和最大显存利用进行了优化：
- **批次大小**: 
  - 2080 Super: 12 × 4累积 × 512序列 = 24,576 tokens
  - 3080: 32 × 2累积 × 512序列 = 32,768 tokens
- **学习率**: 4e-4 (峰值)，500步warmup (训练步数的1/10)，余弦衰减
- **优化器**: Adam (β1=0.9, β2=0.98, ε=1e-8, weight_decay=0.01)
- **训练步数**: 5000 steps (快速测试，原值500K)
- **序列长度**: 512 tokens
- **显存利用**: 尽可能使用所有可用显存

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

#### 性能参数 (最大显存利用配置)
```yaml
per_device_train_batch_size: 12     # 尽可能使用显存 (原值: 6)
gradient_accumulation_steps: 4      # 减少累积步数，因为批次增大了
max_length: 512                     # 固定序列长度
learning_rate: 4e-4                 # ESM-2论文峰值学习率
max_steps: 5000                     # 快速测试 (原值: 500000)
warmup_steps: 500                   # 训练步数的1/10
lr_scheduler_type: "cosine"         # 余弦衰减
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

#### 性能参数 (最大显存利用配置)
```yaml
per_device_train_batch_size: 32     # 尽可能使用显存 (原值: 20)
gradient_accumulation_steps: 2      # 减少累积步数，因为批次增大了
max_length: 512                     # 固定序列长度
learning_rate: 4e-4                 # ESM-2论文峰值学习率
max_steps: 5000                     # 快速测试 (原值: 500000)
warmup_steps: 500                   # 训练步数的1/10
lr_scheduler_type: "cosine"         # 余弦衰减
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

### 2080 Super 单卡优化

#### 显存使用优化
```bash
# 监控显存使用
nvidia-smi -l 1

# 如果显存不足，可以调整以下参数:
per_device_train_batch_size: 8      # 从12减少到8
gradient_accumulation_steps: 6      # 相应增加累积步数
# 或者减少序列长度:
max_length: 384                     # 从512减少到384
```

#### 学习率调度优化
```bash
# 当前使用余弦衰减，可以调整warmup比例:
warmup_steps: 250                   # 训练步数的1/20 (更快收敛)
warmup_steps: 1000                  # 训练步数的1/5 (更稳定)
```

### 3080 单卡优化

#### 显存使用优化
```bash
# 监控显存使用
nvidia-smi -l 1

# 如果显存不足，可以调整以下参数:
per_device_train_batch_size: 24     # 从32减少到24
gradient_accumulation_steps: 3      # 相应增加累积步数

# 现代特性启用
pip install flash-attn --no-build-isolation
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