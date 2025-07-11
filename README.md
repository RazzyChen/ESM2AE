# ESM2AE: 基于ESM2的蛋白质序列自编码器

ESM2AE是一个利用强大的ESM2蛋白质语言模型构建的自监督学习框架。它通过训练一个自编码器来重建ESM2的表征，从而学习蛋白质序列的紧凑且信息丰富的潜在表示。

## 核心特性

- **ESM2主干**: 利用预训练的ESM2模型 (`facebook/esm2_t33_650M_UR50D`) 作为强大的特征提取器。
- **自编码器结构**: 包含一个编码器和一个解码器，均由线性层和SwiGLU激活函数构成，用于将ESM2的高维输出压缩至低维潜在空间并重建。
- **自监督学习**: 训练过程完全自监督，通过最小化重建表征与原始ESM2表征之间的均方误差（MSE）进行优化。
- **分布式训练**: 支持使用 `Accelerate` 和 `Ray` 进行高效的分布式训练，并集成了 `DeepSpeed` (ZERO-2) 进行显存优化。
- **SimCLR对比损失 (可选)**: 模型中包含了SimCLR的实现 (`model/utils/Simclr.py`)，可以轻松扩展以引入对比学习，进一步提升表征质量。

## 项目结构

```
ESM2AE/
├── model/                # 模型定义
│   ├── backbone/         # 核心模型结构 (ESM2AE)
│   ├── dataloader/       # 数据加载器 (LMDB)
│   └── utils/            # 辅助工具 (模型保存, SimCLR)
├── tools/                # 数据预处理脚本
│   ├── csv2fasta.py      # CSV转FASTA
│   ├── fasta_filter.py   # 按长度过滤FASTA
│   ├── fasta2lmdb.py     # FASTA转LMDB (核心)
│   ├── remove_duplicates.py # 去重
│   └── sequence_analysis.py # 序列分析
├── train_config/         # 训练配置文件
│   ├── trainer_config.yaml # 主训练配置 (Hydra)
│   └── ZERO2.yaml        # DeepSpeed配置
├── train_ray.py          # 使用Ray进行分布式训练的脚本
└── README.md             # 项目说明
```

## 环境准备

建议使用 `uv` 或 `pip` 来管理依赖。

```bash
# 推荐使用uv
pip install uv
uv pip install -r requirements.txt

# 或者使用pip
pip install -r requirements.txt
```

## 数据准备流程

训练数据需要处理成LMDB格式以实现高效读取。

1. **准备FASTA文件**:
   如果你的原始数据是CSV格式，可以使用 `csv2fasta.py` 将其转换为FASTA格式。
   ```bash
   python tools/csv2fasta.py your_data.csv your_data.fasta
   ```

2. **(可选) 过滤序列**:
   为了避免超长序列带来的显存问题，可以过滤掉长度超过特定阈值的序列（默认为1024）。
   ```bash
   python tools/fasta_filter.py your_data.fasta filtered.fasta --max_length 1024
   ```

3. **转换为LMDB格式**:
   这是最关键的一步。使用 `fasta2lmdb.py` 脚本将FASTA文件转换为LMDB数据库。该脚本经过优化，支持多进程处理。
   ```bash
   # 创建输出目录
   mkdir -p dataset

   # 运行转换脚本
   python tools/fasta2lmdb.py \
       --fasta_file filtered.fasta \
       --lmdb_file ./dataset/train_dataset \
       --processes 4  # 根据你的CPU核心数调整
   ```

## 训练模型

项目支持两种分布式训练方式：`Accelerate` 和 `Ray`。

### 使用 Ray + DeepSpeed (推荐)

`train_ray.py` 脚本集成了Ray、Hydra和DeepSpeed，提供了强大的分布式训练能力。

**配置**:
- **`train_config/trainer_config.yaml`**: 配置学习率、批大小、周期数等超参数。
- **`train_config/ZERO2.yaml`**: DeepSpeed配置文件，用于显存优化。
- **`ray`**: 在 `trainer_config.yaml` 中配置Ray的worker数量。

**启动训练**:
```bash
python train_ray.py
```

### 使用 Accelerate + DeepSpeed

如果你更熟悉 `Accelerate`，也可以使用它来启动训练。

**配置**:
首先，配置 `Accelerate`：
```bash
accelerate config
```
根据提示选择 `DEEPSPEED` 并配置相关参数，或者直接使用项目提供的 `train_config/ZERO2.yaml`。

**启动训练**:
```bash
accelerate launch --config_file ./train_config/ZERO2.yaml train_ray.py
```
*注意：`train_ray.py` 脚本同时兼容 `accelerate` 和 `ray` 的启动方式。*

## 监控

训练过程通过 `wandb` 进行监控。请确保在 `train_config/trainer_config.yaml` 中设置好你的 `wandb` 项目名称。

## 工具脚本说明

- **`tools/sequence_analysis.py`**: 分析序列数据集的长度分布、token数量等，并生成图表。
- **`tools/lmdb_read.py`**: 用于检查和验证生成的LMDB数据库。
- **`tools/remove_duplicates.py`**: 从CSV文件中移除重复的序列。