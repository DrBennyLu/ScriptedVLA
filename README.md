# ScriptedVLA - 基于Qwen VLM和DiT的视觉-语言-动作模型

一个清晰易懂的VLA（Vision-Language-Action）训练和推理项目，基于Qwen开源小VLM模型和Transformer的DiT（Diffusion Transformer）动作头。

## 项目特点

- 🎯 **简单易懂**：代码结构清晰，注释详细，适合学习和研究
- 🔧 **易于配置**：使用YAML配置文件，方便调整超参数
- 🚀 **完整流程**：包含数据加载、模型训练、推理等完整功能
- 📦 **现代化工具**：使用uv进行虚拟环境管理
- 🧩 **模块化设计**：各组件独立，易于扩展和修改

## 项目结构

```
ScriptedVLA/
├── config.yaml                  # 配置文件
├── pyproject.toml               # 项目依赖配置（uv）
├── train.py                     # 训练脚本（自定义数据）
├── train_public_datasets.py     # 训练脚本（公开数据集）
├── inference.py                 # 推理脚本
├── create_dummy_data.py         # 创建测试数据
├── README.md                    # 项目说明
├── QUICKSTART.md                # 快速开始指南
└── src/
    ├── model/                   # 模型定义
    │   ├── __init__.py
    │   ├── vlm.py              # Qwen VLM模型
    │   ├── action_head.py      # DiT动作头
    │   └── vla.py              # 完整VLA模型
    ├── data/                    # 数据处理
    │   ├── __init__.py
    │   ├── dataset.py          # 自定义数据集类
    │   ├── download_datasets.py # 数据集下载工具
    │   ├── libero_dataset.py   # LIBERO数据集适配器
    │   └── act_dataset.py      # ACT数据集适配器
    └── utils/                   # 工具函数
        ├── __init__.py
        ├── config.py           # 配置加载
        └── logger.py           # 日志工具
```

## 快速开始

### 1. 环境设置

使用uv创建虚拟环境并安装依赖：

```bash
# 安装uv（如果还没有）
pip install uv

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

### 2. 准备数据

#### 方式A：使用公开数据集（推荐）

项目支持自动下载和使用公开VLA数据集：

**LIBERO数据集：**
```bash
# 下载LIBERO数据集
python -m src.data.download_datasets --dataset libero --name libero_spatial

# 或在训练时自动下载
python train_public_datasets.py --dataset libero --dataset-name libero_spatial --download
```

**ACT数据集：**
```bash
# 下载ACT数据集
python -m src.data.download_datasets --dataset act

# 或在训练时自动下载
python train_public_datasets.py --dataset act --download
```

#### 方式B：使用自定义数据

数据格式支持两种方式：

**方式1：统一annotations.json**
```
data/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   └── ...
│   └── annotations.json
└── val/
    ├── images/
    └── annotations.json
```

`annotations.json`格式：
```json
[
  {
    "image_path": "images/image_001.jpg",
    "text": "Pick up the red block",
    "action": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
  },
  ...
]
```

**方式2：每个样本一个目录**
```
data/
├── train/
│   ├── sample_001/
│   │   ├── image.jpg
│   │   └── annotation.json
│   └── ...
└── val/
    └── ...
```

### 3. 配置模型

编辑 `config.yaml` 文件，调整模型和训练参数：

```yaml
model:
  vlm:
    model_name: "Qwen/Qwen-VL-Chat"  # 或使用更小的模型
    image_size: 224
    freeze_vlm: false  # 是否冻结VLM参数
  
  action_head:
    hidden_dim: 768
    num_layers: 6
    action_dim: 7  # 动作维度

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 1e-4
  ...
```

### 4. 训练模型

**使用公开数据集训练：**
```bash
# 在LIBERO数据集上训练
python train_public_datasets.py --dataset libero --dataset-name libero_spatial --download

# 在ACT数据集上训练
python train_public_datasets.py --dataset act --download
```

**使用自定义数据训练：**
```bash
python train.py --config config.yaml
```

**从检查点恢复训练：**
```bash
python train.py --config config.yaml --resume ./checkpoints/checkpoint_epoch_50.pt
```

### 5. 推理

```bash
python inference.py \
    --config config.yaml \
    --checkpoint ./checkpoints/best_model.pt \
    --image path/to/image.jpg \
    --text "Pick up the object"
```

## 模型架构

### VLM模块（Qwen）
- 基于Qwen-VL模型进行视觉-语言理解
- 处理图像和文本输入，输出融合特征
- 支持冻结VLM参数以加快训练

### 动作头（DiT）
- 基于Diffusion Transformer架构
- 从VLM特征预测机器人动作
- 包含多层Transformer和位置编码

### 完整VLA模型
- 结合VLM和动作头
- 可选交叉注意力机制增强特征融合
- 端到端训练

## 配置说明

### 模型配置
- `vlm.model_name`: Qwen模型名称
- `vlm.image_size`: 输入图像尺寸
- `vlm.freeze_vlm`: 是否冻结VLM参数
- `action_head.hidden_dim`: Transformer隐藏层维度
- `action_head.num_layers`: Transformer层数
- `action_head.action_dim`: 动作维度（如7维：x, y, z, roll, pitch, yaw, gripper）

### 训练配置
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `num_epochs`: 训练轮数
- `optimizer`: 优化器配置
- `scheduler`: 学习率调度器配置

### 数据配置
- `dataset_type`: 数据集类型，可选 "custom", "libero", "act"
- `train_data_path`: 训练数据路径（自定义数据集）
- `val_data_path`: 验证数据路径（自定义数据集）
- `libero.dataset_name`: LIBERO数据集名称（libero_spatial, libero_object, libero_goal, libero_100）
- `libero.dataset_path`: LIBERO数据集路径
- `act.dataset_path`: ACT数据集路径
- `act.chunk_size`: ACT动作块大小
- `num_workers`: 数据加载线程数

## 公开数据集支持

### LIBERO数据集

LIBERO是一个用于长期机器人操作任务的基准数据集，包含多个子数据集：

- **libero_spatial**: 空间推理任务
- **libero_object**: 物体操作任务
- **libero_goal**: 目标条件任务
- **libero_100**: 100个任务集合

**使用示例：**
```bash
# 下载LIBERO数据集
python -m src.data.download_datasets --dataset libero --name libero_spatial

# 在LIBERO上训练
python train_public_datasets.py --dataset libero --dataset-name libero_spatial --download
```

### ACT数据集

ACT (Action Chunking Transformer) 是一个用于机器人操作的数据集，支持动作块预测。

**使用示例：**
```bash
# 下载ACT数据集
python -m src.data.download_datasets --dataset act

# 在ACT上训练
python train_public_datasets.py --dataset act --download
```

**注意：** 首次使用这些数据集时，可能需要安装额外的依赖：
```bash
# LIBERO需要
pip install libero

# ACT数据集可能需要h5py（已包含在依赖中）
```

## 开发说明

### 添加新功能
1. 模型扩展：在 `src/model/` 中添加新模块
2. 数据处理：在 `src/data/` 中添加数据增强或新数据集
3. 工具函数：在 `src/utils/` 中添加辅助功能

### 代码规范
- 使用类型提示
- 添加文档字符串
- 保持代码简洁清晰

## 常见问题

**Q: 如何调整动作维度？**
A: 修改 `config.yaml` 中的 `action_head.action_dim` 参数。

**Q: 如何冻结VLM参数？**
A: 设置 `vlm.freeze_vlm: true` 在配置文件中。

**Q: 支持哪些图像格式？**
A: 支持PIL/Pillow支持的所有格式（JPEG, PNG等）。

**Q: 如何添加自定义损失函数？**
A: 在 `train.py` 中修改 `criterion` 的定义。

**Q: 如何使用公开数据集？**
A: 使用 `train_public_datasets.py` 脚本，并指定 `--dataset` 参数。首次使用需要添加 `--download` 标志。

**Q: LIBERO数据集下载失败怎么办？**
A: 确保已安装 `libero` 包：`pip install libero`。如果仍有问题，请检查网络连接或参考LIBERO官方文档。

**Q: 如何切换不同的数据集？**
A: 在 `config.yaml` 中设置 `data.dataset_type` 为 "libero" 或 "act"，或使用 `train_public_datasets.py` 的 `--dataset` 参数。

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request！

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen-VL) - 视觉语言模型
- [Transformers](https://github.com/huggingface/transformers) - 模型库
- [DiT](https://github.com/facebookresearch/DiT) - Diffusion Transformer架构灵感

