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
├── dataset_statistics.py       # 数据集统计和筛选工具
├── download_model.py            # 模型下载脚本
├── evaluate_vlm_capabilities.py # VLM能力测评脚本
├── README.md                    # 项目说明
├── QUICKSTART.md                # 快速开始指南
├── EXAMPLES.md                  # 使用示例
└── src/
    └── ScriptedVLA/            # Python包（符合uv标准结构）
        ├── __init__.py
        ├── model/               # 模型定义
        │   ├── __init__.py
        │   ├── vlm.py          # Qwen VLM模型
        │   ├── action_head.py  # DiT动作头
        │   └── vla.py         # 完整VLA模型
        ├── data/                # 数据处理
        │   ├── __init__.py
        │   ├── dataset.py      # 自定义数据集类
        │   ├── download_datasets.py # 数据集下载工具
        │   ├── libero_dataset.py   # LIBERO数据集适配器
        │   └── act_dataset.py      # ACT数据集适配器
        └── utils/               # 工具函数
            ├── __init__.py
            ├── config.py       # 配置加载
            └── logger.py       # 日志工具
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
python -m ScriptedVLA.data.download_datasets --dataset libero --name libero_spatial

# 或在训练时自动下载
python train_public_datasets.py --dataset libero --dataset-name libero_spatial --download
```

**ACT数据集：**
```bash
# 下载ACT数据集
python -m ScriptedVLA.data.download_datasets --dataset act

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
    "image_paths": {
      "global_img": "images/task_000_ep000_step000_global_img.jpg",
      "left_wrist_img": "images/task_000_ep000_step000_left_wrist_img.jpg"
    },
    "text": "Pick up the red block",
    "state": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    "action": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0],
    "task_name": "task_000",
    "episode_id": 0,
    "step_id": 0
  },
  ...
]
```

**数据层次结构说明：**
- `task_name`: 任务名称（字符串），例如 "task_000", "pick_and_place" 等
- `episode_id`: Episode编号（整数），每个任务下的episode从0开始
- `step_id`: Step编号（整数），每个episode下的step从0开始

这种层次化结构便于：
- 按任务组织数据
- 按episode进行训练和评估
- 跟踪数据来源和上下文

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

### 5. 下载模型

```bash
# 下载Qwen2-VL-2B-Instruct模型
python download_model.py --model Qwen/Qwen2-VL-2B-Instruct

# 或下载其他模型
python download_model.py --model Qwen/Qwen-VL-Chat
```

### 6. 评估VLM能力

```bash
# 运行完整的机器人能力测评
python evaluate_vlm_capabilities.py --model Qwen/Qwen2-VL-2B-Instruct

# 使用配置文件中的模型
python evaluate_vlm_capabilities.py --config config.yaml
```

测评包括：
- **物体识别能力**：识别图像中的物体、颜色、数量等
- **空间感知能力**：理解物体的位置关系、距离、方向等
- **因果推理能力**：根据图文进行动作-结果推理、场景理解、逻辑推理等

### 7. 推理

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
- `train_data_path`: 训练数据路径（自定义数据集），默认 `./dataset/train`
- `val_data_path`: 验证数据路径（自定义数据集），默认 `./dataset/val`
- `cameras.names`: 相机名称列表，例如 `["global_img", "left_wrist_img"]`
- `cameras.num_cameras`: 相机数量
- `robot_state.use_state`: 是否使用机器人状态信息
- `robot_state.state_dim`: 状态维度，默认7
- `action.action_dim`: 动作维度，默认7
- `libero.dataset_name`: LIBERO数据集名称（libero_spatial, libero_object, libero_goal, libero_100）
- `libero.dataset_path`: LIBERO数据集路径
- `act.dataset_path`: ACT数据集路径
- `act.chunk_size`: ACT动作块大小
- `num_workers`: 数据加载线程数

**数据层次结构：**
数据集支持层次化标识：
- `task_name`: 任务名称，用于区分不同任务
- `episode_id`: Episode编号，每个任务可以有多个episode
- `step_id`: Step编号，每个episode包含多个step

训练和评估时会自动统计任务级别的性能指标。

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
python -m ScriptedVLA.data.download_datasets --dataset libero --name libero_spatial

# 在LIBERO上训练
python train_public_datasets.py --dataset libero --dataset-name libero_spatial --download
```

### ACT数据集

ACT (Action Chunking Transformer) 是一个用于机器人操作的数据集，支持动作块预测。

**使用示例：**
```bash
# 下载ACT数据集
python -m ScriptedVLA.data.download_datasets --dataset act

# 在ACT上训练
python train_public_datasets.py --dataset act --download
```

**注意：** 首次使用这些数据集时，可能需要安装额外的依赖：
```bash
# LIBERO需要
pip install libero

# ACT数据集可能需要h5py（已包含在依赖中）
```

## 项目结构说明

本项目采用标准的Python包结构，符合uv和现代Python包管理工具的要求：

- **包名**: `ScriptedVLA`（在 `src/ScriptedVLA/` 目录下）
- **导入方式**: `from ScriptedVLA.model import ...`（安装后可直接导入）
- **安装方式**: `uv pip install -e .`（以可编辑模式安装，代码修改立即生效）

这种结构的优势：
- ✅ 符合PEP 517/518标准
- ✅ 支持uv、pip、poetry等现代包管理工具
- ✅ 便于代码组织和模块化
- ✅ 支持作为库被其他项目引用

## 开发说明

### 添加新功能
1. 模型扩展：在 `src/ScriptedVLA/model/` 中添加新模块
2. 数据处理：在 `src/ScriptedVLA/data/` 中添加数据增强或新数据集
3. 工具函数：在 `src/ScriptedVLA/utils/` 中添加辅助功能

### 导入说明
所有脚本文件使用以下导入方式：
```python
from ScriptedVLA.model import VLAModel
from ScriptedVLA.data import VLADataset, LIBERODataset
from ScriptedVLA.utils import load_config, setup_logger
```

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

**Q: 如何下载和测试Qwen2-VL-2B-Instruct模型？**
A: 
1. 下载模型：`python download_model.py --model Qwen/Qwen2-VL-2B-Instruct`
2. 运行能力测评：`python evaluate_vlm_capabilities.py --model Qwen/Qwen2-VL-2B-Instruct`
3. 测评结果会保存为JSON文件，包含物体识别、空间感知、因果推理等测试结果。

**Q: VLM能力测评包含哪些测试？**
A: 测评脚本包含三类测试：
- **物体识别**：简单物体识别、颜色识别、数量统计
- **空间感知**：位置关系、距离判断、方向判断
- **因果推理**：动作-结果推理、场景理解、逻辑推理

**Q: 数据集的层次结构有什么用？**
A: 层次化结构（task_name, episode_id, step_id）支持：
- 按任务组织和管理数据
- 按episode进行训练和评估
- 任务级别的性能分析（训练时会自动统计）
- 数据来源追踪和调试
- 支持更复杂的数据筛选和分析

**Q: 如何创建层次化的测试数据？**
A: 使用 `create_dummy_data.py` 的层次化参数：
```bash
python create_dummy_data.py \
    --num_tasks 3 \
    --episodes_per_task 5 \
    --steps_per_episode 10 \
    --cameras global_img left_wrist_img
```
这将创建3个任务，每个任务5个episode，每个episode 10个step。

**Q: 如何查看数据集的统计信息？**
A: 使用 `dataset_statistics.py` 脚本：
```bash
# 查看数据集统计
python dataset_statistics.py --data_path ./dataset/train

# 按任务筛选
python dataset_statistics.py --data_path ./dataset/train --task task_000 task_001

# 按episode筛选
python dataset_statistics.py --data_path ./dataset/train --episode 0 1 2
```

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request！

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen-VL) - 视觉语言模型
- [Transformers](https://github.com/huggingface/transformers) - 模型库
- [DiT](https://github.com/facebookresearch/DiT) - Diffusion Transformer架构灵感

