# ScriptedVLA - 基于Qwen VLM和Flow Matching的视觉-语言-动作模型

一个清晰易懂的VLA（Vision-Language-Action）训练和推理项目，基于Qwen开源小VLM模型和Flow Matching动作头。不玩套路，不做过度封装，不做过度模块化设计，旨在提供一个清晰、易于理解的VLA模型实现。Script is all you need.  --- author: @Benny Lu (hitlxg@gmail.com)


## 项目特点

- 🎯 **简单易懂**：代码结构清晰，注释详细，绝无过度封装，适合学习和研究
- 🔧 **易于配置**：使用YAML配置文件，方便调整超参数
- 🚀 **完整流程**：包含数据加载、模型训练、推理等完整功能
- 📦 **现代化工具**：使用uv进行虚拟环境管理
- 🧩 **模块化设计**：各组件独立，易于扩展和修改
- 🤖 **LeRobot支持**：原生支持LeRobot数据集格式（v2.1和v3.0），兼容HuggingFace开源数据集
- 🔄 **统一接口**：统一的字典格式输入，自动处理状态维度，简化使用流程
- 🧪 **完整测试**：包含完整的测试套件，确保代码质量

## 项目结构

```
ScriptedVLA/
├── config.yaml                  # 配置文件
├── pyproject.toml               # 项目依赖配置（uv）
├── train.py                     # 训练脚本（支持LeRobot、自定义数据等）
├── train_public_datasets.py     # 训练脚本（公开数据集）
├── inference.py                 # 推理脚本
├── create_dummy_data.py         # 创建测试数据
├── dataset_statistics.py        # 数据集统计和筛选工具
├── download_model.py            # 模型下载脚本
├── analyze_state_dimensions.py  # 状态维度分析工具
├── README.md                    # 项目说明
├── QUICKSTART.md                # 快速开始指南
├── EXAMPLES.md                  # 使用示例
├── CHANGELOG.md                 # 更新日志
├── LEROBOT_VERSION_SOLUTION.md  # LeRobot版本解决方案
├── UNIFIED_INPUT_FORMAT.md      # 统一输入格式说明
├── STATE_DIMENSION_ANALYSIS.md  # 状态维度分析文档
├── VLM_EVALUATION.md            # VLM能力测评指南
├── test/                        # 测试目录
│   ├── test_vla_qwen_groot.py   # VLA模型测试
│   ├── test_vlm.py              # VLM模型测试
│   ├── test_action_head.py      # 动作头测试
│   ├── test_lerobot_training.py # LeRobot训练测试
│   ├── test_lerobot_dataset_loader.py # LeRobot数据加载测试
│   ├── test_training.py         # 训练流程测试
│   └── evaluate_vlm_capabilities.py # VLM能力测评脚本
└── src/
    └── ScriptedVLA/            # Python包（符合uv标准结构）
        ├── __init__.py
        ├── model/               # 模型定义
        │   ├── __init__.py
        │   ├── vlm.py          # Qwen VLM模型
        │   ├── action_head.py  # Flow Matching动作头（包含DiT Block、AdaLayerNorm、TimestepEncoder）

        │   └── vla_qwen_groot.py  # Qwen-GR00T VLA模型
        ├── data/                # 数据处理
        │   ├── __init__.py
        │   ├── dataset.py      # 自定义数据集类
        │   ├── download_datasets.py # 数据集下载工具
        │   ├── libero_dataset.py   # LIBERO数据集适配器
        │   ├── act_dataset.py      # ACT数据集适配器
        │   └── lerobot_dataset_adapter.py # LeRobot数据集适配器
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
    norm_type: "layer_norm"  # 归一化类型：'layer_norm' 或 'ada_norm'
    norm_elementwise_affine: false  # 是否使用元素级仿射变换
    norm_eps: 1e-5  # 归一化的epsilon值
    compute_dtype: "float32"  # 计算数据类型

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 1e-4
  ...
```

### 4. 训练模型

**使用LeRobot数据集训练（推荐，默认方式）：**
```bash
# 使用LeRobot数据集训练（默认使用./dataset/libero_object）
python train.py --config config.yaml

# 指定LeRobot数据集路径
python train.py --config config.yaml --dataset_path ./dataset/libero_object

# 设置最大训练步数和保存间隔
python train.py --config config.yaml --max_steps 20000 --save_steps 5000
```

**使用公开数据集训练：**
```bash
# 在LIBERO数据集上训练
python train_public_datasets.py --dataset libero --dataset-name libero_spatial --download

# 在ACT数据集上训练
python train_public_datasets.py --dataset act --download
```

**使用自定义数据训练：**
```bash
# 使用--no_lerobot参数禁用LeRobot数据集，使用原有训练逻辑
python train.py --config config.yaml --no_lerobot
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

### 动作头（Flow Matching）
- 基于Flow Matching架构
- 从VLM特征预测机器人动作序列（action horizon）
- 包含多层Transformer（DiT Block）和位置编码
- 支持动作块预测（action chunking）
- **时间嵌入支持**：通过 `TimestepEncoder` 将时间步编码为嵌入向量
- **自适应归一化**：支持 `AdaLayerNorm`，通过时间嵌入动态调整归一化参数
- **灵活的归一化类型**：可选择 `layer_norm`（标准层归一化）或 `ada_norm`（自适应归一化）

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
- `action_head.norm_type`: 归一化类型，可选 `"layer_norm"`（默认）或 `"ada_norm"`（自适应归一化，使用时间嵌入）
- `action_head.norm_elementwise_affine`: 是否使用元素级仿射变换（默认 `false`）
- `action_head.norm_eps`: 归一化的epsilon值（默认 `1e-5`）
- `action_head.compute_dtype`: 计算数据类型（默认 `float32`）

### 训练配置
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `num_epochs`: 训练轮数
- `optimizer`: 优化器配置
- `scheduler`: 学习率调度器配置

### 数据配置
- `dataset_type`: 数据集类型，可选 "custom", "libero", "act", "lerobot"
- `train_data_path`: 训练数据路径（自定义数据集），默认 `./dataset/train`
- `val_data_path`: 验证数据路径（自定义数据集），默认 `./dataset/val`
- `cameras.names`: 相机名称列表，例如 `["global_img", "left_wrist_img"]`
- `cameras.num_cameras`: 相机数量
- `robot_state.use_state`: 是否使用机器人状态信息
- `robot_state.state_dim`: 状态维度，默认7
- `action.action_dim`: 动作维度，默认7
- `action_head.action_horizon`: 动作序列长度（action chunk大小），默认11
- `libero.dataset_name`: LIBERO数据集名称（libero_spatial, libero_object, libero_goal, libero_100）
- `libero.dataset_path`: LIBERO数据集路径
- `act.dataset_path`: ACT数据集路径
- `act.chunk_size`: ACT动作块大小
- `lerobot.dataset_path`: LeRobot数据集路径（可以是HF数据集名称或本地路径）
- `lerobot.camera_names`: LeRobot数据集中的相机名称列表
- `lerobot.action_horizon`: LeRobot动作序列长度
- `lerobot.pad_action_chunk`: 是否填充动作块
- `num_workers`: 数据加载线程数

**数据层次结构：**
数据集支持层次化标识：
- `task_name`: 任务名称，用于区分不同任务
- `episode_id`: Episode编号，每个任务可以有多个episode
- `step_id`: Step编号，每个episode包含多个step

训练和评估时会自动统计任务级别的性能指标。

## 公开数据集支持

### LeRobot数据集（推荐）

LeRobot是HuggingFace上的开源机器人学习数据集格式，支持v2.1和v3.0版本。项目默认使用LeRobot数据集进行训练。

**支持的LeRobot数据集：**
- `lerobot/pusht`: PushT数据集
- `k1000dai/libero-object-smolvla`: LIBERO Object数据集（LeRobot格式）
- 其他HuggingFace上的LeRobot格式数据集

**使用示例：**
```bash
# 使用LeRobot数据集训练（默认方式）
python train.py --config config.yaml --dataset_path ./dataset/libero_object

# 从HuggingFace加载数据集
# 在config.yaml中设置：
# data:
#   dataset_type: "lerobot"
#   lerobot:
#     dataset_path: "lerobot/pusht"
```

**LeRobot数据集特点：**
- 支持Parquet格式存储（v3.0）和HDF5格式（v2.1）
- 自动版本检测和兼容性处理
- 支持action chunking（动作序列预测）
- 包含任务描述和元数据


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
from ScriptedVLA.model import QwenGR00TVLAModel
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

**Q: 什么是 AdaLayerNorm？如何使用？**
A: `AdaLayerNorm` 是一种自适应层归一化，通过时间嵌入（temb）动态调整归一化参数。要使用它，在配置文件中设置 `action_head.norm_type: "ada_norm"`。这可以让模型根据时间步信息自适应调整归一化，可能提高Flow Matching的性能。

**Q: 时间嵌入（temb）是什么？**
A: 时间嵌入是将Flow Matching中的时间步编码为向量表示，通过 `TimestepEncoder` 实现。当使用 `ada_norm` 时，时间嵌入会被传递给 `DiTBlock` 的 `AdaLayerNorm`，用于调整归一化参数。

**Q: 支持哪些图像格式？**
A: 支持PIL/Pillow支持的所有格式（JPEG, PNG等）。

**Q: 如何添加自定义损失函数？**
A: 在 `train.py` 中修改 `criterion` 的定义。

**Q: 如何使用公开数据集？**
A: 使用 `train_public_datasets.py` 脚本，并指定 `--dataset` 参数。首次使用需要添加 `--download` 标志。


**Q: 如何切换不同的数据集？**
A: 在 `config.yaml` 中设置 `data.dataset_type` 为 "lerobot", "libero" 或 "act"。默认情况下，`train.py` 会使用LeRobot数据集，可以使用 `--no_lerobot` 参数禁用。

**Q: 如何使用LeRobot数据集？**
A: 
1. 安装依赖：`pip install lerobot datasets`
2. 准备数据集（本地路径或HuggingFace数据集名称）
3. 运行训练：`python train.py --config config.yaml --dataset_path ./dataset/libero_object`
4. 或在config.yaml中配置：设置 `data.dataset_type: "lerobot"` 和 `data.lerobot.dataset_path`

**Q: LeRobot数据集版本兼容性如何？**
A: 项目自动支持LeRobot v2.1格式。
详见 `LEROBOT_VERSION_SOLUTION.md`。

**Q: 模型的输入格式是什么？**
A: 项目使用统一的字典格式输入，移除了`examples`参数。详见 `UNIFIED_INPUT_FORMAT.md`：
```python
inputs = {
    "images": List[PIL.Image] or List[List[PIL.Image]],
    "instructions": List[str],
    "states": Optional[torch.Tensor],  # [B, state_dim]
    "actions": Optional[torch.Tensor]  # [B, action_horizon, action_dim]
}
```

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

**Q: 如何运行测试？**
A: 项目包含完整的测试套件：
```bash
# 运行所有测试
pytest test/

# 运行特定测试
pytest test/test_vla_qwen_groot.py
pytest test/test_lerobot_training.py
```

**Q: 状态维度不匹配怎么办？**
A: 项目已实现自动状态维度规范化。如果遇到维度问题，请参考 `STATE_DIMENSION_ANALYSIS.md` 了解详细说明和解决方案。

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request！

## 相关文档

- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [EXAMPLES.md](EXAMPLES.md) - 使用示例
- [CHANGELOG.md](CHANGELOG.md) - 更新日志
- [UNIFIED_INPUT_FORMAT.md](UNIFIED_INPUT_FORMAT.md) - 统一输入格式说明
- [LEROBOT_VERSION_SOLUTION.md](LEROBOT_VERSION_SOLUTION.md) - LeRobot版本解决方案
- [STATE_DIMENSION_ANALYSIS.md](STATE_DIMENSION_ANALYSIS.md) - 状态维度分析
- [VLM_EVALUATION.md](VLM_EVALUATION.md) - VLM能力测评指南

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen-VL) - 视觉语言模型
- [Transformers](https://github.com/huggingface/transformers) - 模型库
- [LeRobot](https://github.com/huggingface/lerobot) - 机器人学习数据集格式
- [Flow Matching](https://arxiv.org/abs/2210.02747) - Flow Matching架构灵感

