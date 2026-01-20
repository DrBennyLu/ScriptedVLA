# ScriptedVLA - Vision-Language-Action Model based on Qwen VLM and Flow Matching

A clear and easy-to-understand VLA (Vision-Language-Action) training and inference project, based on Qwen open-source small VLM model and Flow Matching action head. No tricks, no over-encapsulation, no over-modularization design, aiming to provide a clear and easy-to-understand VLA model implementation. Script is all you need.  --- author: @Benny Lu (hitlxg@gmail.com)

---

# ScriptedVLA - 基于Qwen VLM和Flow Matching的视觉-语言-动作模型

一个清晰易懂的VLA（Vision-Language-Action）训练和推理项目，基于Qwen开源小VLM模型和Flow Matching动作头。不玩套路，不做过度封装，不做过度模块化设计，旨在提供一个清晰、易于理解的VLA模型实现。Script is all you need.  --- author: @Benny Lu (hitlxg@gmail.com)

---

## English Version

### Project Features

- 🎯 **Simple and Clear**: Clean code structure with detailed comments, no over-encapsulation, suitable for learning and research
- 🔧 **Easy Configuration**: YAML configuration file for convenient hyperparameter adjustment
- 🚀 **Complete Pipeline**: Includes data loading, model training, inference, and other complete functionalities
- 📦 **Modern Tools**: Uses uv for virtual environment management
- 🧩 **Modular Design**: Independent components, easy to extend and modify
- 🤖 **LeRobot Support**: Native support for LeRobot dataset format (v2.1 and v3.0), compatible with HuggingFace open-source datasets
- 🔄 **Unified Interface**: Unified dictionary format input, automatic state dimension handling, simplified usage
- 🧪 **Complete Testing**: Includes comprehensive test suite to ensure code quality

### Project Structure

```
ScriptedVLA/
├── config.yaml                  # Configuration file
├── pyproject.toml               # Project dependencies (uv)
├── train.py                     # Training script (supports LeRobot datasets)
├── train_public_datasets.py     # Training script for public datasets
├── inference.py                 # Inference script
├── create_dummy_data.py         # Create test data
├── dataset_statistics.py        # Dataset statistics and filtering tools
├── download_model.py            # Model download script
├── analyze_state_dimensions.py  # State dimension analysis tool
├── README.md                    # Project documentation
├── VLM_EVALUATION.md            # VLM capability evaluation guide
├── test/                        # Test directory
│   ├── test_vla_qwen_groot.py   # VLA model tests
│   ├── test_vlm.py              # VLM model tests
│   ├── test_action_head.py      # Action head tests
│   ├── test_lerobot_training.py # LeRobot training tests
│   ├── test_lerobot_dataset_loader.py # LeRobot dataset loader tests
│   ├── test_training.py         # Training pipeline tests
│   ├── test_inference.py        # Inference tests
│   └── evaluate_vlm_capabilities.py # VLM capability evaluation script
└── src/
    └── ScriptedVLA/            # Python package (uv standard structure)
        ├── __init__.py
        ├── model/               # Model definitions
        │   ├── __init__.py
        │   ├── vlm.py          # Qwen VLM model
        │   ├── action_head.py  # Flow Matching action head
        │   └── vla_qwen_groot.py  # Qwen-GR00T VLA model
        ├── data/                # Data processing
        │   ├── __init__.py
        │   ├── dataset.py      # Custom dataset classes
        │   ├── download_datasets.py # Dataset download utilities
        │   ├── libero_dataset.py   # LIBERO dataset adapter
        │   ├── act_dataset.py      # ACT dataset adapter
        │   └── lerobot_dataset_adapter.py # LeRobot dataset adapter
        └── utils/               # Utility functions
            ├── __init__.py
            ├── config.py       # Configuration loading
            ├── logger.py       # Logging utilities
            └── normalization.py # State normalization utilities
```

### Quick Start

#### 1. Environment Setup

Create a virtual environment and install dependencies using uv:

```bash
# Install uv (if not already installed)
pip install uv

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

#### 2. Prepare Data

The project primarily supports **LeRobot dataset format** (v2.1 and v3.0), which is the default and recommended approach.

**Using LeRobot Datasets:**

```bash
# Train with LeRobot dataset (default uses ./dataset/libero_object)
python train.py --config config.yaml

# Specify LeRobot dataset path
python train.py --config config.yaml --dataset_path ./dataset/libero_object

# Train with HuggingFace LeRobot dataset
# Set in config.yaml:
# dataset:
#   local_path: null
#   # Or use HF dataset name directly
```

**Supported LeRobot Datasets:**
- `lerobot/pusht`: PushT dataset
- `k1000dai/libero-object-smolvla`: LIBERO Object dataset (LeRobot format)
- Other HuggingFace LeRobot format datasets

#### 3. Configure Model

Edit `config.yaml` to adjust model and training parameters:

```yaml
model:
  vlm:
    model_name: "Qwen/Qwen2-VL-2B-Instruct"  # Recommended model
    image_size: 448
    freeze_vlm: true  # Freeze VLM parameters
  
  action_head:
    hidden_dim: 1536  # Match VLM output dimension
    num_layers: 6
    num_heads: 12
    action_dim: 7  # Action dimension
    action_horizon: 50  # Action sequence length

dataset:
  local_path: "./dataset/libero_object"
  action_horizon: 50
  image_size: 224
  image_keys:
    - "observation.images.wrist_image"
  state_key: "observation.state"
  action_dim: 7

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 1e-4
  ...
```

#### 4. Train Model

```bash
# Train with LeRobot dataset (default)
python train.py --config config.yaml

# Specify dataset path
python train.py --config config.yaml --dataset_path ./dataset/libero_object

# Set training steps and save interval
python train.py --config config.yaml --max_steps 20000 --save_steps 5000

# Resume from checkpoint
python train.py --config config.yaml --resume ./checkpoints/checkpoint_epoch_50.pt
```

#### 5. Download Models

```bash
# Download Qwen2-VL-2B-Instruct model
python download_model.py --model Qwen/Qwen2-VL-2B-Instruct

# Or download other models
python download_model.py --model Qwen/Qwen-VL-Chat
```

#### 6. Evaluate VLM Capabilities

```bash
# Run complete robot capability evaluation
python test/evaluate_vlm_capabilities.py --model Qwen/Qwen2-VL-2B-Instruct

# Use model from config file
python test/evaluate_vlm_capabilities.py --config config.yaml
```

The evaluation includes:
- **Object Recognition**: Identify objects, colors, quantities in images
- **Spatial Perception**: Understand positional relationships, distances, directions
- **Causal Reasoning**: Action-result reasoning, scene understanding, logical reasoning

#### 7. Inference

```bash
python inference.py \
    --config config.yaml \
    --checkpoint ./checkpoints/best_model.pt \
    --image path/to/image.jpg \
    --text "Pick up the object"
```

### Model Architecture

#### VLM Module (Qwen)
- Based on Qwen-VL model for vision-language understanding
- Processes image and text inputs, outputs fused features
- Supports freezing VLM parameters to speed up training

#### Action Head (Flow Matching)
- Based on Flow Matching architecture
- Predicts robot action sequences (action horizon) from VLM features
- Contains multi-layer Transformer (DiT Block) and positional encoding
- Supports action chunking (predicting future action sequences)
- **Timestep Encoding**: Encodes timesteps into embedding vectors via `TimestepEncoder`
- **Adaptive Normalization**: Supports `AdaLayerNorm`, dynamically adjusting normalization parameters via timestep embeddings

#### Complete VLA Model
- Combines VLM and action head
- Optional cross-attention mechanism for enhanced feature fusion
- End-to-end training

### Configuration Guide

#### Model Configuration
- `vlm.model_name`: Qwen model name
- `vlm.image_size`: Input image size (recommended: 448)
- `vlm.freeze_vlm`: Whether to freeze VLM parameters
- `action_head.hidden_dim`: Transformer hidden dimension (should match VLM output)
- `action_head.num_layers`: Number of Transformer layers
- `action_head.num_heads`: Number of attention heads
- `action_head.action_dim`: Action dimension (e.g., 7D: x, y, z, roll, pitch, yaw, gripper)
- `action_head.action_horizon`: Action sequence length (chunk size)

#### Training Configuration
- `batch_size`: Batch size
- `learning_rate`: Learning rate
- `num_epochs`: Number of training epochs
- `max_steps`: Maximum training steps
- `optimizer`: Optimizer configuration
- `scheduler`: Learning rate scheduler configuration

#### Dataset Configuration
- `dataset.local_path`: Local dataset path (LeRobot format)
- `dataset.action_horizon`: Action sequence length
- `dataset.image_size`: Image size
- `dataset.image_keys`: Image key names from dataset (e.g., `["observation.images.wrist_image"]`)
- `dataset.state_key`: State key name from dataset
- `dataset.action_dim`: Action dimension
- `dataset.task_description.use_batch_task`: Get task description from batch (recommended)
- `dataset.task_description.use_tasks_jsonl`: Get task description from tasks.jsonl (fallback)

### LeRobot Dataset Support

LeRobot is an open-source robot learning dataset format on HuggingFace, supporting v2.1 and v3.0 versions. The project uses LeRobot datasets by default for training.

**LeRobot Dataset Features:**
- Supports Parquet format storage (v3.0) and HDF5 format (v2.1)
- Automatic version detection and compatibility handling
- Supports action chunking (action sequence prediction)
- Includes task descriptions and metadata

**Usage Example:**
```bash
# Train with LeRobot dataset (default)
python train.py --config config.yaml --dataset_path ./dataset/libero_object

# Use HuggingFace dataset
# In config.yaml, set dataset.local_path to null or use HF dataset name
```

### Package Structure

This project uses a standard Python package structure, compatible with uv and modern Python package management tools:

- **Package Name**: `ScriptedVLA` (in `src/ScriptedVLA/` directory)
- **Import Style**: `from ScriptedVLA.model import ...` (after installation)
- **Installation**: `uv pip install -e .` (editable mode, code changes take effect immediately)

**Advantages:**
- ✅ Complies with PEP 517/518 standards
- ✅ Supports uv, pip, poetry and other modern package managers
- ✅ Easy code organization and modularization
- ✅ Can be imported as a library by other projects

### Development Guide

#### Adding New Features
1. **Model Extensions**: Add new modules in `src/ScriptedVLA/model/`
2. **Data Processing**: Add data augmentation or new datasets in `src/ScriptedVLA/data/`
3. **Utilities**: Add helper functions in `src/ScriptedVLA/utils/`

#### Import Style
All script files use the following import style:
```python
from ScriptedVLA.model import QwenGR00TVLAModel
from ScriptedVLA.data import VLADataset, LeRobotDatasetAdapter
from ScriptedVLA.utils import load_config, setup_logger, Normalizer
```

#### Code Standards
- Use type hints
- Add docstrings
- Keep code clean and clear

### Testing

The project includes a comprehensive test suite:

```bash
# Run all tests
pytest test/

# Run specific tests
pytest test/test_vla_qwen_groot.py
pytest test/test_lerobot_training.py
pytest test/test_training.py
pytest test/test_inference.py
```

### Common Questions

**Q: How to adjust action dimension?**  
A: Modify the `action_head.action_dim` parameter in `config.yaml`.

**Q: How to freeze VLM parameters?**  
A: Set `vlm.freeze_vlm: true` in the configuration file.

**Q: What is AdaLayerNorm? How to use it?**  
A: `AdaLayerNorm` is an adaptive layer normalization that dynamically adjusts normalization parameters via timestep embeddings (temb). This can improve Flow Matching performance by allowing the model to adapt normalization based on timestep information.

**Q: What is timestep embedding (temb)?**  
A: Timestep embedding encodes timesteps in Flow Matching into vector representations, implemented via `TimestepEncoder`. When using `ada_norm`, timestep embeddings are passed to `DiTBlock`'s `AdaLayerNorm` to adjust normalization parameters.

**Q: What image formats are supported?**  
A: All formats supported by PIL/Pillow (JPEG, PNG, etc.).

**Q: How to use LeRobot datasets?**  
A:
1. Install dependencies: `pip install lerobot datasets`
2. Prepare dataset (local path or HuggingFace dataset name)
3. Run training: `python train.py --config config.yaml --dataset_path ./dataset/libero_object`
4. Or configure in config.yaml: set `dataset.local_path` and related parameters

**Q: What is the model input format?**  
A: The project uses a unified dictionary format input:
```python
inputs = {
    "images": List[PIL.Image] or List[List[PIL.Image]],
    "instructions": List[str],
    "states": Optional[torch.Tensor],  # [B, state_dim]
    "actions": Optional[torch.Tensor]  # [B, action_horizon, action_dim]
}
```

**Q: How to download and test Qwen2-VL-2B-Instruct model?**  
A:
1. Download model: `python download_model.py --model Qwen/Qwen2-VL-2B-Instruct`
2. Run capability evaluation: `python test/evaluate_vlm_capabilities.py --model Qwen/Qwen2-VL-2B-Instruct`
3. Results are saved as JSON file, containing object recognition, spatial perception, causal reasoning test results.

**Q: What tests are included in VLM capability evaluation?**  
A: The evaluation script includes three types of tests:
- **Object Recognition**: Simple object recognition, color recognition, quantity counting
- **Spatial Perception**: Positional relationships, distance judgment, direction judgment
- **Causal Reasoning**: Action-result reasoning, scene understanding, logical reasoning

**Q: How to run tests?**  
A: The project includes a complete test suite:
```bash
# Run all tests
pytest test/

# Run specific tests
pytest test/test_vla_qwen_groot.py
pytest test/test_lerobot_training.py
```

**Q: What if state dimensions don't match?**  
A: The project implements automatic state dimension normalization. If you encounter dimension issues, check the normalization utilities in `src/ScriptedVLA/utils/normalization.py`.

### License

This project is open source under the MIT License.

### Contributing

Issues and Pull Requests are welcome!

### Related Documentation

- [VLM_EVALUATION.md](VLM_EVALUATION.md) - VLM Capability Evaluation Guide

### Acknowledgments

- [Qwen](https://github.com/QwenLM/Qwen-VL) - Vision Language Model
- [Transformers](https://github.com/huggingface/transformers) - Model Library
- [LeRobot](https://github.com/huggingface/lerobot) - Robot Learning Dataset Format
- [Flow Matching](https://arxiv.org/abs/2210.02747) - Flow Matching Architecture Inspiration

---

## 中文版本

### 项目特点

- 🎯 **简单易懂**：代码结构清晰，注释详细，绝无过度封装，适合学习和研究
- 🔧 **易于配置**：使用YAML配置文件，方便调整超参数
- 🚀 **完整流程**：包含数据加载、模型训练、推理等完整功能
- 📦 **现代化工具**：使用uv进行虚拟环境管理
- 🧩 **模块化设计**：各组件独立，易于扩展和修改
- 🤖 **LeRobot支持**：原生支持LeRobot数据集格式（v2.1和v3.0），兼容HuggingFace开源数据集
- 🔄 **统一接口**：统一的字典格式输入，自动处理状态维度，简化使用流程
- 🧪 **完整测试**：包含完整的测试套件，确保代码质量

### 项目结构

```
ScriptedVLA/
├── config.yaml                  # 配置文件
├── pyproject.toml               # 项目依赖配置（uv）
├── train.py                     # 训练脚本（支持LeRobot数据集）
├── train_public_datasets.py     # 公开数据集训练脚本
├── inference.py                 # 推理脚本
├── create_dummy_data.py         # 创建测试数据
├── dataset_statistics.py        # 数据集统计和筛选工具
├── download_model.py            # 模型下载脚本
├── analyze_state_dimensions.py  # 状态维度分析工具
├── README.md                    # 项目说明文档
├── VLM_EVALUATION.md            # VLM能力测评指南
├── test/                        # 测试目录
│   ├── test_vla_qwen_groot.py   # VLA模型测试
│   ├── test_vlm.py              # VLM模型测试
│   ├── test_action_head.py      # 动作头测试
│   ├── test_lerobot_training.py # LeRobot训练测试
│   ├── test_lerobot_dataset_loader.py # LeRobot数据加载测试
│   ├── test_training.py         # 训练流程测试
│   ├── test_inference.py        # 推理测试
│   └── evaluate_vlm_capabilities.py # VLM能力测评脚本
└── src/
    └── ScriptedVLA/            # Python包（符合uv标准结构）
        ├── __init__.py
        ├── model/               # 模型定义
        │   ├── __init__.py
        │   ├── vlm.py          # Qwen VLM模型
        │   ├── action_head.py  # Flow Matching动作头
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
            ├── logger.py       # 日志工具
            └── normalization.py # 状态归一化工具
```

### 快速开始

#### 1. 环境设置

使用uv创建虚拟环境并安装依赖：

```bash
# 安装uv（如果还没有）
pip install uv

# 创建虚拟环境并安装依赖
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
uv pip install -e .
```

#### 2. 准备数据

项目主要支持 **LeRobot数据集格式**（v2.1和v3.0），这是默认和推荐的方式。

**使用LeRobot数据集：**

```bash
# 使用LeRobot数据集训练（默认使用./dataset/libero_object）
python train.py --config config.yaml

# 指定LeRobot数据集路径
python train.py --config config.yaml --dataset_path ./dataset/libero_object

# 使用HuggingFace LeRobot数据集训练
# 在config.yaml中设置：
# dataset:
#   local_path: null
#   # 或直接使用HF数据集名称
```

**支持的LeRobot数据集：**
- `lerobot/pusht`: PushT数据集
- `k1000dai/libero-object-smolvla`: LIBERO Object数据集（LeRobot格式）
- 其他HuggingFace上的LeRobot格式数据集

#### 3. 配置模型

编辑 `config.yaml` 文件，调整模型和训练参数：

```yaml
model:
  vlm:
    model_name: "Qwen/Qwen2-VL-2B-Instruct"  # 推荐模型
    image_size: 448
    freeze_vlm: true  # 冻结VLM参数
  
  action_head:
    hidden_dim: 1536  # 与VLM输出维度匹配
    num_layers: 6
    num_heads: 12
    action_dim: 7  # 动作维度
    action_horizon: 50  # 动作序列长度

dataset:
  local_path: "./dataset/libero_object"
  action_horizon: 50
  image_size: 224
  image_keys:
    - "observation.images.wrist_image"
  state_key: "observation.state"
  action_dim: 7

training:
  batch_size: 8
  num_epochs: 100
  learning_rate: 1e-4
  ...
```

#### 4. 训练模型

```bash
# 使用LeRobot数据集训练（默认）
python train.py --config config.yaml

# 指定数据集路径
python train.py --config config.yaml --dataset_path ./dataset/libero_object

# 设置训练步数和保存间隔
python train.py --config config.yaml --max_steps 20000 --save_steps 5000

# 从检查点恢复训练
python train.py --config config.yaml --resume ./checkpoints/checkpoint_epoch_50.pt
```

#### 5. 下载模型

```bash
# 下载Qwen2-VL-2B-Instruct模型
python download_model.py --model Qwen/Qwen2-VL-2B-Instruct

# 或下载其他模型
python download_model.py --model Qwen/Qwen-VL-Chat
```

#### 6. 评估VLM能力

```bash
# 运行完整的机器人能力测评
python test/evaluate_vlm_capabilities.py --model Qwen/Qwen2-VL-2B-Instruct

# 使用配置文件中的模型
python test/evaluate_vlm_capabilities.py --config config.yaml
```

测评包括：
- **物体识别能力**：识别图像中的物体、颜色、数量等
- **空间感知能力**：理解物体的位置关系、距离、方向等
- **因果推理能力**：根据图文进行动作-结果推理、场景理解、逻辑推理等

#### 7. 推理

```bash
python inference.py \
    --config config.yaml \
    --checkpoint ./checkpoints/best_model.pt \
    --image path/to/image.jpg \
    --text "Pick up the object"
```

### 模型架构

#### VLM模块（Qwen）
- 基于Qwen-VL模型进行视觉-语言理解
- 处理图像和文本输入，输出融合特征
- 支持冻结VLM参数以加快训练

#### 动作头（Flow Matching）
- 基于Flow Matching架构
- 从VLM特征预测机器人动作序列（action horizon）
- 包含多层Transformer（DiT Block）和位置编码
- 支持动作块预测（预测未来动作序列）
- **时间嵌入**：通过 `TimestepEncoder` 将时间步编码为嵌入向量
- **自适应归一化**：支持 `AdaLayerNorm`，通过时间嵌入动态调整归一化参数

#### 完整VLA模型
- 结合VLM和动作头
- 可选交叉注意力机制增强特征融合
- 端到端训练

### 配置说明

#### 模型配置
- `vlm.model_name`: Qwen模型名称
- `vlm.image_size`: 输入图像尺寸（推荐：448）
- `vlm.freeze_vlm`: 是否冻结VLM参数
- `action_head.hidden_dim`: Transformer隐藏层维度（应与VLM输出匹配）
- `action_head.num_layers`: Transformer层数
- `action_head.num_heads`: 注意力头数
- `action_head.action_dim`: 动作维度（如7维：x, y, z, roll, pitch, yaw, gripper）
- `action_head.action_horizon`: 动作序列长度（chunk大小）

#### 训练配置
- `batch_size`: 批次大小
- `learning_rate`: 学习率
- `num_epochs`: 训练轮数
- `max_steps`: 最大训练步数
- `optimizer`: 优化器配置
- `scheduler`: 学习率调度器配置

#### 数据集配置
- `dataset.local_path`: 本地数据集路径（LeRobot格式）
- `dataset.action_horizon`: 动作序列长度
- `dataset.image_size`: 图像尺寸
- `dataset.image_keys`: 数据集中的图像键名（例如：`["observation.images.wrist_image"]`）
- `dataset.state_key`: 数据集中的状态键名
- `dataset.action_dim`: 动作维度
- `dataset.task_description.use_batch_task`: 从batch获取任务描述（推荐）
- `dataset.task_description.use_tasks_jsonl`: 从tasks.jsonl获取任务描述（备选）

### LeRobot数据集支持

LeRobot是HuggingFace上的开源机器人学习数据集格式，支持v2.1和v3.0版本。项目默认使用LeRobot数据集进行训练。

**LeRobot数据集特点：**
- 支持Parquet格式存储（v3.0）和HDF5格式（v2.1）
- 自动版本检测和兼容性处理
- 支持action chunking（动作序列预测）
- 包含任务描述和元数据

**使用示例：**
```bash
# 使用LeRobot数据集训练（默认）
python train.py --config config.yaml --dataset_path ./dataset/libero_object

# 使用HuggingFace数据集
# 在config.yaml中，设置dataset.local_path为null或使用HF数据集名称
```

### 包结构说明

本项目采用标准的Python包结构，符合uv和现代Python包管理工具的要求：

- **包名**: `ScriptedVLA`（在 `src/ScriptedVLA/` 目录下）
- **导入方式**: `from ScriptedVLA.model import ...`（安装后可直接导入）
- **安装方式**: `uv pip install -e .`（以可编辑模式安装，代码修改立即生效）

**优势：**
- ✅ 符合PEP 517/518标准
- ✅ 支持uv、pip、poetry等现代包管理工具
- ✅ 便于代码组织和模块化
- ✅ 支持作为库被其他项目引用

### 开发说明

#### 添加新功能
1. **模型扩展**：在 `src/ScriptedVLA/model/` 中添加新模块
2. **数据处理**：在 `src/ScriptedVLA/data/` 中添加数据增强或新数据集
3. **工具函数**：在 `src/ScriptedVLA/utils/` 中添加辅助功能

#### 导入说明
所有脚本文件使用以下导入方式：
```python
from ScriptedVLA.model import QwenGR00TVLAModel
from ScriptedVLA.data import VLADataset, LeRobotDatasetAdapter
from ScriptedVLA.utils import load_config, setup_logger, Normalizer
```

#### 代码规范
- 使用类型提示
- 添加文档字符串
- 保持代码简洁清晰

### 测试

项目包含完整的测试套件：

```bash
# 运行所有测试
pytest test/

# 运行特定测试
pytest test/test_vla_qwen_groot.py
pytest test/test_lerobot_training.py
pytest test/test_training.py
pytest test/test_inference.py
```

### 常见问题

**Q: 如何调整动作维度？**  
A: 修改 `config.yaml` 中的 `action_head.action_dim` 参数。

**Q: 如何冻结VLM参数？**  
A: 在配置文件中设置 `vlm.freeze_vlm: true`。

**Q: 什么是 AdaLayerNorm？如何使用？**  
A: `AdaLayerNorm` 是一种自适应层归一化，通过时间嵌入（temb）动态调整归一化参数。这可以让模型根据时间步信息自适应调整归一化，可能提高Flow Matching的性能。

**Q: 什么是时间嵌入（temb）？**  
A: 时间嵌入是将Flow Matching中的时间步编码为向量表示，通过 `TimestepEncoder` 实现。当使用 `ada_norm` 时，时间嵌入会被传递给 `DiTBlock` 的 `AdaLayerNorm`，用于调整归一化参数。

**Q: 支持哪些图像格式？**  
A: 支持PIL/Pillow支持的所有格式（JPEG, PNG等）。

**Q: 如何使用LeRobot数据集？**  
A:
1. 安装依赖：`pip install lerobot datasets`
2. 准备数据集（本地路径或HuggingFace数据集名称）
3. 运行训练：`python train.py --config config.yaml --dataset_path ./dataset/libero_object`
4. 或在config.yaml中配置：设置 `dataset.local_path` 和相关参数

**Q: 模型的输入格式是什么？**  
A: 项目使用统一的字典格式输入：
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
2. 运行能力测评：`python test/evaluate_vlm_capabilities.py --model Qwen/Qwen2-VL-2B-Instruct`
3. 结果会保存为JSON文件，包含物体识别、空间感知、因果推理等测试结果。

**Q: VLM能力测评包含哪些测试？**  
A: 测评脚本包含三类测试：
- **物体识别**：简单物体识别、颜色识别、数量统计
- **空间感知**：位置关系、距离判断、方向判断
- **因果推理**：动作-结果推理、场景理解、逻辑推理

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
A: 项目已实现自动状态维度规范化。如果遇到维度问题，请查看 `src/ScriptedVLA/utils/normalization.py` 中的归一化工具。

## 许可证

本项目基于MIT许可证开源。

## 贡献

欢迎提交Issue和Pull Request！

## 相关文档

- [VLM_EVALUATION.md](VLM_EVALUATION.md) - VLM能力测评指南

## 致谢

- [Qwen](https://github.com/QwenLM/Qwen-VL) - 视觉语言模型
- [Transformers](https://github.com/huggingface/transformers) - 模型库
- [LeRobot](https://github.com/huggingface/lerobot) - 机器人学习数据集格式
- [Flow Matching](https://arxiv.org/abs/2210.02747) - Flow Matching架构灵感

