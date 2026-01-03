# 快速开始指南

## 1. 环境设置（使用uv）

```bash
# 安装uv（如果还没有）
pip install uv

# 创建虚拟环境
uv venv

# 激活虚拟环境
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Windows CMD:
.venv\Scripts\activate.bat
# Linux/Mac:
source .venv/bin/activate

# 安装依赖
uv pip install -e .
```

## 2. 准备数据

### 方式A：使用公开数据集（推荐）

**LIBERO数据集：**
```bash
# 下载LIBERO数据集
python -m ScriptedVLA.data.download_datasets --dataset libero --name libero_spatial
```

**ACT数据集：**
```bash
# 下载ACT数据集
python -m ScriptedVLA.data.download_datasets --dataset act
```

### 方式B：创建测试数据

```bash
# 创建虚拟数据集用于测试（简单模式）
python create_dummy_data.py --num_samples 100 --val_samples 20

# 创建层次化数据集（推荐）
python create_dummy_data.py \
    --num_tasks 3 \
    --episodes_per_task 5 \
    --steps_per_episode 10 \
    --cameras global_img left_wrist_img
```

这将创建：
- `./dataset/train/` - 训练数据（包含task_name, episode_id, step_id）
- `./dataset/val/` - 验证数据

数据层次结构：
- 任务 (task_name): 例如 "task_000", "task_001"
- Episode (episode_id): 每个任务下的episode编号
- Step (step_id): 每个episode下的step编号

## 3. 配置模型

编辑 `config.yaml` 文件，根据需要调整参数：

```yaml
model:
  vlm:
    model_name: "Qwen/Qwen-VL-Chat"  # 或更小的模型
    freeze_vlm: false  # 如果显存不足，可以设为true
  
  action_head:
    action_dim: 7  # 根据你的任务调整动作维度
```

## 4. 训练模型

### 使用公开数据集训练

**LIBERO数据集：**
```bash
python train_public_datasets.py \
    --dataset libero \
    --dataset-name libero_spatial \
    --download
```

**ACT数据集：**
```bash
python train_public_datasets.py \
    --dataset act \
    --download
```

### 使用自定义数据训练

```bash
python train.py --config config.yaml
```

训练过程中会：
- 自动保存检查点到 `./checkpoints/`
- 记录日志到 `./logs/`
- 显示训练进度和损失

## 5. 推理

```bash
python inference.py \
    --config config.yaml \
    --checkpoint ./checkpoints/best_model.pt \
    --image path/to/your/image.jpg \
    --text "Pick up the red block"
```

## 常见问题

### Q: 显存不足怎么办？
A: 
1. 减小 `batch_size`（在config.yaml中）
2. 设置 `freeze_vlm: true` 冻结VLM参数
3. 使用更小的Qwen模型
4. 启用梯度累积：`gradient_accumulation_steps: 4`

### Q: 如何添加自己的数据？
A: 
1. 准备数据目录结构（参考README.md）
2. 修改 `config.yaml` 中的 `data.train_data_path` 和 `data.val_data_path`
3. 确保数据格式正确（JSON格式，包含image_path, text, action）

### Q: 如何修改动作维度？
A: 在 `config.yaml` 中修改 `model.action_head.action_dim`，并确保数据中的action维度匹配。

### Q: 训练很慢怎么办？
A:
1. 使用GPU（确保CUDA可用）
2. 增加 `num_workers`（数据加载线程数）
3. 使用混合精度训练（设置 `fp16: true` 或 `bf16: true`）

## 下一步

- 阅读 `README.md` 了解详细文档
- 查看 `EXAMPLES.md` 了解更多使用示例
- 查看 `src/ScriptedVLA/` 目录了解代码结构
- 根据你的具体任务调整模型架构和训练参数

## 公开数据集说明

### LIBERO数据集

LIBERO是一个用于长期机器人操作任务的基准数据集，包含：
- **libero_spatial**: 空间推理任务（推荐入门）
- **libero_object**: 物体操作任务
- **libero_goal**: 目标条件任务
- **libero_100**: 100个任务集合

首次使用需要安装：
```bash
pip install libero
```

### ACT数据集

ACT (Action Chunking Transformer) 数据集用于机器人操作任务，支持动作块预测。

更多信息请参考 `EXAMPLES.md` 中的详细示例。

