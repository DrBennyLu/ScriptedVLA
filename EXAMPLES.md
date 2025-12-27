# 使用示例

本文档提供详细的使用示例，帮助您快速上手VLA模型训练。

## 示例1：在LIBERO数据集上训练

### 步骤1：下载数据集

```bash
# 方法1：使用下载脚本
python -m src.data.download_datasets --dataset libero --name libero_spatial

# 方法2：在训练时自动下载
python train_public_datasets.py --dataset libero --dataset-name libero_spatial --download
```

### 步骤2：配置模型

编辑 `config.yaml`，设置数据集类型：

```yaml
data:
  dataset_type: "libero"
  libero:
    dataset_name: "libero_spatial"
    dataset_path: "./data/libero_libero_spatial"
    task_names: null  # null表示使用所有任务
```

### 步骤3：开始训练

```bash
python train_public_datasets.py \
    --dataset libero \
    --dataset-name libero_spatial \
    --config config.yaml
```

### 步骤4：监控训练

训练过程中会：
- 自动保存检查点到 `./checkpoints/`
- 记录日志到 `./logs/`
- 显示训练进度和验证损失

## 示例2：在ACT数据集上训练

### 步骤1：下载数据集

```bash
python -m src.data.download_datasets --dataset act
```

### 步骤2：配置模型

编辑 `config.yaml`：

```yaml
data:
  dataset_type: "act"
  act:
    dataset_path: "./data/act"
    chunk_size: 1  # 动作块大小
```

### 步骤3：开始训练

```bash
python train_public_datasets.py \
    --dataset act \
    --config config.yaml
```

## 示例3：使用自定义数据训练

### 步骤1：准备数据

创建数据目录结构：

```bash
mkdir -p data/train/images
mkdir -p data/val/images
```

创建 `data/train/annotations.json`：

```json
[
  {
    "image_path": "images/image_001.jpg",
    "text": "Pick up the red block",
    "action": [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0]
  },
  {
    "image_path": "images/image_002.jpg",
    "text": "Place the block on the table",
    "action": [0.2, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]
  }
]
```

### 步骤2：配置模型

编辑 `config.yaml`：

```yaml
data:
  dataset_type: "custom"
  train_data_path: "./data/train"
  val_data_path: "./data/val"
```

### 步骤3：开始训练

```bash
python train.py --config config.yaml
```

## 示例4：创建测试数据

如果需要快速测试，可以使用虚拟数据：

```bash
# 创建100个训练样本和20个验证样本
python create_dummy_data.py --num_samples 100 --val_samples 20
```

然后使用自定义数据训练流程进行训练。

## 示例5：从检查点恢复训练

如果训练中断，可以从检查点恢复：

```bash
# 使用公开数据集训练脚本
python train_public_datasets.py \
    --dataset libero \
    --dataset-name libero_spatial \
    --resume ./checkpoints/checkpoint_libero_epoch_50.pt

# 使用自定义数据训练脚本
python train.py \
    --config config.yaml \
    --resume ./checkpoints/checkpoint_epoch_50.pt
```

## 示例6：推理

训练完成后，可以使用模型进行推理：

```bash
python inference.py \
    --config config.yaml \
    --checkpoint ./checkpoints/best_model_libero.pt \
    --image path/to/test_image.jpg \
    --text "Pick up the object"
```

## 示例7：调整模型配置

### 冻结VLM参数（节省显存）

编辑 `config.yaml`：

```yaml
model:
  vlm:
    freeze_vlm: true  # 冻结VLM参数，只训练动作头
```

### 调整动作维度

如果您的动作空间不同，修改配置：

```yaml
model:
  action_head:
    action_dim: 14  # 例如：7维位置 + 7维速度
```

### 使用更小的批次大小

如果显存不足：

```yaml
training:
  batch_size: 4  # 减小批次大小
  gradient_accumulation_steps: 2  # 增加梯度累积以保持有效批次大小
```

## 示例8：使用不同的LIBERO子数据集

LIBERO提供多个子数据集，可以根据需要选择：

```bash
# 空间推理任务
python train_public_datasets.py --dataset libero --dataset-name libero_spatial

# 物体操作任务
python train_public_datasets.py --dataset libero --dataset-name libero_object

# 目标条件任务
python train_public_datasets.py --dataset libero --dataset-name libero_goal

# 100个任务集合
python train_public_datasets.py --dataset libero --dataset-name libero_100
```

## 故障排除

### 问题1：LIBERO数据集下载失败

**解决方案：**
```bash
# 确保安装了libero包
pip install libero

# 如果仍然失败，尝试手动安装
pip install git+https://github.com/PRIOR-LAB/LIBERO.git
```

### 问题2：显存不足

**解决方案：**
1. 减小批次大小：`batch_size: 4`
2. 冻结VLM：`freeze_vlm: true`
3. 使用梯度累积：`gradient_accumulation_steps: 4`
4. 启用混合精度：`fp16: true`

### 问题3：数据加载慢

**解决方案：**
1. 增加数据加载线程：`num_workers: 8`
2. 启用内存固定：`pin_memory: true`
3. 使用SSD存储数据

### 问题4：动作维度不匹配

**解决方案：**
1. 检查数据集中的动作维度
2. 修改 `config.yaml` 中的 `action_head.action_dim`
3. 确保数据预处理正确

## 进阶使用

### 自定义数据集适配器

如果需要添加新的数据集支持，可以参考 `src/data/libero_dataset.py` 和 `src/data/act_dataset.py` 的实现。

### 自定义损失函数

在 `train.py` 中修改损失函数：

```python
# 例如：使用L1损失
criterion = nn.L1Loss()

# 或使用组合损失
criterion = nn.MSELoss() + 0.1 * nn.L1Loss()
```

### 添加数据增强

在数据集类中添加数据增强：

```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    # ... 其他增强
])
```

更多示例和问题，请参考项目文档或提交Issue。

