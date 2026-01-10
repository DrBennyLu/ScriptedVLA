# VLA模型统一输入格式说明

## 变更概述

为了从根本上解决`states`维度不统一的问题，我们统一了整个项目VLA模型的输入格式，移除了`examples`参数，统一使用字典格式输入。

## 统一输入格式

### 输入字典结构

```python
inputs = {
    "images": Union[torch.Tensor, Dict[str, torch.Tensor], List[PIL.Image]],
    "instructions": List[str],
    "states": Optional[torch.Tensor],  # [B, state_dim] 或 [B, 1, state_dim]
    "actions": Optional[torch.Tensor]   # [B, action_horizon, action_dim]（训练时需要）
}
```

### 字段说明

1. **images** (必需):
   - `List[PIL.Image]`: 单相机模式，每个元素是一个PIL图像
   - `List[List[PIL.Image]]`: 多相机模式，每个元素是一个相机图像列表
   - `torch.Tensor [B, C, H, W]`: 单相机张量格式
   - `Dict[str, torch.Tensor]`: 多相机张量格式，键为相机名称

2. **instructions** (必需):
   - `List[str]`: 文本指令列表，长度应与batch size一致

3. **states** (可选):
   - `torch.Tensor`: 机器人状态
   - 支持的维度：
     - `[B, state_dim]`: 标准格式
     - `[B, 1, state_dim]`: 带时间步维度
     - `[state_dim]`: 单样本，会自动扩展为`[1, state_dim]`
   - 注意：所有维度都会被自动规范化处理

4. **actions** (可选，训练时需要):
   - `torch.Tensor [B, action_horizon, action_dim]`: 目标动作序列

## 使用示例

### 训练模式

```python
from src.ScriptedVLA.model.vla_qwen_groot import QwenGR00TVLAModel
from PIL import Image
import torch
import numpy as np

# 创建模型
model = QwenGR00TVLAModel(
    vlm_config={"model_name": "Qwen/Qwen2-VL-2B-Instruct"},
    action_head_config={"hidden_dim": 768, "action_dim": 7},
    use_state=True,
    state_dim=7
)
model.train()

# 准备输入
inputs = {
    "images": [Image.new('RGB', (224, 224), color='red') for _ in range(2)],
    "instructions": ["Pick up the red object.", "Place the blue object."],
    "actions": torch.tensor(np.random.uniform(-1, 1, size=(2, 16, 7))),
    "states": torch.tensor(np.random.uniform(-1, 1, size=(2, 7)))  # [B, state_dim]
}

# 前向传播
outputs = model(inputs=inputs)
loss = outputs["action_loss"]
```

### 推理模式

```python
model.eval()

# 准备输入（不需要actions）
inputs = {
    "images": [Image.new('RGB', (224, 224), color='green')],
    "instructions": ["Move forward."],
    "states": torch.tensor(np.random.uniform(-1, 1, size=(1, 7)))  # [B, state_dim]
}

# 预测动作
with torch.no_grad():
    outputs = model.predict_action(inputs=inputs)
    actions = outputs["normalized_actions"]  # [B, T, action_dim]
```

## States维度自动规范化

模型内部会自动处理`states`的维度，支持以下输入格式：

1. `[state_dim]` → 自动扩展为 `[1, state_dim]`
2. `[B, state_dim]` → 保持不变
3. `[B, 1, state_dim]` → 保持不变
4. `[B, T, state_dim]` → 取第一个时间步，变为 `[B, 1, state_dim]`
5. `[B, 1, 1, state_dim]` → 自动压缩为 `[B, 1, state_dim]`

所有维度都会被统一处理，确保后续处理的一致性。

## 变更影响

### 已更新的文件

1. **src/ScriptedVLA/model/vla_qwen_groot.py**:
   - `forward()`: 移除`examples`参数，统一使用`inputs`字典
   - `predict_action()`: 移除`examples`参数，统一使用`inputs`字典
   - 新增`_normalize_states()`: 统一处理states维度

2. **test/test_vla_qwen_groot.py**:
   - 所有测试函数都已更新为使用新的统一输入格式

### 向后兼容性

**注意**: 此变更**不向后兼容**。所有使用`examples`参数的代码都需要更新。

### 迁移指南

#### 旧代码（已废弃）

```python
# ❌ 旧方式（不再支持）
examples = [
    {
        "image": [Image.new('RGB', (224, 224), color='red')],
        "lang": "Pick up object.",
        "action": np.random.uniform(-1, 1, size=(16, 7)),
        "state": np.random.uniform(-1, 1, size=(1, 7))
    }
]
outputs = model(examples=examples)
```

#### 新代码（推荐）

```python
# ✅ 新方式（统一格式）
inputs = {
    "images": [Image.new('RGB', (224, 224), color='red')],
    "instructions": ["Pick up object."],
    "actions": torch.tensor(np.random.uniform(-1, 1, size=(1, 16, 7))),
    "states": torch.tensor(np.random.uniform(-1, 1, size=(1, 7)))  # [B, state_dim]
}
outputs = model(inputs=inputs)
```

## 优势

1. **维度统一**: 从根本上解决了`states`维度不统一的问题
2. **接口清晰**: 统一的字典格式，字段明确
3. **易于维护**: 减少了多种输入格式的复杂性
4. **类型安全**: 明确的类型提示和验证
5. **自动规范化**: `states`维度自动处理，无需手动转换

## 注意事项

1. **必需字段**: `images`和`instructions`是必需的，缺少会抛出`ValueError`
2. **States维度**: 虽然支持多种维度，但建议使用`[B, state_dim]`格式
3. **Actions格式**: 训练时`actions`应为`[B, action_horizon, action_dim]`
4. **设备一致性**: 所有tensor会自动移动到模型所在设备

