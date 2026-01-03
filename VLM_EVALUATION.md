# VLM能力测评指南

本文档介绍如何使用VLM能力测评脚本来评估Qwen VLM模型的机器人相关能力。

## 快速开始

### 1. 下载模型

首先下载Qwen2-VL-2B-Instruct模型：

```bash
python download_model.py --model Qwen/Qwen2-VL-2B-Instruct
```

或者使用配置文件中的模型：

```bash
python download_model.py --config config.yaml
```

### 2. 运行测评

运行完整的能力测评：

```bash
python evaluate_vlm_capabilities.py --model Qwen/Qwen2-VL-2B-Instruct
```

或使用配置文件：

```bash
python evaluate_vlm_capabilities.py --config config.yaml
```

## 测评内容

### 1. 物体识别能力

测试模型识别图像中物体的能力，包括：

- **简单物体识别**：识别基本形状（圆形、方形、三角形等）
- **颜色识别**：识别和描述图像中的颜色
- **数量统计**：统计图像中物体的数量

**示例测试图像**：
- 包含多种形状和颜色的合成图像
- 用于测试模型的基础视觉理解能力

### 2. 空间感知能力

测试模型对空间关系的理解，包括：

- **位置关系**：理解物体的相对位置（左、右、上、下）
- **距离判断**：判断物体之间的远近关系
- **方向判断**：识别物体的排列方向（水平、垂直等）

**示例测试图像**：
- 包含多个物体，具有明确的空间关系
- 用于测试模型的空间理解能力

### 3. 因果推理能力

测试模型根据图文进行推理的能力，包括：

- **动作-结果推理**：根据图像内容推理动作的可能结果
- **场景理解**：分析场景并描述可能的前因后果
- **逻辑推理**：识别物体排列的规律并预测下一个

**示例测试图像**：
- 包含动作指示或逻辑序列的图像
- 用于测试模型的推理能力

## 输出结果

测评完成后，会生成一个JSON格式的结果文件：

```
vlm_evaluation_results_YYYYMMDD_HHMMSS.json
```

结果文件包含：
- 模型信息（名称、设备、时间戳）
- 每个测试类别的详细结果
- 每个子测试的提示、响应和预期关键词

### 结果示例

```json
{
  "model_name": "Qwen/Qwen2-VL-2B-Instruct",
  "device": "cuda",
  "timestamp": "2024-01-01T12:00:00",
  "tests": [
    {
      "category": "物体识别",
      "results": [
        {
          "test": "简单物体识别",
          "prompt": "请描述图像中有什么物体...",
          "response": "图像中有圆形、方形和三角形...",
          "expected_keywords": ["圆形", "方形", "三角形"]
        }
      ]
    }
  ]
}
```

## 高级用法

### 指定设备

```bash
# 使用CPU
python evaluate_vlm_capabilities.py --model Qwen/Qwen2-VL-2B-Instruct --device cpu

# 使用GPU
python evaluate_vlm_capabilities.py --model Qwen/Qwen2-VL-2B-Instruct --device cuda
```

### 不保存结果

```bash
python evaluate_vlm_capabilities.py --model Qwen/Qwen2-VL-2B-Instruct --no-save
```

### 使用其他模型

```bash
# 使用Qwen-VL-Chat
python evaluate_vlm_capabilities.py --model Qwen/Qwen-VL-Chat

# 使用其他Qwen模型
python evaluate_vlm_capabilities.py --model Qwen/Qwen-VL
```

## 测试图像说明

测评脚本会自动生成测试图像，包括：

1. **物体识别图像**：包含多种形状和颜色的简单图像
2. **空间关系图像**：展示物体位置关系的图像
3. **距离判断图像**：包含近距离和远距离物体的图像
4. **方向判断图像**：展示水平/垂直排列的图像
5. **因果推理图像**：包含动作指示或逻辑序列的图像

所有测试图像都是程序生成的，确保测试的一致性和可重复性。

## 注意事项

1. **首次运行**：首次运行需要下载模型，可能需要较长时间
2. **显存要求**：Qwen2-VL-2B-Instruct需要约4-6GB显存
3. **网络连接**：需要稳定的网络连接以下载模型
4. **HuggingFace访问**：确保可以访问HuggingFace模型库

## 故障排除

### 模型下载失败

```bash
# 检查网络连接
ping huggingface.co

# 使用代理或VPN
export HF_ENDPOINT=https://hf-mirror.com
python download_model.py --model Qwen/Qwen2-VL-2B-Instruct
```

### 显存不足

```bash
# 使用CPU模式
python evaluate_vlm_capabilities.py --model Qwen/Qwen2-VL-2B-Instruct --device cpu
```

### 生成失败

如果文本生成失败，脚本会尝试使用特征提取作为备选方案，并记录错误信息。

## 扩展测评

你可以通过修改 `evaluate_vlm_capabilities.py` 来添加自定义测试：

1. 在 `VLMCapabilityEvaluator` 类中添加新的测试方法
2. 创建对应的测试图像生成函数
3. 在 `run_all_tests` 中调用新测试

示例：

```python
def test_custom_ability(self) -> Dict:
    """自定义测试"""
    test_image = self._create_custom_image()
    prompt = "你的测试提示"
    response = self.generate_text_response(test_image, prompt)
    return {
        "category": "自定义能力",
        "results": [{"test": "自定义测试", "response": response}]
    }
```

## 参考

- [Qwen2-VL文档](https://github.com/QwenLM/Qwen-VL)
- [Transformers文档](https://huggingface.co/docs/transformers)

