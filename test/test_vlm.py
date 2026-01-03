"""
测试QwenVLM模块
测试VLM的基本功能、前向传播、build_qwenvl_inputs等
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from ScriptedVLA.model.vlm import QwenVLM


def test_vlm_initialization():
    """测试VLM初始化"""
    print("=" * 60)
    print("测试1: VLM初始化")
    print("=" * 60)
    
    try:
        vlm = QwenVLM(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            image_size=224,
            max_seq_length=512,
            freeze=False,
            cache_dir="./cache/models"
        )
        print("✓ VLM初始化成功")
        print(f"  Hidden dim: {vlm.get_hidden_dim()}")
        return vlm
    except Exception as e:
        print(f"✗ VLM初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vlm_forward_tensor():
    """测试VLM前向传播（使用tensor输入）"""
    print("\n" + "=" * 60)
    print("测试2: VLM前向传播（Tensor输入）")
    print("=" * 60)
    
    try:
        vlm = QwenVLM(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            cache_dir="./cache/models"
        )
        vlm.eval()
        
        # 创建测试图像
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)  # [B, C, H, W]
        texts = ["Describe this image.", "What objects are in the image?"]
        
        # 前向传播
        with torch.no_grad():
            outputs = vlm(images, texts, return_dict=True, output_hidden_states=True)
        
        # 检查输出
        assert "features" in outputs, "缺少features字段"
        assert "last_hidden_state" in outputs, "缺少last_hidden_state字段"
        assert "hidden_states" in outputs, "缺少hidden_states字段"
        
        features = outputs["features"]
        last_hidden = outputs["last_hidden_state"]
        hidden_states = outputs["hidden_states"]
        
        print(f"✓ 前向传播成功")
        print(f"  features shape: {features.shape}")  # [B, hidden_dim]
        print(f"  last_hidden_state shape: {last_hidden.shape}")  # [B, seq_len, hidden_dim]
        print(f"  hidden_states数量: {len(hidden_states)}")
        print(f"  最后一层hidden_state shape: {hidden_states[-1].shape}")
        
        # 验证形状
        assert features.shape[0] == batch_size, f"Batch size不匹配: {features.shape[0]} != {batch_size}"
        assert last_hidden.shape[0] == batch_size, f"Batch size不匹配: {last_hidden.shape[0]} != {batch_size}"
        
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_build_qwenvl_inputs():
    """测试build_qwenvl_inputs方法"""
    print("\n" + "=" * 60)
    print("测试3: build_qwenvl_inputs方法")
    print("=" * 60)
    
    try:
        vlm = QwenVLM(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            cache_dir="./cache/models"
        )
        
        # 创建测试图像
        batch_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue')
        ]
        instructions = ["Describe this image.", "What do you see?"]
        
        # 构建输入
        inputs = vlm.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        
        print("✓ build_qwenvl_inputs成功")
        print(f"  输入键: {list(inputs.keys())}")
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key} shape: {value.shape}")
            else:
                print(f"  {key}: {type(value)}")
        
        # 验证输入
        assert "input_ids" in inputs or "pixel_values" in inputs, "缺少必要的输入字段"
        
        return True
    except Exception as e:
        print(f"✗ build_qwenvl_inputs失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vlm_forward_with_inputs():
    """测试VLM前向传播（使用build_qwenvl_inputs的输出）"""
    print("\n" + "=" * 60)
    print("测试4: VLM前向传播（使用build_qwenvl_inputs输出）")
    print("=" * 60)
    
    try:
        vlm = QwenVLM(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            cache_dir="./cache/models"
        )
        vlm.eval()
        
        # 创建测试图像
        batch_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue')
        ]
        instructions = ["Describe this image.", "What do you see?"]
        
        # 构建输入
        qwen_inputs = vlm.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        
        # 前向传播
        with torch.no_grad():
            outputs = vlm(
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
                **qwen_inputs
            )
        
        # 检查输出
        assert "last_hidden_state" in outputs, "缺少last_hidden_state字段"
        assert "hidden_states" in outputs, "缺少hidden_states字段"
        
        last_hidden = outputs["last_hidden_state"]
        hidden_states = outputs["hidden_states"]
        
        print("✓ 前向传播成功（使用build_qwenvl_inputs输出）")
        print(f"  last_hidden_state shape: {last_hidden.shape}")
        print(f"  hidden_states数量: {len(hidden_states)}")
        
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vlm_get_hidden_dim():
    """测试get_hidden_dim方法"""
    print("\n" + "=" * 60)
    print("测试5: get_hidden_dim方法")
    print("=" * 60)
    
    try:
        vlm = QwenVLM(
            model_name="Qwen/Qwen2-VL-2B-Instruct",
            cache_dir="./cache/models"
        )
        
        hidden_dim = vlm.get_hidden_dim()
        
        print(f"✓ get_hidden_dim成功")
        print(f"  Hidden dim: {hidden_dim}")
        
        assert isinstance(hidden_dim, int), "Hidden dim应该是整数"
        assert hidden_dim > 0, "Hidden dim应该大于0"
        
        return True
    except Exception as e:
        print(f"✗ get_hidden_dim失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试QwenVLM模块")
    print("=" * 60)
    
    results = []
    
    # 测试1: 初始化
    vlm = test_vlm_initialization()
    results.append(("初始化", vlm is not None))
    
    if vlm is not None:
        # 测试2: 前向传播（tensor）
        results.append(("前向传播（Tensor）", test_vlm_forward_tensor()))
        
        # 测试3: build_qwenvl_inputs
        results.append(("build_qwenvl_inputs", test_build_qwenvl_inputs()))
        
        # 测试4: 前向传播（使用build_qwenvl_inputs）
        results.append(("前向传播（build_qwenvl_inputs）", test_vlm_forward_with_inputs()))
        
        # 测试5: get_hidden_dim
        results.append(("get_hidden_dim", test_vlm_get_hidden_dim()))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    passed = sum(1 for _, result in results if result)
    total = len(results)
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
    print(f"\n总计: {passed}/{total} 测试通过")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

