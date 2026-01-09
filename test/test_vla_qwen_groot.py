"""
测试QwenGR00TVLAModel模块
测试Qwen-GR00T架构的VLA模型
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from ScriptedVLA.model.vla_qwen_groot import QwenGR00TVLAModel


def test_qwen_groot_initialization():
    """测试QwenGR00TVLAModel初始化"""
    print("=" * 60)
    print("测试1: QwenGR00TVLAModel初始化")
    print("=" * 60)
    
    try:
        vlm_config = {
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "image_size": 224,
            "max_seq_length": 512,
            "freeze_vlm": True,
            "cache_dir": "./cache/models"
        }
        
        action_head_config = {
            "hidden_dim": 1536,
            "num_layers": 2,  # 使用较少的层数以加快测试
            "num_heads": 8,
            "mlp_ratio": 4.0,
            "action_dim": 7,
            "action_horizon": 4,
            "dropout": 0.1,
            "use_cross_attention": True,
            "num_target_vision_tokens": 32,
            "max_seq_len": 1024,
            "add_pos_embed": True,
            "num_inference_timesteps": 5  # 使用较少的步数以加快测试
        }
        
        model = QwenGR00TVLAModel(
            vlm_config=vlm_config,
            action_head_config=action_head_config,
            camera_names=["global_img"],
            use_state=True,
            state_dim=7,
            future_action_window_size=3,
            past_action_window_size=0
        )
        
        print("✓ QwenGR00TVLAModel初始化成功")
        print(f"  Future action window size: {model.future_action_window_size}")
        print(f"  Past action window size: {model.past_action_window_size}")
        print(f"  Chunk len: {model.chunk_len}")
        return model
    except Exception as e:
        print(f"✗ QwenGR00TVLAModel初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_qwen_groot_forward_examples():
    """测试QwenGR00T前向传播（使用examples格式）"""
    print("\n" + "=" * 60)
    print("测试2: QwenGR00T前向传播（examples格式，训练模式）")
    print("=" * 60)
    
    try:
        vlm_config = {
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "cache_dir": "./cache/models"
        }
        
        action_head_config = {
            "hidden_dim": 1536,
            "num_layers": 2,
            "num_heads": 8,
            "action_dim": 7,
            "action_horizon": 4,
            "num_inference_timesteps": 5
        }
        
        model = QwenGR00TVLAModel(
            vlm_config=vlm_config,
            action_head_config=action_head_config,
            use_state=True,
            state_dim=7,
            future_action_window_size=3
        )
        model.train()
        
        # 创建examples格式的数据
        batch_size = 2
        examples = []
        for i in range(batch_size):
            example = {
                "image": [Image.new('RGB', (224, 224), color='red')],
                "lang": f"Pick up object {i}.",
                "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32),
                "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32)
            }
            examples.append(example)
        
        # 前向传播
        outputs = model(examples=examples)
        
        print("✓ 前向传播成功（examples格式，训练模式）")
        print(f"  Examples数量: {len(examples)}")
        print(f"  每个example的键: {list(examples[0].keys())}")
        
        # 检查输出
        assert "action_loss" in outputs, "缺少action_loss字段"
        print(f"  输出action_loss: {outputs['action_loss'].item():.4f}")
        
        assert isinstance(outputs["action_loss"], torch.Tensor), "action_loss应该是tensor"
        assert outputs["action_loss"].dim() == 0, "action_loss应该是标量"
        
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qwen_groot_forward_images_texts():
    """测试QwenGR00T前向传播（使用images和texts格式）"""
    print("\n" + "=" * 60)
    print("测试3: QwenGR00T前向传播（images/texts格式）")
    print("=" * 60)
    
    try:
        vlm_config = {
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "cache_dir": "./cache/models"
        }
        
        action_head_config = {
            "hidden_dim": 768,
            "num_layers": 2,
            "num_heads": 8,
            "action_dim": 7,
            "action_horizon": 4,
            "num_inference_timesteps": 5
        }
        
        model = QwenGR00TVLAModel(
            vlm_config=vlm_config,
            action_head_config=action_head_config,
            use_state=True,
            state_dim=7,
            future_action_window_size=3
        )
        model.train()
        
        # 使用images和texts格式
        batch_images = [
            Image.new('RGB', (224, 224), color='red'),
            Image.new('RGB', (224, 224), color='blue')
        ]
        instructions = ["Pick up the red object.", "Place the blue object."]
        actions = torch.tensor(
            np.random.uniform(-1, 1, size=(2, 16, 7)).astype(np.float32)
        )
        states = torch.tensor(
            np.random.uniform(-1, 1, size=(2, 1, 7)).astype(np.float32)
        )
        
        # 前向传播
        outputs = model(
            images=batch_images,
            texts=instructions,
            actions=actions,
            states=states
        )
        
        print("✓ 前向传播成功（images/texts格式）")
        print(f"  输入images数量: {len(batch_images)}")
        print(f"  输入instructions数量: {len(instructions)}")
        print(f"  输入actions shape: {actions.shape}")
        print(f"  输入states shape: {states.shape}")
        
        if "action_loss" in outputs:
            print(f"  输出action_loss: {outputs['action_loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qwen_groot_predict_action():
    """测试QwenGR00T推理模式"""
    print("\n" + "=" * 60)
    print("测试4: QwenGR00T推理模式（predict_action）")
    print("=" * 60)
    
    try:
        vlm_config = {
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "cache_dir": "./cache/models"
        }
        
        action_head_config = {
            "hidden_dim": 768,
            "num_layers": 2,
            "num_heads": 8,
            "action_dim": 7,
            "action_horizon": 4,
            "num_inference_timesteps": 5
        }
        
        model = QwenGR00TVLAModel(
            vlm_config=vlm_config,
            action_head_config=action_head_config,
            use_state=True,
            state_dim=7,
            future_action_window_size=3
        )
        model.eval()
        
        # 创建examples格式的数据
        examples = [
            {
                "image": [Image.new('RGB', (224, 224), color='red')],
                "lang": "Pick up the red object.",
                "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32)
            },
            {
                "image": [Image.new('RGB', (224, 224), color='blue')],
                "lang": "Place the blue object.",
                "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32)
            }
        ]
        
        # 推理模式
        with torch.no_grad():
            outputs = model.predict_action(examples=examples)
        
        print("✓ 推理模式成功")
        print(f"  输入examples数量: {len(examples)}")
        
        # 检查输出
        assert "normalized_actions" in outputs, "缺少normalized_actions字段"
        print(f"  输出normalized_actions shape: {outputs['normalized_actions'].shape}")
        print(f"  输出normalized_actions dtype: {outputs['normalized_actions'].dtype}")
        
        # 验证形状
        assert outputs["normalized_actions"].shape[0] == len(examples), \
            f"Batch size不匹配: {outputs['normalized_actions'].shape[0]} != {len(examples)}"
        
        return True
    except Exception as e:
        print(f"✗ 推理模式失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qwen_groot_without_state():
    """测试不使用状态的QwenGR00T模型"""
    print("\n" + "=" * 60)
    print("测试5: QwenGR00T模型（不使用状态）")
    print("=" * 60)
    
    try:
        vlm_config = {
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "cache_dir": "./cache/models"
        }
        
        action_head_config = {
            "hidden_dim": 768,
            "num_layers": 2,
            "num_heads": 8,
            "action_dim": 7,
            "action_horizon": 4,
            "num_inference_timesteps": 5
        }
        
        model = QwenGR00TVLAModel(
            vlm_config=vlm_config,
            action_head_config=action_head_config,
            use_state=False,  # 不使用状态
            state_dim=7,
            future_action_window_size=3
        )
        model.train()
        
        # 创建examples格式的数据（不包含state）
        examples = [
            {
                "image": [Image.new('RGB', (224, 224), color='red')],
                "lang": "Pick up the red object.",
                "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32)
            }
        ]
        
        # 前向传播
        outputs = model(examples=examples)
        
        print("✓ 不使用状态的训练模式成功")
        if "action_loss" in outputs:
            print(f"  输出action_loss: {outputs['action_loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 不使用状态的测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_qwen_groot_repeated_diffusion_steps():
    """测试重复扩散步数"""
    print("\n" + "=" * 60)
    print("测试6: QwenGR00T重复扩散步数")
    print("=" * 60)
    
    try:
        vlm_config = {
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "cache_dir": "./cache/models"
        }
        
        action_head_config = {
            "hidden_dim": 768,
            "num_layers": 2,
            "num_heads": 8,
            "action_dim": 7,
            "action_horizon": 4,
            "num_inference_timesteps": 5
        }
        
        model = QwenGR00TVLAModel(
            vlm_config=vlm_config,
            action_head_config=action_head_config,
            use_state=True,
            state_dim=7,
            future_action_window_size=3
        )
        model.train()
        
        examples = [
            {
                "image": [Image.new('RGB', (224, 224), color='red')],
                "lang": "Pick up the red object.",
                "action": np.random.uniform(-1, 1, size=(16, 7)).astype(np.float32),
                "state": np.random.uniform(-1, 1, size=(1, 7)).astype(np.float32)
            }
        ]
        
        # 使用重复扩散步数
        repeated_steps = 2
        outputs = model(examples=examples, repeated_diffusion_steps=repeated_steps)
        
        print("✓ 重复扩散步数测试成功")
        print(f"  重复步数: {repeated_steps}")
        if "action_loss" in outputs:
            print(f"  输出action_loss: {outputs['action_loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 重复扩散步数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试QwenGR00TVLAModel模块")
    print("=" * 60)
    
    results = []
    
    # 测试1: 初始化
    model = test_qwen_groot_initialization()
    results.append(("初始化", model is not None))
    
    if model is not None:
        # 测试2: examples格式前向传播
        results.append(("前向传播（examples格式）", test_qwen_groot_forward_examples()))
        
        # 测试3: images/texts格式前向传播
        results.append(("前向传播（images/texts格式）", test_qwen_groot_forward_images_texts()))
        
        # 测试4: 推理模式
        results.append(("推理模式", test_qwen_groot_predict_action()))
        
        # 测试5: 不使用状态
        results.append(("不使用状态", test_qwen_groot_without_state()))
        
        # 测试6: 重复扩散步数
        results.append(("重复扩散步数", test_qwen_groot_repeated_diffusion_steps()))
    
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

