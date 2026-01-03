"""
测试VLAModel模块
测试VLA模型的基本功能、前向传播、推理等
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from PIL import Image
from ScriptedVLA.model.vla import VLAModel


def test_vla_initialization():
    """测试VLA模型初始化"""
    print("=" * 60)
    print("测试1: VLAModel初始化")
    print("=" * 60)
    
    try:
        vlm_config = {
            "model_name": "Qwen/Qwen2-VL-2B-Instruct",
            "image_size": 224,
            "max_seq_length": 512,
            "freeze_vlm": False,
            "cache_dir": "./cache/models"
        }
        
        action_head_config = {
            "hidden_dim": 768,
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
        
        model = VLAModel(
            vlm_config=vlm_config,
            action_head_config=action_head_config,
            use_cross_attention=True,
            cross_attention_layers=2,
            camera_names=["global_img"],
            use_state=True,
            state_dim=7
        )
        
        print("✓ VLAModel初始化成功")
        return model
    except Exception as e:
        print(f"✗ VLAModel初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_vla_forward_single_camera():
    """测试VLA前向传播（单相机）"""
    print("\n" + "=" * 60)
    print("测试2: VLA前向传播（单相机，训练模式）")
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
        
        model = VLAModel(
            vlm_config=vlm_config,
            action_head_config=action_head_config,
            camera_names=["global_img"],
            use_state=True,
            state_dim=7
        )
        model.train()
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)  # [B, C, H, W]
        texts = ["Pick up the red object.", "Place the blue object on the table."]
        states = torch.randn(batch_size, 7)  # [B, state_dim]
        actions = torch.randn(batch_size, 4, 7)  # [B, action_horizon, action_dim]
        
        # 前向传播（训练模式）
        outputs = model(
            images=images,
            texts=texts,
            states=states,
            actions=actions
        )
        
        print("✓ 前向传播成功（单相机，训练模式）")
        print(f"  输入images shape: {images.shape}")
        print(f"  输入states shape: {states.shape}")
        print(f"  输入actions shape: {actions.shape}")
        
        # 检查输出
        if "loss" in outputs:
            print(f"  输出loss: {outputs['loss'].item():.4f}")
            assert isinstance(outputs["loss"], torch.Tensor), "损失应该是tensor"
        elif "action_loss" in outputs:
            print(f"  输出action_loss: {outputs['action_loss'].item():.4f}")
            assert isinstance(outputs["action_loss"], torch.Tensor), "损失应该是tensor"
        
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vla_forward_multi_camera():
    """测试VLA前向传播（多相机）"""
    print("\n" + "=" * 60)
    print("测试3: VLA前向传播（多相机）")
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
        
        model = VLAModel(
            vlm_config=vlm_config,
            action_head_config=action_head_config,
            camera_names=["global_img", "left_wrist_img"],
            use_state=True,
            state_dim=7
        )
        model.train()
        
        batch_size = 2
        images = {
            "global_img": torch.randn(batch_size, 3, 224, 224),
            "left_wrist_img": torch.randn(batch_size, 3, 224, 224)
        }
        texts = ["Pick up the red object.", "Place the blue object."]
        states = torch.randn(batch_size, 7)
        actions = torch.randn(batch_size, 4, 7)
        
        # 前向传播
        outputs = model(
            images=images,
            texts=texts,
            states=states,
            actions=actions
        )
        
        print("✓ 前向传播成功（多相机）")
        print(f"  输入images keys: {list(images.keys())}")
        for key, value in images.items():
            print(f"    {key} shape: {value.shape}")
        
        if "loss" in outputs:
            print(f"  输出loss: {outputs['loss'].item():.4f}")
        elif "action_loss" in outputs:
            print(f"  输出action_loss: {outputs['action_loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vla_predict_action():
    """测试VLA推理模式"""
    print("\n" + "=" * 60)
    print("测试4: VLA推理模式（predict_action）")
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
        
        model = VLAModel(
            vlm_config=vlm_config,
            action_head_config=action_head_config,
            camera_names=["global_img"],
            use_state=True,
            state_dim=7
        )
        model.eval()
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        texts = ["Pick up the red object.", "Place the blue object."]
        states = torch.randn(batch_size, 7)
        
        # 推理模式
        with torch.no_grad():
            outputs = model.predict_action(
                images=images,
                texts=texts,
                states=states
            )
        
        print("✓ 推理模式成功")
        print(f"  输入images shape: {images.shape}")
        print(f"  输入states shape: {states.shape}")
        
        # 检查输出
        if "actions" in outputs:
            print(f"  输出actions shape: {outputs['actions'].shape}")
            assert outputs["actions"].shape[0] == batch_size, "Batch size不匹配"
        elif "normalized_actions" in outputs:
            print(f"  输出normalized_actions shape: {outputs['normalized_actions'].shape}")
            assert outputs["normalized_actions"].shape[0] == batch_size, "Batch size不匹配"
        
        return True
    except Exception as e:
        print(f"✗ 推理模式失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vla_without_state():
    """测试不使用状态的VLA模型"""
    print("\n" + "=" * 60)
    print("测试5: VLA模型（不使用状态）")
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
        
        model = VLAModel(
            vlm_config=vlm_config,
            action_head_config=action_head_config,
            camera_names=["global_img"],
            use_state=False,  # 不使用状态
            state_dim=7
        )
        model.train()
        
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)
        texts = ["Pick up the red object.", "Place the blue object."]
        actions = torch.randn(batch_size, 4, 7)
        
        # 前向传播（不使用状态）
        outputs = model(
            images=images,
            texts=texts,
            states=None,
            actions=actions
        )
        
        print("✓ 不使用状态的训练模式成功")
        if "loss" in outputs:
            print(f"  输出loss: {outputs['loss'].item():.4f}")
        elif "action_loss" in outputs:
            print(f"  输出action_loss: {outputs['action_loss'].item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 不使用状态的测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试VLAModel模块")
    print("=" * 60)
    
    results = []
    
    # 测试1: 初始化
    model = test_vla_initialization()
    results.append(("初始化", model is not None))
    
    if model is not None:
        # 测试2: 单相机前向传播
        results.append(("前向传播（单相机）", test_vla_forward_single_camera()))
        
        # 测试3: 多相机前向传播
        results.append(("前向传播（多相机）", test_vla_forward_multi_camera()))
        
        # 测试4: 推理模式
        results.append(("推理模式", test_vla_predict_action()))
        
        # 测试5: 不使用状态
        results.append(("不使用状态", test_vla_without_state()))
    
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

