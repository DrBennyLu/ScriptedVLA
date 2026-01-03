"""
测试FlowMatchingActionHead模块
测试动作头的基本功能、训练模式、推理模式等
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from ScriptedVLA.model.action_head import FlowMatchingActionHead, ActionEncoder, MLP


def test_action_head_initialization():
    """测试动作头初始化"""
    print("=" * 60)
    print("测试1: FlowMatchingActionHead初始化")
    print("=" * 60)
    
    try:
        action_head = FlowMatchingActionHead(
            input_dim=768,
            hidden_dim=768,
            num_layers=2,  # 使用较少的层数以加快测试
            num_heads=8,
            action_dim=7,
            action_horizon=4,
            state_dim=7,
            num_inference_timesteps=10  # 使用较少的步数以加快测试
        )
        print("✓ 动作头初始化成功")
        print(f"  Hidden dim: {action_head.hidden_dim}")
        print(f"  Action dim: {action_head.action_dim}")
        print(f"  Action horizon: {action_head.action_horizon}")
        return action_head
    except Exception as e:
        print(f"✗ 动作头初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_action_encoder():
    """测试ActionEncoder"""
    print("\n" + "=" * 60)
    print("测试2: ActionEncoder")
    print("=" * 60)
    
    try:
        action_encoder = ActionEncoder(action_dim=7, hidden_size=768)
        
        batch_size = 2
        action_horizon = 4
        actions = torch.randn(batch_size, action_horizon, 7)  # [B, T, action_dim]
        timesteps = torch.randint(0, 1000, (batch_size,))  # [B]
        
        # 前向传播
        encoded = action_encoder(actions, timesteps)
        
        print("✓ ActionEncoder前向传播成功")
        print(f"  输入actions shape: {actions.shape}")
        print(f"  输入timesteps shape: {timesteps.shape}")
        print(f"  输出encoded shape: {encoded.shape}")
        
        # 验证形状
        assert encoded.shape == (batch_size, action_horizon, 768), \
            f"输出形状不正确: {encoded.shape} != ({batch_size}, {action_horizon}, 768)"
        
        return True
    except Exception as e:
        print(f"✗ ActionEncoder测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_head_forward_training():
    """测试动作头训练模式"""
    print("\n" + "=" * 60)
    print("测试3: FlowMatchingActionHead训练模式")
    print("=" * 60)
    
    try:
        action_head = FlowMatchingActionHead(
            input_dim=768,
            hidden_dim=768,
            num_layers=2,
            num_heads=8,
            action_dim=7,
            action_horizon=4,
            state_dim=7,
            num_inference_timesteps=10
        )
        action_head.train()
        
        batch_size = 2
        seq_len = 10
        vlm_features = torch.randn(batch_size, seq_len, 768)  # [B, seq_len, hidden_dim]
        actions = torch.randn(batch_size, 4, 7)  # [B, action_horizon, action_dim]
        states = torch.randn(batch_size, 7)  # [B, state_dim]
        
        # 前向传播（训练模式）
        loss = action_head(
            vlm_features,
            actions=actions,
            states=states
        )
        
        print("✓ 训练模式前向传播成功")
        print(f"  输入vlm_features shape: {vlm_features.shape}")
        print(f"  输入actions shape: {actions.shape}")
        print(f"  输入states shape: {states.shape}")
        print(f"  输出loss: {loss.item():.4f}")
        
        # 验证损失
        assert isinstance(loss, torch.Tensor), "损失应该是tensor"
        assert loss.dim() == 0, "损失应该是标量"
        assert loss.item() >= 0, "损失应该非负"
        
        return True
    except Exception as e:
        print(f"✗ 训练模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_head_forward_inference():
    """测试动作头推理模式"""
    print("\n" + "=" * 60)
    print("测试4: FlowMatchingActionHead推理模式")
    print("=" * 60)
    
    try:
        action_head = FlowMatchingActionHead(
            input_dim=768,
            hidden_dim=768,
            num_layers=2,
            num_heads=8,
            action_dim=7,
            action_horizon=4,
            state_dim=7,
            num_inference_timesteps=5  # 使用较少的步数以加快测试
        )
        action_head.eval()
        
        batch_size = 2
        seq_len = 10
        vlm_features = torch.randn(batch_size, seq_len, 768)  # [B, seq_len, hidden_dim]
        states = torch.randn(batch_size, 7)  # [B, state_dim]
        
        # 推理模式
        with torch.no_grad():
            pred_actions = action_head.predict_action(
                vlm_features,
                states=states
            )
        
        print("✓ 推理模式前向传播成功")
        print(f"  输入vlm_features shape: {vlm_features.shape}")
        print(f"  输入states shape: {states.shape}")
        print(f"  输出pred_actions shape: {pred_actions.shape}")
        
        # 验证形状
        assert pred_actions.shape == (batch_size, 4, 7), \
            f"输出形状不正确: {pred_actions.shape} != ({batch_size}, 4, 7)"
        
        return True
    except Exception as e:
        print(f"✗ 推理模式测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_head_without_state():
    """测试不使用状态的动作头"""
    print("\n" + "=" * 60)
    print("测试5: FlowMatchingActionHead（不使用状态）")
    print("=" * 60)
    
    try:
        action_head = FlowMatchingActionHead(
            input_dim=768,
            hidden_dim=768,
            num_layers=2,
            num_heads=8,
            action_dim=7,
            action_horizon=4,
            state_dim=None,  # 不使用状态
            num_inference_timesteps=5
        )
        action_head.train()
        
        batch_size = 2
        seq_len = 10
        vlm_features = torch.randn(batch_size, seq_len, 768)
        actions = torch.randn(batch_size, 4, 7)
        
        # 训练模式（不使用状态）
        loss = action_head(
            vlm_features,
            actions=actions,
            states=None
        )
        
        print("✓ 不使用状态的训练模式成功")
        print(f"  输出loss: {loss.item():.4f}")
        
        # 推理模式（不使用状态）
        action_head.eval()
        with torch.no_grad():
            pred_actions = action_head.predict_action(
                vlm_features,
                states=None
            )
        
        print(f"  推理输出pred_actions shape: {pred_actions.shape}")
        
        return True
    except Exception as e:
        print(f"✗ 不使用状态的测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_head_cross_attention():
    """测试交叉注意力功能"""
    print("\n" + "=" * 60)
    print("测试6: FlowMatchingActionHead交叉注意力")
    print("=" * 60)
    
    try:
        action_head = FlowMatchingActionHead(
            input_dim=768,
            hidden_dim=768,
            num_layers=2,
            num_heads=8,
            action_dim=7,
            action_horizon=4,
            use_cross_attention=True,  # 使用交叉注意力
            state_dim=7,
            num_inference_timesteps=5
        )
        action_head.train()
        
        batch_size = 2
        seq_len = 10
        vlm_features = torch.randn(batch_size, seq_len, 768)  # [B, seq_len, hidden_dim]
        actions = torch.randn(batch_size, 4, 7)
        states = torch.randn(batch_size, 7)
        
        # 训练模式（使用交叉注意力）
        loss = action_head(
            vlm_features,
            actions=actions,
            states=states
        )
        
        print("✓ 交叉注意力测试成功")
        print(f"  输入vlm_features shape: {vlm_features.shape}")
        print(f"  输出loss: {loss.item():.4f}")
        
        return True
    except Exception as e:
        print(f"✗ 交叉注意力测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("开始测试FlowMatchingActionHead模块")
    print("=" * 60)
    
    results = []
    
    # 测试1: 初始化
    action_head = test_action_head_initialization()
    results.append(("初始化", action_head is not None))
    
    if action_head is not None:
        # 测试2: ActionEncoder
        results.append(("ActionEncoder", test_action_encoder()))
        
        # 测试3: 训练模式
        results.append(("训练模式", test_action_head_forward_training()))
        
        # 测试4: 推理模式
        results.append(("推理模式", test_action_head_forward_inference()))
        
        # 测试5: 不使用状态
        results.append(("不使用状态", test_action_head_without_state()))
        
        # 测试6: 交叉注意力
        results.append(("交叉注意力", test_action_head_cross_attention()))
    
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
