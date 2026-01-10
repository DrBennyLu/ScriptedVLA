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
            hidden_dim=1536,
            num_layers=6,  # 使用较少的层数以加快测试
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
        
        # 计算可训练参数总量
        total_params = sum(p.numel() for p in action_head.parameters() if p.requires_grad)
        trainable_params = sum(p.numel() for p in action_head.parameters() if p.requires_grad)
        total_all_params = sum(p.numel() for p in action_head.parameters())
        
        print(f"  可训练参数总量: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
        print(f"  总参数数量: {total_all_params:,} ({total_all_params / 1e6:.2f}M)")
        
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


def test_states_dimension_normalization():
    """测试states维度规范化"""
    print("\n" + "=" * 60)
    print("测试7: States维度规范化")
    print("=" * 60)
    
    try:
        action_head = FlowMatchingActionHead(
            hidden_dim=768,
            num_layers=2,
            num_heads=8,
            action_dim=7,
            action_horizon=4,
            state_dim=7,
            num_inference_timesteps=5
        )
        action_head.train()
        
        batch_size = 2
        seq_len = 10
        vlm_features = torch.randn(batch_size, seq_len, 768)
        actions = torch.randn(batch_size, 4, 7)
        
        # 测试1: [B, state_dim] 格式（标准格式）
        print("\n  测试1: states为[B, state_dim]格式")
        states_2d = torch.randn(batch_size, 7)
        print(f"    输入states shape: {states_2d.shape}")
        loss_2d = action_head(vlm_features, actions=actions, states=states_2d)
        print(f"    ✓ [B, state_dim]格式测试成功，loss: {loss_2d.item():.4f}")
        
        # 测试2: [B, 1, state_dim] 格式
        print("\n  测试2: states为[B, 1, state_dim]格式")
        states_3d = torch.randn(batch_size, 1, 7)
        print(f"    输入states shape: {states_3d.shape}")
        loss_3d = action_head(vlm_features, actions=actions, states=states_3d)
        print(f"    ✓ [B, 1, state_dim]格式测试成功，loss: {loss_3d.item():.4f}")
        
        # 测试3: [B, T, state_dim] 格式（多时间步，应该取第一个）
        print("\n  测试3: states为[B, T, state_dim]格式（多时间步）")
        states_multi = torch.randn(batch_size, 5, 7)
        print(f"    输入states shape: {states_multi.shape}")
        loss_multi = action_head(vlm_features, actions=actions, states=states_multi)
        print(f"    ✓ [B, T, state_dim]格式测试成功，loss: {loss_multi.item():.4f}")
        
        # 测试4: 推理模式，测试不同维度
        print("\n  测试4: 推理模式，测试不同states维度")
        action_head.eval()
        with torch.no_grad():
            # [B, state_dim]
            pred_2d = action_head.predict_action(vlm_features, states=states_2d)
            print(f"    ✓ [B, state_dim]推理成功，输出shape: {pred_2d.shape}")
            
            # [B, 1, state_dim]
            pred_3d = action_head.predict_action(vlm_features, states=states_3d)
            print(f"    ✓ [B, 1, state_dim]推理成功，输出shape: {pred_3d.shape}")
            
            # 验证输出形状一致
            assert pred_2d.shape == pred_3d.shape, \
                f"不同states维度格式的输出形状应该一致: {pred_2d.shape} != {pred_3d.shape}"
        
        print("\n✓ States维度规范化测试成功")
        return True
        
    except Exception as e:
        print(f"✗ States维度规范化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vlm_features_dimension_alignment():
    """测试VLM features和DiT query维度对齐"""
    print("\n" + "=" * 60)
    print("测试8: VLM features与DiT query维度对齐")
    print("=" * 60)
    
    try:
        hidden_dim = 768
        action_head = FlowMatchingActionHead(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=8,
            action_dim=7,
            action_horizon=4,
            use_cross_attention=True,
            state_dim=7,
            num_inference_timesteps=5
        )
        action_head.train()
        
        batch_size = 2
        
        # 测试1: VLM features是3D张量 [B, seq_len, hidden_dim]
        print("\n  测试1: VLM features为3D张量 [B, seq_len, hidden_dim]")
        seq_len_3d = 10
        vlm_features_3d = torch.randn(batch_size, seq_len_3d, hidden_dim)
        actions = torch.randn(batch_size, 4, 7)
        states = torch.randn(batch_size, 7)
        
        # 检查维度
        print(f"    VLM features shape: {vlm_features_3d.shape}")
        print(f"    DiT hidden_dim: {hidden_dim}")
        print(f"    VLM features最后一维: {vlm_features_3d.shape[-1]}")
        assert vlm_features_3d.shape[-1] == hidden_dim, \
            f"VLM features维度不匹配: {vlm_features_3d.shape[-1]} != {hidden_dim}"
        
        # 前向传播测试
        loss_3d = action_head(
            vlm_features_3d,
            actions=actions,
            states=states
        )
        print(f"    ✓ 3D features前向传播成功，loss: {loss_3d.item():.4f}")
        
        # 测试2: VLM features是2D张量 [B, hidden_dim]（会被unsqueeze为[B, 1, hidden_dim]）
        print("\n  测试2: VLM features为2D张量 [B, hidden_dim]")
        vlm_features_2d = torch.randn(batch_size, hidden_dim)
        
        print(f"    VLM features shape: {vlm_features_2d.shape}")
        print(f"    DiT hidden_dim: {hidden_dim}")
        print(f"    VLM features最后一维: {vlm_features_2d.shape[-1]}")
        assert vlm_features_2d.shape[-1] == hidden_dim, \
            f"VLM features维度不匹配: {vlm_features_2d.shape[-1]} != {hidden_dim}"
        
        # 前向传播测试
        loss_2d = action_head(
            vlm_features_2d,
            actions=actions,
            states=states
        )
        print(f"    ✓ 2D features前向传播成功，loss: {loss_2d.item():.4f}")
        
        # 测试3: 测试维度不匹配的情况（应该被vlm_projection处理）
        print("\n  测试3: VLM features维度与hidden_dim不同（需要投影）")
        different_dim = 512
        action_head_proj = FlowMatchingActionHead(
            hidden_dim=hidden_dim,
            num_layers=2,
            num_heads=8,
            action_dim=7,
            action_horizon=4,
            use_cross_attention=True,
            state_dim=7,
            num_inference_timesteps=5,
            cross_attention_dim=different_dim  # VLM维度与hidden_dim不同
        )
        action_head_proj.train()
        
        vlm_features_diff = torch.randn(batch_size, 10, different_dim)
        print(f"    VLM features shape: {vlm_features_diff.shape}")
        print(f"    VLM features维度: {different_dim}")
        print(f"    DiT hidden_dim: {hidden_dim}")
        print(f"    动作头cross_attention_dim: {action_head_proj.cross_attention_dim}")
        
        # 检查是否有投影层
        if action_head_proj.vlm_projection is not None:
            print(f"    ✓ 检测到vlm_projection层: {action_head_proj.vlm_projection}")
            # 测试投影
            projected = action_head_proj.vlm_projection(vlm_features_diff)
            print(f"    投影后shape: {projected.shape}")
            assert projected.shape[-1] == hidden_dim, \
                f"投影后维度不匹配: {projected.shape[-1]} != {hidden_dim}"
            print(f"    ✓ 投影层维度对齐成功")
        else:
            print(f"    ⚠ 未检测到vlm_projection层（维度相同，不需要投影）")
        
        # 前向传播测试
        loss_proj = action_head_proj(
            vlm_features_diff,
            actions=actions,
            states=states
        )
        print(f"    ✓ 不同维度features前向传播成功，loss: {loss_proj.item():.4f}")
        
        # 测试4: 验证DiT块中的交叉注意力维度
        print("\n  测试4: 验证DiT块交叉注意力维度")
        # 获取第一个DiT块
        dit_block = action_head.blocks[0]
        if dit_block.use_cross_attention:
            # 创建query和encoder_hidden_states
            query_seq_len = 5  # action序列长度
            query = torch.randn(batch_size, query_seq_len, hidden_dim)
            encoder_hidden_states = torch.randn(batch_size, seq_len_3d, hidden_dim)
            
            print(f"    Query shape: {query.shape}")
            print(f"    Encoder hidden states shape: {encoder_hidden_states.shape}")
            print(f"    Query最后一维: {query.shape[-1]}")
            print(f"    Encoder最后一维: {encoder_hidden_states.shape[-1]}")
            
            assert query.shape[-1] == encoder_hidden_states.shape[-1], \
                f"Query和Encoder维度不匹配: {query.shape[-1]} != {encoder_hidden_states.shape[-1]}"
            
            # 测试交叉注意力
            with torch.no_grad():
                output = dit_block(
                    query,
                    encoder_hidden_states=encoder_hidden_states
                )
            print(f"    输出shape: {output.shape}")
            assert output.shape == query.shape, \
                f"输出shape不匹配: {output.shape} != {query.shape}"
            print(f"    ✓ DiT块交叉注意力维度对齐成功")
        else:
            print(f"    ⚠ DiT块未启用交叉注意力")
        
        print("\n✓ VLM features与DiT query维度对齐测试成功")
        return True
        
    except Exception as e:
        print(f"✗ VLM features维度对齐测试失败: {e}")
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
        
        # 测试7: States维度规范化
        results.append(("States维度规范化", test_states_dimension_normalization()))
        
        # 测试8: VLM features维度对齐
        results.append(("VLM features维度对齐", test_vlm_features_dimension_alignment()))
    
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
