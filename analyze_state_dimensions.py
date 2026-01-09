"""
分析state维度变化的可能原因
"""
import numpy as np
import torch

# 模拟测试代码中的情况
print("=" * 60)
print("情况1: 正常的states处理流程")
print("=" * 60)

# 测试代码中：state = np.random.uniform(-1, 1, size=(1, 7))
state_1 = np.random.uniform(-1, 1, size=(1, 7))
state_2 = np.random.uniform(-1, 1, size=(1, 7))
print(f"单个state形状: {state_1.shape}")

# vla_qwen_groot.py中的处理
states_list = [state_1, state_2]  # [B, 1, 7]
print(f"states_list长度: {len(states_list)}")
print(f"states_list[0]形状: {states_list[0].shape}")

# np.array(states_list)的处理
states_array = np.array(states_list)
print(f"np.array(states_list)形状: {states_array.shape}")  # 应该是 [2, 1, 7]

# 转换为tensor
states_tensor = torch.tensor(states_array)
print(f"torch.tensor后形状: {states_tensor.shape}")  # 应该是 [2, 1, 7]

# 在action_head.py中的处理
if states_tensor.dim() == 3 and states_tensor.shape[1] == 1:
    states_2d = states_tensor.squeeze(1)
    print(f"squeeze(1)后形状: {states_2d.shape}")  # 应该是 [2, 7]

# 模拟state_encoder (MLP)
hidden_dim = 768
state_features = torch.randn(states_2d.shape[0], hidden_dim)  # [2, 768]
print(f"state_encoder输出形状: {state_features.shape}")

# unsqueeze(1)
state_features_3d = state_features.unsqueeze(1)
print(f"unsqueeze(1)后形状: {state_features_3d.shape}")  # 应该是 [2, 1, 768]

print("\n" + "=" * 60)
print("情况2: 可能导致4维的情况")
print("=" * 60)

# 情况2.1: 如果states在某个地方被错误地unsqueeze了
states_4d_wrong = states_tensor.unsqueeze(2)  # [2, 1, 1, 7]
print(f"错误unsqueeze后: {states_4d_wrong.shape}")

# 如果这个4维的states被传入state_encoder会怎样？
# MLP期望输入是[B, input_dim]，但收到[B, 1, 1, 7]会报错
# 但如果先squeeze(1)，会变成[B, 1, 7]，然后squeeze(1)会变成[B, 7]
# 但如果squeeze(2)，会变成[B, 1, 7]，然后squeeze(1)会变成[B, 7]

# 情况2.2: 如果state_encoder的输出被错误地unsqueeze了
state_features_4d = state_features.unsqueeze(1).unsqueeze(2)  # [2, 1, 1, 768]
print(f"错误unsqueeze两次后: {state_features_4d.shape}")

print("\n" + "=" * 60)
print("情况3: repeated_diffusion_steps的影响")
print("=" * 60)

# 如果repeated_diffusion_steps > 1
repeated_diffusion_steps = 4
states_repeated = states_tensor.repeat(repeated_diffusion_steps, 1, 1)
print(f"repeat({repeated_diffusion_steps}, 1, 1)后形状: {states_repeated.shape}")  # [8, 1, 7]

# 如果这个被传入action_head
if states_repeated.dim() == 3 and states_repeated.shape[1] == 1:
    states_repeated_2d = states_repeated.squeeze(1)
    print(f"squeeze(1)后形状: {states_repeated_2d.shape}")  # [8, 7]

# state_encoder输出
state_features_repeated = torch.randn(states_repeated_2d.shape[0], hidden_dim)  # [8, 768]
print(f"state_encoder输出形状: {state_features_repeated.shape}")

# unsqueeze(1)
state_features_repeated_3d = state_features_repeated.unsqueeze(1)
print(f"unsqueeze(1)后形状: {state_features_repeated_3d.shape}")  # [8, 1, 768]

print("\n" + "=" * 60)
print("结论: 4维最可能的原因")
print("=" * 60)
print("1. states在某个地方被错误地unsqueeze，变成了[B, 1, 1, state_dim]")
print("2. state_features在unsqueeze(1)之后又被错误地unsqueeze了一次")
print("3. 在某个地方，states被错误地reshape成了4维")
print("4. 如果state_encoder的MLP有问题，可能会输出错误的维度")

