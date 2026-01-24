# MIT License
#
# Copyright (c) 2024 ScriptedVLA Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Benny Lu
"""
LIBERO仿真环境验证脚本
在真实的libero-object仿真环境中运行训练好的模型，验证模型效果
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import argparse
import time
from PIL import Image

from src.ScriptedVLA.model import QwenGR00TVLAModel
from src.ScriptedVLA.utils import (
    load_config,
    get_model_config,
    get_data_config,
    Normalizer
)
from inference import (
    find_latest_checkpoint,
    load_model_from_checkpoint,
    prepare_images_input
)

# 尝试导入LIBERO相关库
HAS_LIBERO = False
HAS_MUJOCO = False
LIBERO_ENV = None

try:
    import libero
    from libero.libero import benchmark
    HAS_LIBERO = True
except ImportError:
    pass

try:
    import mujoco
    HAS_MUJOCO = True
except ImportError:
    pass

# 尝试导入gym环境（LIBERO可能使用gym接口）
try:
    import gym
    import gymnasium
    HAS_GYM = True
except ImportError:
    HAS_GYM = False


def _run_inference(
    model: QwenGR00TVLAModel,
    images: Dict[str, torch.Tensor],
    instruction: str,
    states: Optional[torch.Tensor] = None,
    normalizer: Optional[Normalizer] = None
) -> np.ndarray:
    """
    运行推理（内部函数，避免循环导入）
    
    Args:
        model: 训练好的模型
        images: 图像字典 {camera_name: tensor}
        instruction: 文本指令
        states: 机器人状态
        normalizer: 动作归一化器
        
    Returns:
        预测的动作序列 [T, action_dim]
    """
    device = next(model.parameters()).device
    
    # 准备图像输入
    images_input = prepare_images_input(images, device)
    
    # 处理状态
    if states is not None:
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(np.array(states), dtype=torch.float32)
        states = states.to(device)
        if states.dim() == 1:
            states = states.unsqueeze(0)
    elif model.use_state:
        states = torch.zeros(1, model.state_dim, device=device)
    
    # 推理
    inputs = {
        "images": images_input,
        "instructions": [instruction],
        "states": states
    }
    
    with torch.no_grad():
        outputs = model.predict_action(inputs=inputs)
    
    # 提取动作
    if "normalized_actions" in outputs:
        actions = outputs["normalized_actions"]  # [B, T, action_dim]
    elif "actions" in outputs:
        actions = outputs["actions"]  # [B, T, action_dim]
    else:
        raise ValueError("Model output does not contain actions")
    
    # 转换为numpy
    if isinstance(actions, torch.Tensor):
        actions = actions.cpu().numpy()
    
    # 反归一化
    if normalizer is not None:
        actions = normalizer.denormalize_action(actions)
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
    
    # 返回action chunk [T, action_dim]
    if len(actions.shape) == 3:  # [B, T, action_dim]
        return actions[0]  # [T, action_dim]
    elif len(actions.shape) == 2:  # [B, action_dim] 或 [T, action_dim]
        if actions.shape[0] == 1:  # [1, action_dim]
            return actions[0:1]  # [1, action_dim]
        else:
            return actions  # [T, action_dim]
    else:
        return actions.reshape(1, -1)  # [1, action_dim]


def check_environment():
    """检查仿真环境是否可用"""
    issues = []
    
    if not HAS_LIBERO:
        issues.append("LIBERO库未安装。请运行: pip install libero")
    
    if not HAS_MUJOCO:
        issues.append("MuJoCo库未安装。请运行: pip install mujoco")
    
    if not HAS_GYM:
        issues.append("Gym库未安装。请运行: pip install gym gymnasium")
    
    return len(issues) == 0, issues


def create_libero_env(task_name: str = None, headless: bool = False):
    """
    创建LIBERO仿真环境
    
    Args:
        task_name: 任务名称，如果为None则使用libero_object的第一个任务
        headless: 是否无头模式（不显示GUI）
        
    Returns:
        环境对象和任务信息
    """
    if not HAS_LIBERO:
        raise ImportError("LIBERO库未安装。请运行: pip install libero")
    
    # 获取libero_object基准
    benchmark_dict = benchmark.get_benchmark_dict()
    
    if "libero_object" not in benchmark_dict:
        raise ValueError("无法找到libero_object基准。请检查LIBERO安装。")
    
    task_suite = benchmark_dict["libero_object"]()
    tasks = task_suite.get_tasks()
    
    if not tasks:
        raise ValueError("libero_object中没有找到任务。")
    
    # 选择任务
    if task_name is None:
        selected_task = tasks[0]
        print(f"未指定任务，使用第一个任务: {selected_task.name}")
    else:
        selected_task = None
        for task in tasks:
            if task.name == task_name:
                selected_task = task
                break
        if selected_task is None:
            print(f"警告: 未找到任务 {task_name}，使用第一个任务: {tasks[0].name}")
            selected_task = tasks[0]
    
    print(f"选择任务: {selected_task.name}")
    print(f"任务描述: {selected_task.language}")
    
    # 创建环境
    # LIBERO通常使用gym接口，但具体实现可能因版本而异
    # 这里提供一个通用的接口
    
    try:
        # 尝试使用gym接口创建环境
        # LIBERO环境ID格式通常是 "libero-{task_name}-v0" 或类似格式
        env_id = f"libero-{selected_task.name}-v0"
        
        try:
            env = gym.make(env_id)
        except:
            try:
                env = gymnasium.make(env_id)
            except:
                # 如果gym接口不可用，尝试直接使用LIBERO API
                # 注意：这需要根据LIBERO的实际API调整
                print("警告: 无法通过gym创建环境，尝试使用LIBERO直接API...")
                # 这里需要根据LIBERO的实际API实现
                # 由于LIBERO的API可能因版本而异，这里提供一个占位符
                raise NotImplementedError(
                    "LIBERO环境创建需要根据实际LIBERO版本调整API调用。\n"
                    "请参考LIBERO官方文档：https://github.com/Lifelong-Robot-Learning/LIBERO"
                )
        
        return env, selected_task
        
    except Exception as e:
        raise RuntimeError(
            f"无法创建LIBERO环境: {e}\n"
            "可能的原因：\n"
            "1. LIBERO环境未正确安装\n"
            "2. MuJoCo未正确配置\n"
            "3. Windows环境可能需要额外配置（建议使用WSL或Linux）\n"
            "请参考LIBERO安装文档：https://github.com/Lifelong-Robot-Learning/LIBERO"
        )


def run_episode_in_simulation(
    model: QwenGR00TVLAModel,
    env,
    task,
    normalizer: Optional[Normalizer] = None,
    max_steps: int = 200,
    render: bool = True,
    action_horizon: int = 1
) -> Dict:
    """
    在仿真环境中运行一个episode
    
    Args:
        model: 训练好的模型
        env: 仿真环境
        task: 任务对象
        normalizer: 动作归一化器
        max_steps: 最大步数
        render: 是否渲染（显示GUI）
        action_horizon: 动作序列长度（模型预测的chunk大小）
        
    Returns:
        包含episode信息的字典
    """
    device = next(model.parameters()).device
    
    # 重置环境
    if hasattr(env, 'reset'):
        obs = env.reset()
    else:
        obs, info = env.reset(return_info=True)
    
    episode_info = {
        "success": False,
        "steps": 0,
        "reward": 0.0,
        "actions": [],
        "observations": [],
        "task_name": task.name,
        "task_description": task.language
    }
    
    # 获取任务指令
    instruction = task.language
    
    print(f"\n开始执行任务: {task.name}")
    print(f"任务描述: {instruction}")
    print(f"最大步数: {max_steps}")
    print("-" * 60)
    
    # 动作缓冲区（用于存储预测的action chunk）
    action_buffer = []
    action_buffer_idx = 0
    
    for step in range(max_steps):
        # 获取当前观察
        if isinstance(obs, dict):
            # 多相机情况
            images_dict = {}
            for key, value in obs.items():
                if "image" in key.lower() or "rgb" in key.lower():
                    # 提取相机名称
                    if "observation" in key and "images" in key:
                        # 格式: observation.images.{camera_name}
                        parts = key.split(".")
                        if len(parts) >= 3:
                            camera_name = parts[-1]
                        else:
                            camera_name = key
                    else:
                        camera_name = key
                    
                    # 转换图像格式
                    if isinstance(value, np.ndarray):
                        # 确保图像格式正确 [H, W, C] -> [C, H, W]
                        if len(value.shape) == 3:
                            if value.shape[2] == 3:  # RGB
                                img_tensor = torch.from_numpy(value).permute(2, 0, 1).float() / 255.0
                            else:  # 可能是 [H, W, C] 但C不是3
                                img_tensor = torch.from_numpy(value).float()
                                if img_tensor.max() > 1.0:
                                    img_tensor = img_tensor / 255.0
                            images_dict[camera_name] = img_tensor
                        else:
                            print(f"警告: 图像维度不正确: {value.shape}")
                    elif isinstance(value, Image.Image):
                        img_array = np.array(value)
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                        images_dict[camera_name] = img_tensor
            
            # 获取状态（如果存在）
            state = None
            if "state" in obs:
                state = obs["state"]
            elif "observation.state" in obs:
                state = obs["observation.state"]
        else:
            # 单相机情况（假设obs是图像）
            if isinstance(obs, np.ndarray):
                if len(obs.shape) == 3:  # [H, W, C]
                    img_tensor = torch.from_numpy(obs).permute(2, 0, 1).float() / 255.0
                    images_dict = {"image": img_tensor}
                else:
                    raise ValueError(f"不支持的观察格式: {obs.shape}")
            else:
                raise ValueError(f"不支持的观察类型: {type(obs)}")
        
        # 如果动作缓冲区为空，进行推理
        if len(action_buffer) == 0:
            # 运行模型推理
            try:
                predicted_actions = _run_inference(
                    model=model,
                    images=images_dict,
                    instruction=instruction,
                    states=state,
                    normalizer=normalizer
                )
                
                # predicted_actions应该是 [T, action_dim] 格式
                if len(predicted_actions.shape) == 1:
                    # [action_dim] -> [1, action_dim]
                    predicted_actions = predicted_actions.reshape(1, -1)
                
                # 填充动作缓冲区
                action_buffer = predicted_actions.tolist()
                action_buffer_idx = 0
                
            except Exception as e:
                print(f"推理失败: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # 从缓冲区获取动作
        if len(action_buffer) > 0:
            action = np.array(action_buffer[action_buffer_idx])
            action_buffer_idx += 1
            
            # 如果缓冲区用完了，清空以便下次推理
            if action_buffer_idx >= len(action_buffer):
                action_buffer = []
                action_buffer_idx = 0
        else:
            # 如果没有动作，使用零动作
            action = np.zeros(7)  # 默认7维动作
        
        # 执行动作
        if hasattr(env, 'step'):
            result = env.step(action)
            if len(result) == 4:
                next_obs, reward, done, info = result
            else:
                next_obs, reward, terminated, truncated, info = result
                done = terminated or truncated
        else:
            raise ValueError("环境没有step方法")
        
        # 记录信息
        episode_info["steps"] += 1
        episode_info["reward"] += reward
        episode_info["actions"].append(action.copy())
        episode_info["observations"].append(obs.copy() if isinstance(obs, np.ndarray) else obs)
        
        # 检查是否成功
        if info.get("success", False) or info.get("is_success", False):
            episode_info["success"] = True
            print(f"\n✓ 任务成功完成！步数: {step + 1}")
            break
        
        # 检查是否结束
        if done:
            print(f"\nEpisode结束。步数: {step + 1}")
            break
        
        # 更新观察
        obs = next_obs
        
        # 渲染（如果启用）
        if render and hasattr(env, 'render'):
            try:
                env.render()
            except:
                pass
        
        # 打印进度
        if (step + 1) % 10 == 0:
            print(f"步数: {step + 1}/{max_steps}, 累计奖励: {episode_info['reward']:.2f}")
    
    return episode_info


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="在LIBERO仿真环境中验证训练好的模型"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./checkpoints",
        help="检查点目录"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default=None,
        help="要执行的任务名称（如果为None，使用第一个任务）"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=200,
        help="每个episode的最大步数"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1,
        help="要运行的episode数量"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="无头模式（不显示GUI）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（'cuda'或'cpu'），如果为None则自动选择"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("LIBERO仿真环境验证")
    print("=" * 60)
    
    # 检查环境
    print("\n[Step 0] 检查仿真环境...")
    env_ok, issues = check_environment()
    if not env_ok:
        print("  ✗ 环境检查失败:")
        for issue in issues:
            print(f"    - {issue}")
        print("\n请按照上述提示安装缺失的依赖。")
        print("\n注意：Windows环境下可能需要：")
        print("  1. 使用WSL (Windows Subsystem for Linux)")
        print("  2. 或使用Docker容器")
        print("  3. 或参考LIBERO官方文档的Windows安装指南")
        return
    
    print("  ✓ 环境检查通过")
    
    # 1. 查找最新checkpoint
    print("\n[Step 1] 查找最新checkpoint...")
    checkpoint_dir = Path(args.checkpoint_dir)
    latest_checkpoint = find_latest_checkpoint(checkpoint_dir)
    
    if latest_checkpoint is None:
        print(f"  ✗ 未找到checkpoint文件在: {checkpoint_dir}")
        return
    
    print(f"  ✓ 找到最新checkpoint: {latest_checkpoint}")
    
    # 2. 加载模型
    print("\n[Step 2] 加载模型...")
    try:
        model, normalizer = load_model_from_checkpoint(
            str(latest_checkpoint),
            args.config,
            args.device
        )
        device = next(model.parameters()).device
        print(f"  ✓ 模型加载成功，设备: {device}")
    except Exception as e:
        print(f"  ✗ 模型加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 创建仿真环境
    print("\n[Step 3] 创建仿真环境...")
    try:
        env, task = create_libero_env(task_name=args.task_name, headless=args.headless)
        print(f"  ✓ 环境创建成功")
    except Exception as e:
        print(f"  ✗ 环境创建失败: {e}")
        print("\n提示：")
        print("  1. 确保已安装LIBERO: pip install libero")
        print("  2. 确保已安装MuJoCo: pip install mujoco")
        print("  3. Windows用户可能需要使用WSL或Docker")
        print("  4. 参考LIBERO官方文档：https://github.com/Lifelong-Robot-Learning/LIBERO")
        return
    
    # 4. 运行episodes
    print("\n[Step 4] 运行仿真episodes...")
    results = []
    
    for episode_idx in range(args.num_episodes):
        print(f"\n{'=' * 60}")
        print(f"Episode {episode_idx + 1}/{args.num_episodes}")
        print(f"{'=' * 60}")
        
        try:
            episode_info = run_episode_in_simulation(
                model=model,
                env=env,
                task=task,
                normalizer=normalizer,
                max_steps=args.max_steps,
                render=not args.headless
            )
            results.append(episode_info)
            
            print(f"\nEpisode {episode_idx + 1} 结果:")
            print(f"  成功: {episode_info['success']}")
            print(f"  步数: {episode_info['steps']}")
            print(f"  累计奖励: {episode_info['reward']:.2f}")
            
        except Exception as e:
            print(f"  ✗ Episode {episode_idx + 1} 失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 5. 统计结果
    if results:
        print("\n" + "=" * 60)
        print("总体统计")
        print("=" * 60)
        
        success_count = sum(1 for r in results if r["success"])
        avg_steps = np.mean([r["steps"] for r in results])
        avg_reward = np.mean([r["reward"] for r in results])
        
        print(f"总episodes: {len(results)}")
        print(f"成功次数: {success_count}")
        print(f"成功率: {success_count / len(results) * 100:.1f}%")
        print(f"平均步数: {avg_steps:.1f}")
        print(f"平均奖励: {avg_reward:.2f}")
        print("=" * 60)
    
    # 关闭环境
    if hasattr(env, 'close'):
        env.close()


if __name__ == "__main__":
    main()
