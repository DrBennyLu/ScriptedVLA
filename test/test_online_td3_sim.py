"""
本地 LeRobot 数据上的 TD3 模拟在线训练脚本。

分步调试（任选其一）::

    python -m test.test_online_td3_sim --step dataset --dataset ./dataset/libero_object
    python -m test.test_online_td3_sim --step models
    python -m test.test_online_td3_sim --step replay
    python -m test.test_online_td3_sim --step warmup --save-agent ./out/agent.pt
    python -m test.test_online_td3_sim --step online --load-agent ./out/agent.pt
    python -m test.test_online_td3_sim --step eval --load-agent ./out/agent.pt
    python -m test.test_online_td3_sim --step full

默认 --step full 为完整流程。

author: Benny Lu
license: MIT
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ScriptedVLA.model import RLTokenBottleneck, TD3ChunkAgent, TD3ChunkConfig
from src.ScriptedVLA.utils import (
    create_normalizer_from_dataset,
    create_normalizer_from_lerobot_meta,
    load_script_config,
)
from test.test_inference import (
    get_test_model_config,
    load_model_from_checkpoint_with_lora_support,
    validate_checkpoint,
)
from train_rl_token import create_delta_timestamps

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    class LeRobotDatasetSubset(LeRobotDataset):
        """
        LeRobotDataset 子类：
        修复 episodes 传入子集时，内部以原始 episode_index 查询导致的索引不一致问题。
        """

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 提前构建映射，避免每次 index() 线性查找
            self._episode_pos_map = None
            if self.episodes is not None:
                self._episode_pos_map = {int(ep): i for i, ep in enumerate(self.episodes)}

        def _get_query_indices(self, idx: int, ep_idx: int):
            """
            修正子集 episode 下标映射后，再调用父类查询逻辑。

            Args:
                idx: 数据集中的样本索引。
                ep_idx: 原始 episode 编号。

            Returns:
                父类 _get_query_indices 返回的索引结果。
            """
            if self._episode_pos_map is not None and ep_idx in self._episode_pos_map:
                ep_idx = self._episode_pos_map[ep_idx]
            return super()._get_query_indices(idx, ep_idx)

except ImportError as exc:
    raise ImportError("请先安装 lerobot==0.3.3") from exc


# 固定测试子集（用于 dataset 步骤之外的快速迭代）
FIXED_EPISODE_SLICE = [
    0, 22, 25, 28, 30, 41, 47, 59, 63, 73, 91, 116, 119, 172, 206, 234, 236,
    237, 238, 239, 240, 242, 243, 266, 277, 286, 287, 307, 314, 315, 332, 339,
    348, 350, 352, 353, 365, 366, 368, 370, 390, 393, 400, 411, 420,
]


# ---------------------------------------------------------------------------
# 基础工具：随机种子、张量转 PIL、从样本里取 task / episode 标量
# ---------------------------------------------------------------------------


def set_seed(seed: int = 42) -> None:
    """
    设置随机种子，尽量保证实验可复现。

    Args:
        seed: 随机种子值。

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_pil_image(img_tensor: torch.Tensor):
    """
    LeRobot 图像张量 [C,H,W] 转成 PIL RGB，供 VLA 使用。

    Args:
        img_tensor: 图像张量，形状通常为 [C,H,W] 或 [1,C,H,W]。

    Returns:
        PIL.Image.Image: RGB 图像对象。
    """
    from PIL import Image

    if img_tensor.dim() == 4:
        img_tensor = img_tensor.squeeze(0)
    img_tensor = img_tensor.permute(1, 2, 0)
    img_array = img_tensor.cpu().numpy()
    if img_array.dtype != np.uint8:
        if img_array.max() <= 1.0 and img_array.min() >= 0.0:
            img_array = (img_array * 255).astype(np.uint8)
        else:
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    if img_array.shape[2] == 1:
        img_array = np.repeat(img_array, 3, axis=2)
    return Image.fromarray(img_array, mode="RGB")


def sample_task_index(sample: Dict) -> int:
    """
    从单帧样本里直接读取 task_index。
    数据格式约定固定：sample[\"task_index\"] 为标量张量。

    Args:
        sample: LeRobot 单帧样本字典。

    Returns:
        int: 当前帧所属任务编号。
    """
    return int(sample["task_index"].item())


def sample_episode_index(sample: Dict) -> int:
    """
    从单帧样本里直接读取 episode_index。
    数据格式约定固定：sample[\"episode_index\"] 为标量张量。

    Args:
        sample: LeRobot 单帧样本字典。

    Returns:
        int: 当前帧所属 episode 编号。
    """
    return int(sample["episode_index"].item())


def load_lerobot_dataset(
    dataset_path: Path,
    action_horizon: int,
    episodes: Optional[List[int]] = None,
    repo_id: Optional[str] = None,
) -> LeRobotDataset:
    """
    读 meta/info.json，按 fps 构造 delta_timestamps，打开 LeRobotDataset。
    注意：这里统一使用 LeRobotDatasetSubset，保证 episodes 子集索引稳定。

    Args:
        dataset_path: 本地数据集路径。
        action_horizon: 动作序列长度。
        episodes: episode 子集列表；None 表示加载全部。
        repo_id: 数据集名称；None 时使用目录名。

    Returns:
        LeRobotDataset: 可按索引访问的 LeRobot 数据集对象。
    """
    info_file = dataset_path / "meta" / "info.json"
    if not info_file.exists():
        raise FileNotFoundError(f"missing info.json: {info_file}")
    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    fps = info.get("fps", 10)
    return LeRobotDatasetSubset(
        repo_id=repo_id or dataset_path.name,
        root=str(dataset_path),
        delta_timestamps=create_delta_timestamps(action_horizon, fps),
        episodes=episodes,
    )


def build_single_input(sample: Dict, image_keys: List[str], state_key: str, device: torch.device) -> Dict:
    """
    把一帧 LeRobot 样本转成 VLA 前向所需的 inputs 字典。

    Args:
        sample: 单帧样本。
        image_keys: 要读取的图像键列表。
        state_key: 状态键名。
        device: 目标设备。

    Returns:
        Dict: 包含 images / instructions / states(可选) 的输入字典。
    """
    images = []
    for key in image_keys:
        if key not in sample:
            raise KeyError(f"sample missing image key: {key}")
        images.append(tensor_to_pil_image(sample[key]))
    if len(images) == 1:
        images = [images[0]]
    else:
        images = [images]

    task_text = str(sample.get("task", ""))
    inputs = {"images": images, "instructions": [task_text]}
    if state_key in sample:
        state = sample[state_key]
        if isinstance(state, torch.Tensor):
            if state.dim() == 1:
                state = state.unsqueeze(0)
            inputs["states"] = state.to(device=device, dtype=torch.float32)
    return inputs


def load_trained_modules(
    config_path: str,
    dataset_path: str,
    vla_checkpoint: str,
    rl_checkpoint: str,
    device: torch.device,
):
    """
    加载并校验 VLA 权重与 RL token 编码器。

    Args:
        config_path: 配置文件路径。
        dataset_path: 数据集路径。
        vla_checkpoint: VLA 权重路径。
        rl_checkpoint: RL token 权重路径。
        device: 目标设备。

    Returns:
        tuple: (cfg, model, rl_module)。
    """
    ok, info = validate_checkpoint(vla_checkpoint, config_path, str(device), dataset_path=dataset_path)
    if not ok:
        raise RuntimeError(f"VLA checkpoint validation failed: {info.get('errors', [])}")
    test_config_path = get_test_model_config(config_path, dataset_path=dataset_path, use_test_config=False)
    temp_config_path = test_config_path if test_config_path != config_path else None
    try:
        model, _ = load_model_from_checkpoint_with_lora_support(vla_checkpoint, test_config_path, str(device))
    finally:
        if temp_config_path and Path(temp_config_path).exists():
            Path(temp_config_path).unlink()
    model.eval()

    cfg = load_script_config(config_path, dataset_path=dataset_path)
    rl_cfg = cfg.raw_config.get("model", {}).get("rl_token", {})
    rl_module = RLTokenBottleneck(
        input_dim=model.qwen_vl_interface.get_hidden_dim(),
        model_dim=rl_cfg.get("model_dim"),
        num_encoder_layers=rl_cfg.get("num_encoder_layers", 2),
        num_decoder_layers=rl_cfg.get("num_decoder_layers", 2),
        num_heads=rl_cfg.get("num_heads", 8),
        ffn_dim=rl_cfg.get("ffn_dim"),
        dropout=rl_cfg.get("dropout", 0.1),
        rl_token_dim=rl_cfg.get("rl_token_dim"),
    ).to(device)
    ckpt = torch.load(rl_checkpoint, map_location=device)
    state_dict = ckpt.get("rl_token_state_dict")
    if state_dict is None:
        raise KeyError(f"rl_token_state_dict missing in {rl_checkpoint}")
    rl_module.load_state_dict(state_dict, strict=True)
    rl_module.eval()
    return cfg, model, rl_module


# ---------------------------------------------------------------------------
# 经验池：一条转移、环形缓冲、按 episode 切块写入
# ---------------------------------------------------------------------------


@dataclass
class ReplayTransition:
    sample_index: int
    next_sample_index: int
    episode_id: int
    state: torch.Tensor
    next_state: torch.Tensor
    gt_action_chunk: torch.Tensor
    reward: float
    chunk_return: float
    done: float


class ReplayBuffer:
    """简单 FIFO，存 transition；sample 随机抽 batch。"""

    def __init__(self, capacity: int):
        """
        初始化回放缓冲区。

        Args:
            capacity: 缓冲区最大容量。

        Returns:
            None.
        """
        self.buf: Deque[ReplayTransition] = deque(maxlen=capacity)

    def add(self, transition: ReplayTransition) -> None:
        """
        向缓冲区写入一条 transition。

        Args:
            transition: 单条经验数据。

        Returns:
            None.
        """
        self.buf.append(transition)

    def __len__(self) -> int:
        """
        返回当前缓冲区长度。

        Returns:
            int: 当前经验条数。
        """
        return len(self.buf)

    def sample(self, batch_size: int) -> List[ReplayTransition]:
        """
        随机采样一个 batch。

        Args:
            batch_size: 采样数量。

        Returns:
            List[ReplayTransition]: 采样得到的经验列表。
        """
        return random.sample(list(self.buf), min(batch_size, len(self.buf)))


def collect_task_episode_ids(dataset: LeRobotDataset, task_index: int) -> List[int]:
    """
    遍历全量数据，找出属于指定 task 的 episode_id 列表。

    Args:
        dataset: LeRobot 数据集对象。
        task_index: 目标任务编号。

    Returns:
        List[int]: 去重且排序后的 episode id 列表。
    """
    print(f"[dataset] 正在扫描数据集，筛选 task_index={task_index} 对应的 episode...")
    episode_ids = set()
    for idx in tqdm(range(len(dataset)), desc="scan_task_episodes", leave=False):
        sample = dataset[idx]
        if sample_task_index(sample) != task_index:
            continue
        episode_ids.add(sample_episode_index(sample))
    if not episode_ids:
        raise RuntimeError(f"task_index={task_index} 下没有 episode")
    return sorted(episode_ids)


def build_episode_index(dataset: LeRobotDataset) -> Dict[int, List[int]]:
    """
    在子集数据上构建 episode 索引映射。

    Args:
        dataset: 子集 LeRobot 数据集。

    Returns:
        Dict[int, List[int]]: episode_id -> 帧索引列表（升序）。
    """
    episode_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        episode_to_indices[sample_episode_index(sample)].append(idx)
    for episode_id in episode_to_indices:
        episode_to_indices[episode_id].sort()
    return dict(episode_to_indices)


def build_replay_from_dataset(
    dataset: LeRobotDataset,
    episode_to_indices: Dict[int, List[int]],
    state_key: str,
    chunk_len: int,
    stride: int,
    gamma: float,
    device: torch.device,
    capacity: int,
) -> ReplayBuffer:
    """
    在每个 episode 内按 stride 取起点，截取 chunk_len 步动作作为监督；
    下一帧用 stride 步后的索引，done/reward 按 episode 末尾给稀疏 1。
    Args:
        dataset: 子集 LeRobot 数据集。
        episode_to_indices: episode 到帧索引列表的映射。
        state_key: 状态键名。
        chunk_len: 每次监督使用的动作长度。
        stride: 同 episode 内采样步长。
        device: 张量设备。
        capacity: ReplayBuffer 容量。

    Returns:
        ReplayBuffer: 构建完成的经验池。
    """
    print("[replay] 正在按 episode 构建 ReplayBuffer...")
    replay = ReplayBuffer(capacity=capacity)
    for episode_id, indices in tqdm(
        episode_to_indices.items(),
        total=len(episode_to_indices),
        desc="build_replay",
        leave=False,
    ):
        if len(indices) < chunk_len + 1:
            continue
        for local_start in range(0, len(indices) - 1, stride):
            current_idx = indices[local_start]
            next_idx = indices[min(local_start + 1, len(indices) - 1)]
            sample = dataset[current_idx]
            next_sample = dataset[next_idx]
            actions = sample["action"]
            if actions.shape[0] < chunk_len:
                continue
            action_chunk = actions[:chunk_len].to(dtype=torch.float32)
            state = sample[state_key] if state_key in sample else torch.zeros_like(next_sample[state_key])
            next_state = next_sample[state_key] if state_key in next_sample else state
            terminal = 1.0 if local_start + stride >= len(indices) - 1 else 0.0
            reward = 1.0 if terminal > 0.5 else 0.0
            # chunk 回报：chunk 内若覆盖到 episode 最后一步，则奖励折扣到对应相对位置
            episode_last_local = len(indices) - 1
            chunk_end_local = min(local_start + chunk_len - 1, episode_last_local)
            if local_start <= episode_last_local <= chunk_end_local:
                reward_step_offset = episode_last_local - local_start
                chunk_return = float((gamma ** reward_step_offset) * 1.0)
            else:
                chunk_return = 0.0
            replay.add(
                ReplayTransition(
                    sample_index=current_idx,
                    next_sample_index=next_idx,
                    episode_id=episode_id,
                    state=state.to(device=device, dtype=torch.float32).detach().cpu(),
                    next_state=next_state.to(device=device, dtype=torch.float32).detach().cpu(),
                    gt_action_chunk=action_chunk.detach().cpu(),
                    reward=reward,
                    chunk_return=chunk_return,
                    done=terminal,
                )
            )
    if len(replay) == 0:
        raise RuntimeError("经验池为空：检查 task、chunk_len、stride 或 episode 长度。")
    return replay


# ---------------------------------------------------------------------------
# 训练时现场算 z_rl 与 VLA 参考动作（不写入 replay）
# ---------------------------------------------------------------------------


class OnlineFeatureProvider:
    """按全局帧下标缓存：VLA token -> RL token；VLA predict 前 chunk_len 步作为 reference。"""

    def __init__(
        self,
        dataset: LeRobotDataset,
        image_keys: List[str],
        state_key: str,
        vla_model,
        rl_encoder,
        device: torch.device,
        chunk_len: int,
    ):
        """
        初始化特征提供器。

        Args:
            dataset: 子集 LeRobot 数据集。
            image_keys: 图像键名列表。
            state_key: 状态键名。
            vla_model: VLA 模型实例。
            rl_encoder: RL token 编码器实例。
            device: 目标设备。
            chunk_len: 参考动作长度。

        Returns:
            None.
        """
        self.dataset = dataset
        self.image_keys = image_keys
        self.state_key = state_key
        self.vla_model = vla_model
        self.rl_encoder = rl_encoder
        self.device = device
        self.chunk_len = chunk_len
        self.cache: Dict[int, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

    @torch.no_grad()
    def get(self, sample_index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据样本索引返回 (z_rl, ref_chunk, state)，并使用缓存减少重复计算。

        Args:
            sample_index: 数据集样本索引。

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                z_rl: [1, rl_dim]
                ref_chunk: [1, chunk_len, action_dim]
                state: [1, state_dim]
        """
        if sample_index in self.cache:
            z_rl, ref_chunk, state = self.cache[sample_index]
            # 缓存里存的是 CPU 张量；返回前统一搬到目标设备，避免 batch 内出现 CPU/CUDA 混合
            return (
                z_rl.clone().to(self.device),
                ref_chunk.clone().to(self.device),
                state.clone().to(self.device),
            )

        sample = self.dataset[sample_index]
        inputs = build_single_input(sample, self.image_keys, self.state_key, self.device)
        z_tokens = self.vla_model.extract_vla_tokens(inputs).clone()
        z_rl = self.rl_encoder.encode(z_tokens).float()
        pred = self.vla_model.predict_action(inputs)["normalized_actions"]
        ref_chunk = torch.tensor(pred[:, : self.chunk_len, :], dtype=torch.float32, device=self.device)
        state = sample[self.state_key] if self.state_key in sample else torch.zeros((1, 0), device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(device=self.device, dtype=torch.float32)
        self.cache[sample_index] = (z_rl.detach().cpu(), ref_chunk.detach().cpu(), state.detach().cpu())
        return z_rl, ref_chunk, state


def train_from_batch(
    agent: TD3ChunkAgent,
    provider: OnlineFeatureProvider,
    batch: List[ReplayTransition],
    apply_ref_mask: bool = True,
) -> Dict[str, float]:
    """
    组一个 batch：现场取 z_rl/reference，与 GT chunk 一起喂 TD3 更新。

    Args:
        agent: TD3 agent。
        provider: 特征提供器。
        batch: transition 列表。
        apply_ref_mask: 是否对 reference action 做随机 mask。

    Returns:
        Dict[str, float]: 单步训练指标（loss、q 值等）。
    """
    z_rl = []
    state = []
    ref = []
    action = []
    reward = []
    chunk_return = []
    next_z_rl = []
    next_state = []
    next_ref = []
    done = []
    for tr in batch:
        curr_z, curr_ref, curr_state = provider.get(tr.sample_index)
        nxt_z, nxt_ref, nxt_state = provider.get(tr.next_sample_index)
        z_rl.append(curr_z.squeeze(0))
        state.append(curr_state.squeeze(0))
        ref.append(curr_ref.squeeze(0))
        action.append(tr.gt_action_chunk)
        reward.append([tr.reward])
        chunk_return.append([tr.chunk_return])
        next_z_rl.append(nxt_z.squeeze(0))
        next_state.append(nxt_state.squeeze(0))
        next_ref.append(nxt_ref.squeeze(0))
        done.append([tr.done])

    return agent.train_step(
        z_rl=torch.stack(z_rl, dim=0),
        state=torch.stack(state, dim=0),
        ref_actions=torch.stack(ref, dim=0),
        action=torch.stack(action, dim=0).to(agent.device),
        reward=torch.tensor(reward, dtype=torch.float32, device=agent.device),
        chunk_return=torch.tensor(chunk_return, dtype=torch.float32, device=agent.device),
        next_z_rl=torch.stack(next_z_rl, dim=0),
        next_state=torch.stack(next_state, dim=0),
        next_ref_actions=torch.stack(next_ref, dim=0),
        done=torch.tensor(done, dtype=torch.float32, device=agent.device),
        apply_ref_mask=apply_ref_mask,
    )


def warmup_train(
    agent: TD3ChunkAgent,
    replay: ReplayBuffer,
    provider: OnlineFeatureProvider,
    batch_size: int,
    warmup_updates: int,
    warmup_update_ratio: int,
) -> List[Dict[str, float]]:
    """
    warmup 阶段训练循环。

    Args:
        agent: TD3 agent。
        replay: 经验池。
        provider: 特征提供器。
        batch_size: 每步采样 batch 大小。
        warmup_updates: warmup 更新步数。

    Returns:
        List[Dict[str, float]]: 每步训练日志。
    """
    print(
        f"[warmup] 开始 warmup 训练，共 {warmup_updates} 次外层更新，"
        f"每次更新内梯度步数 G={warmup_update_ratio}..."
    )
    logs: List[Dict[str, float]] = []
    for step in tqdm(range(warmup_updates), desc="warmup_updates", leave=False):
        for g in range(warmup_update_ratio):
            batch = replay.sample(batch_size)
            metrics = train_from_batch(agent, provider, batch, apply_ref_mask=True)
            metrics["phase"] = "warmup"
            metrics["step"] = float(step + 1)
            metrics["warmup_inner_g"] = float(g + 1)
            logs.append(metrics)
    return logs


def online_train_loop(
    agent: TD3ChunkAgent,
    replay: ReplayBuffer,
    provider: OnlineFeatureProvider,
    batch_size: int,
    online_train_episodes: int,
) -> List[Dict[str, float]]:
    """
    online 阶段训练循环（按 episode 轮次）。

    Args:
        agent: TD3 agent。
        replay: 经验池。
        provider: 特征提供器。
        batch_size: 每个 batch 的样本数。
        online_train_episodes: 在线训练 episode 数。

    Returns:
        List[Dict[str, float]]: 在线训练日志。
    """
    print(f"[online] 开始在线训练，共 {online_train_episodes} 个 episode...")
    logs: List[Dict[str, float]] = []
    transitions = list(replay.buf)
    transitions_by_episode: Dict[int, List[ReplayTransition]] = defaultdict(list)
    for tr in transitions:
        transitions_by_episode[tr.episode_id].append(tr)
    episode_ids = sorted(transitions_by_episode.keys())
    for ep in tqdm(range(online_train_episodes), desc="online_episodes"):
        ep_id = episode_ids[ep % len(episode_ids)]
        ep_transitions = transitions_by_episode[ep_id]
        random.shuffle(ep_transitions)
        for start in tqdm(
            range(0, len(ep_transitions), batch_size),
            desc=f"episode_{ep + 1}_batches",
            leave=False,
        ):
            batch = ep_transitions[start : start + batch_size]
            if not batch:
                continue
            metrics = train_from_batch(agent, provider, batch, apply_ref_mask=True)
            metrics["phase"] = "online"
            metrics["episode"] = float(ep + 1)
            logs.append(metrics)
        if (ep + 1) % max(1, online_train_episodes // 10) == 0:
            print(f"[online] 已完成 episode {ep + 1}/{online_train_episodes}")
    return logs


@torch.no_grad()
def eval_with_vla_reference(
    agent: TD3ChunkAgent,
    replay: ReplayBuffer,
    provider: OnlineFeatureProvider,
    max_eval_samples: int = 256,
) -> Dict[str, float]:
    """
    评估 TD3 输出与 GT / VLA reference 的误差，并统计 Q 值。

    Args:
        agent: TD3 agent。
        replay: 经验池。
        provider: 特征提供器。
        max_eval_samples: 最大评估样本数。

    Returns:
        Dict[str, float]: mse / mae / l2 / ref_mse / eval_q_mean。
    """
    print(f"[eval] 开始评估，最多评估 {max_eval_samples} 条样本...")
    selected = list(replay.buf)[:max_eval_samples]
    mse_vals = []
    mae_vals = []
    l2_vals = []
    ref_mse_vals = []
    q_vals = []

    for tr in tqdm(selected, desc="eval_samples", leave=False):
        z_rl, ref_chunk, state = provider.get(tr.sample_index)
        gt = tr.gt_action_chunk.unsqueeze(0).to(agent.device)
        pred = agent.act(z_rl, state, ref_chunk, deterministic=True, apply_ref_mask=False)
        q1, q2 = agent.critic(z_rl.to(agent.device), state.to(agent.device), pred.to(agent.device))
        q_vals.append(torch.min(q1, q2).mean().item())

        mse_vals.append(F.mse_loss(pred, gt).item())
        mae_vals.append(F.l1_loss(pred, gt).item())
        l2_vals.append((pred - gt).pow(2).sum(dim=-1).sqrt().mean().item())
        ref_mse_vals.append(F.mse_loss(ref_chunk.to(agent.device), gt).item())

    if not mse_vals:
        raise RuntimeError("没有可用于评估的样本。")
    return {
        "mse": float(np.mean(mse_vals)),
        "mae": float(np.mean(mae_vals)),
        "mean_l2_per_step": float(np.mean(l2_vals)),
        "ref_mse": float(np.mean(ref_mse_vals)),
        "eval_q_mean": float(np.mean(q_vals)),
    }


def export_q_logs(logs: List[Dict[str, float]], output_dir: Path, base_name: str = "q_curve") -> None:
    """
    把训练日志写 CSV；若装了 matplotlib 再画一张 Q 曲线图。

    Args:
        logs: 训练日志列表。
        output_dir: 输出目录。
        base_name: 输出文件名前缀。

    Returns:
        None.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{base_name}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "phase",
                "step",
                "episode",
                "warmup_inner_g",
                "q1_mean",
                "q2_mean",
                "target_q_mean",
                "chunk_return_mean",
                "critic_loss",
                "actor_loss",
                "actor_constraint_loss",
                "pred_vs_ref_mse",
            ],
        )
        writer.writeheader()
        for row in logs:
            writer.writerow(
                {
                    "phase": row.get("phase", ""),
                    "step": row.get("step", ""),
                    "episode": row.get("episode", ""),
                    "warmup_inner_g": row.get("warmup_inner_g", ""),
                    "q1_mean": row.get("q1_mean", ""),
                    "q2_mean": row.get("q2_mean", ""),
                    "target_q_mean": row.get("target_q_mean", ""),
                    "chunk_return_mean": row.get("chunk_return_mean", ""),
                    "critic_loss": row.get("critic_loss", ""),
                    "actor_loss": row.get("actor_loss", ""),
                    "actor_constraint_loss": row.get("actor_constraint_loss", ""),
                    "pred_vs_ref_mse": row.get("pred_vs_ref_mse", ""),
                }
            )

    try:
        import matplotlib.pyplot as plt

        x = np.arange(len(logs))
        q1 = np.array([row.get("q1_mean", 0.0) for row in logs], dtype=np.float32)
        q2 = np.array([row.get("q2_mean", 0.0) for row in logs], dtype=np.float32)
        tq = np.array([row.get("target_q_mean", 0.0) for row in logs], dtype=np.float32)
        plt.figure(figsize=(9, 5))
        plt.plot(x, q1, label="q1_mean")
        plt.plot(x, q2, label="q2_mean")
        plt.plot(x, tq, label="target_q_mean")
        plt.title("TD3 Q-value Curve")
        plt.xlabel("Update Step")
        plt.ylabel("Q Value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{base_name}.png", dpi=150)
        plt.close()
    except Exception as exc:
        print(f"[warn] 未生成 {base_name}.png: {exc}")


def save_td3_agent(path: Path, agent: TD3ChunkAgent, args, cfg) -> None:
    """
    保存 actor/critic 与 TD3 超参，便于 --load-agent 续跑。

    Args:
        path: 保存路径。
        agent: TD3 agent。
        args: 运行参数对象。
        cfg: 脚本配置对象。

    Returns:
        None.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rl_token_dim = cfg.raw_config.get("model", {}).get("rl_token", {}).get("rl_token_dim", 256)
    c = TD3ChunkConfig(
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        fixed_std=args.actor_std,
        max_action=args.max_action,
        ref_mask_prob=args.ref_mask_prob,
        policy_constraint_beta=args.policy_constraint_beta,
        use_chunk_return_target=args.use_chunk_return_target,
        hidden_dims=[args.hidden_dim, args.hidden_dim],
    )
    meta = {
        "rl_token_dim": rl_token_dim,
        "state_dim": cfg.state_dim,
        "action_dim": cfg.action_dim,
        "chunk_size": args.chunk_len,
        "td3_cfg": {
            "gamma": c.gamma,
            "tau": c.tau,
            "actor_lr": c.actor_lr,
            "critic_lr": c.critic_lr,
            "policy_noise": c.policy_noise,
            "noise_clip": c.noise_clip,
            "policy_delay": c.policy_delay,
            "fixed_std": c.fixed_std,
            "max_action": c.max_action,
            "ref_mask_prob": c.ref_mask_prob,
            "policy_constraint_beta": c.policy_constraint_beta,
            "use_chunk_return_target": c.use_chunk_return_target,
            "hidden_dims": c.hidden_dims,
        },
    }
    torch.save(
        {
            "meta": meta,
            "actor": agent.actor.state_dict(),
            "actor_target": agent.actor_target.state_dict(),
            "critic": agent.critic.state_dict(),
            "critic_target": agent.critic_target.state_dict(),
            "total_updates": int(agent.total_updates),
        },
        path,
    )
    print(f"已保存 TD3 权重: {path}")


def load_td3_agent(path: Path, device: torch.device) -> TD3ChunkAgent:
    """
    从 save_td3_agent 写出的文件恢复 TD3ChunkAgent。

    Args:
        path: 权重文件路径。
        device: 目标设备。

    Returns:
        TD3ChunkAgent: 恢复后的 agent。
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    meta = payload["meta"]
    td3_cfg = TD3ChunkConfig(**meta["td3_cfg"])
    agent = TD3ChunkAgent(
        rl_token_dim=int(meta["rl_token_dim"]),
        state_dim=int(meta["state_dim"]),
        action_dim=int(meta["action_dim"]),
        chunk_size=int(meta["chunk_size"]),
        cfg=td3_cfg,
        device=device,
    )
    agent.actor.load_state_dict(payload["actor"])
    agent.actor_target.load_state_dict(payload["actor_target"])
    agent.critic.load_state_dict(payload["critic"])
    agent.critic_target.load_state_dict(payload["critic_target"])
    agent.total_updates = int(payload.get("total_updates", 0))
    print(f"已加载 TD3 权重: {path}, total_updates={agent.total_updates}")
    return agent


# ---------------------------------------------------------------------------
# 入口：按 --step 分支，每段内顺序写清（少套一层函数名）
# ---------------------------------------------------------------------------


def build_argparser(td3_defaults: Dict) -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    Returns:
        argparse.ArgumentParser: 参数解析器对象。
    """
    p = argparse.ArgumentParser(description="LeRobot 上 TD3 模拟在线训练")
    p.add_argument(
        "--step",
        type=str,
        default="full",
        choices=("dataset", "models", "replay", "warmup", "online", "eval", "full"),
        help="分步调试或 full 全流程",
    )
    p.add_argument("--save-agent", type=str, default=None, help="保存 TD3 权重路径（.pt）")
    p.add_argument("--load-agent", type=str, default=None, help="加载已有 TD3 权重")
    p.add_argument("--config", type=str, default="config.yaml")
    p.add_argument("--dataset", type=str, default="./dataset/libero_object")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--task-index", type=int, default=0)
    p.add_argument("--chunk-len", type=int, default=td3_defaults["chunk_len"])
    p.add_argument("--stride", type=int, default=td3_defaults["stride"])
    p.add_argument("--batch-size", type=int, default=td3_defaults["batch_size"])
    p.add_argument("--replay-capacity", type=int, default=td3_defaults["replay_capacity"])
    p.add_argument("--warmup-updates", type=int, default=td3_defaults["warmup_updates"])
    p.add_argument("--warmup-update-ratio", type=int, default=td3_defaults["warmup_update_ratio"])
    p.add_argument("--online-train-episodes", type=int, default=td3_defaults["online_train_episodes"])
    p.add_argument("--max-eval-samples", type=int, default=td3_defaults["max_eval_samples"])
    p.add_argument("--vla-checkpoint", type=str, default=td3_defaults["vla_checkpoint"])
    p.add_argument("--rl-checkpoint", type=str, default=td3_defaults["rl_checkpoint"])
    p.add_argument("--output-dir", type=str, default=td3_defaults["output_dir"])
    p.add_argument("--gamma", type=float, default=td3_defaults["gamma"])
    p.add_argument("--tau", type=float, default=td3_defaults["tau"])
    p.add_argument("--actor-lr", type=float, default=td3_defaults["actor_lr"])
    p.add_argument("--critic-lr", type=float, default=td3_defaults["critic_lr"])
    p.add_argument("--policy-noise", type=float, default=td3_defaults["policy_noise"])
    p.add_argument("--noise-clip", type=float, default=td3_defaults["noise_clip"])
    p.add_argument("--policy-delay", type=int, default=td3_defaults["policy_delay"])
    p.add_argument("--actor-std", type=float, default=td3_defaults["actor_std"])
    p.add_argument("--max-action", type=float, default=td3_defaults["max_action"])
    p.add_argument("--ref-mask-prob", type=float, default=td3_defaults["ref_mask_prob"])
    p.add_argument("--policy-constraint-beta", type=float, default=td3_defaults["policy_constraint_beta"])
    p.add_argument(
        "--use-chunk-return-target",
        action=argparse.BooleanOptionalAction,
        default=td3_defaults["use_chunk_return_target"],
    )
    p.add_argument("--hidden-dim", type=int, default=td3_defaults["hidden_dim"])
    return p


def _execute(args: argparse.Namespace) -> None:
    """
    根据 args.step 执行对应阶段（分步调试与 full 流程共用）。

    Args:
        args: 命令行参数命名空间。

    Returns:
        None.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 只调用一次 load_script_config
    cfg_once = load_script_config(args.config, dataset_path=args.dataset, seed=args.seed)

    out_dir = Path(args.output_dir)
    warmup_ckpt_path = out_dir / "warmup_agent.pt"
    print("=" * 80)
    print(f"[main] 启动脚本，step={args.step}, device={device}")
    print("=" * 80)

    # ---------- 只检查数据：task 对应哪些 episode，写 json ----------
    if args.step == "dataset":
        print("[main] 当前任务：仅检查数据集与 task 对应 episode 列表")
        set_seed(args.seed)
        cfg = cfg_once
        full_ds = load_lerobot_dataset(Path(args.dataset), cfg.action_horizon)
        selected = collect_task_episode_ids(full_ds, args.task_index)
        sub_ds = load_lerobot_dataset(Path(args.dataset), cfg.action_horizon, episodes=selected)
        print(
            f"[dataset] task_index={args.task_index}, episodes={len(selected)}, len(sub_ds)={len(sub_ds)}"
        )
        if len(sub_ds) > 0:
            keys = sorted(sub_ds[0].keys())
            print(f"[dataset] 首帧字段示例: {keys[:24]}{'...' if len(keys) > 24 else ''}")
        out_dir.mkdir(parents=True, exist_ok=True)
        ep_path = out_dir / "selected_episodes.json"
        with open(ep_path, "w", encoding="utf-8") as f:
            json.dump({"task_index": args.task_index, "episodes": selected}, f, indent=2)
        print(f"[dataset] 已写入 {ep_path}")
        print("[dataset] ✅ 测试成功：数据集与 episode 切片检查完成")
        return

    # ---------- 只加载 VLA + RL token，不做训练 ----------
    if args.step == "models":
        print("[main] 当前任务：仅加载 VLA 与 RL token 模型")
        set_seed(args.seed)
        cfg, _, _ = load_trained_modules(
            args.config, args.dataset, args.vla_checkpoint, args.rl_checkpoint, device
        )
        print(
            f"[models] device={device}, state_dim={cfg.state_dim}, action_dim={cfg.action_dim}, "
            f"action_horizon={cfg.action_horizon}"
        )
        print("[models] ✅ 测试成功：模型与权重加载完成")
        return

    # ---------- 只构建经验池（不加载大模型）----------
    if args.step == "replay":
        print("[main] 当前任务：仅构建 ReplayBuffer")
        cfg = cfg_once
        dataset_path_obj = Path(args.dataset).resolve()
        info_file = dataset_path_obj / "meta" / "info.json"
        if not info_file.exists():
            raise ValueError(f"本地数据集路径不存在或无效: {info_file}")

        with open(info_file, "r", encoding="utf-8") as f:
            info = json.load(f)
        fps = info.get("fps", 10)
        delta_timestamps = create_delta_timestamps(cfg.action_horizon, fps)
        dataset_name = dataset_path_obj.name
        root_path_str = str(dataset_path_obj)

        try:
            sub_ds = LeRobotDatasetSubset(
                repo_id=dataset_name,
                root=root_path_str,
                delta_timestamps=delta_timestamps,
                episodes=FIXED_EPISODE_SLICE,
            )
            print(f"  ✓ 固定测试集创建成功: repo_id={dataset_name}, root={root_path_str}")
            print(f"  ✓ 固定 episode 数量: {len(FIXED_EPISODE_SLICE)}")
        except Exception as e:
            print(f"  ✗ 创建固定测试集失败: {e}")
            import traceback
            traceback.print_exc()
            raise

        try:
            _normalizer = create_normalizer_from_lerobot_meta(
                sub_ds,
                state_key=cfg.state_key,
                action_key="action",
            )
            print("  ✓ 归一化器已从 meta.episodes_stats 创建")
            if _normalizer.action_min is not None:
                print(
                    "  Action 范围: "
                    f"[{_normalizer.action_min.min():.4f}, {_normalizer.action_max.max():.4f}]"
                )
            if _normalizer.state_min is not None:
                print(
                    "  State 范围: "
                    f"[{_normalizer.state_min.min():.4f}, {_normalizer.state_max.max():.4f}]"
                )
        except Exception as e:
            print(f"  从 meta.episodes_stats 创建归一化器失败: {e}，尝试从 episodes_stats.jsonl 创建")
            try:
                _normalizer = create_normalizer_from_dataset(dataset_path_obj)
                print("  ✓ 归一化器已从 episodes_stats.jsonl 创建")
            except Exception as e2:
                print(f"  ✗ 归一化器创建失败: {e2}")
                print("  警告: 将不使用归一化，训练可能不稳定")
                _normalizer = None

        ep_map = build_episode_index(sub_ds)
        replay = build_replay_from_dataset(
            sub_ds,
            ep_map,
            cfg.state_key,
            args.chunk_len,
            args.stride,
            args.gamma,
            device,
            args.replay_capacity,
        )
        print(f"[replay] len(replay)={len(replay)}, state_key={cfg.state_key}")
        tr0 = next(iter(replay.buf))
        print(
            f"[replay] 首条 transition: idx={tr0.sample_index}->{tr0.next_sample_index}, "
            f"gt_chunk.shape={tuple(tr0.gt_action_chunk.shape)}"
        )
        print(f"[replay] ✅ 测试成功：ReplayBuffer 构建完成，样本数={len(replay)}")
        return

    # 以下步骤需要 VLA + RL + 子集数据 + 经验池；先统一准备好 cfg / 模型 / replay / provider 所需的数据集
    set_seed(args.seed)
    print("[main] 当前任务：准备训练/评估所需模型与数据...")
    cfg, vla_model, rl_encoder = load_trained_modules(
        args.config, args.dataset, args.vla_checkpoint, args.rl_checkpoint, device
    )
    dataset_path_obj = Path(args.dataset).resolve()
    info_file = dataset_path_obj / "meta" / "info.json"
    if not info_file.exists():
        raise ValueError(f"本地数据集路径不存在或无效: {info_file}")

    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    fps = info.get("fps", 10)
    delta_timestamps = create_delta_timestamps(cfg.action_horizon, fps)
    dataset_name = dataset_path_obj.name
    root_path_str = str(dataset_path_obj)

    try:
        sub_ds = LeRobotDatasetSubset(
            repo_id=dataset_name,
            root=root_path_str,
            delta_timestamps=delta_timestamps,
            episodes=FIXED_EPISODE_SLICE,
        )
        print(f"  ✓ 固定测试集创建成功: repo_id={dataset_name}, root={root_path_str}")
        print(f"  ✓ 固定 episode 数量: {len(FIXED_EPISODE_SLICE)}")
    except Exception as e:
        print(f"  ✗ 创建固定测试集失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    try:
        _normalizer = create_normalizer_from_lerobot_meta(
            sub_ds,
            state_key=cfg.state_key,
            action_key="action",
        )
        print("  ✓ 归一化器已从 meta.episodes_stats 创建")
        if _normalizer.action_min is not None:
            print(
                "  Action 范围: "
                f"[{_normalizer.action_min.min():.4f}, {_normalizer.action_max.max():.4f}]"
            )
        if _normalizer.state_min is not None:
            print(
                "  State 范围: "
                f"[{_normalizer.state_min.min():.4f}, {_normalizer.state_max.max():.4f}]"
            )
    except Exception as e:
        print(f"  从 meta.episodes_stats 创建归一化器失败: {e}，尝试从 episodes_stats.jsonl 创建")
        try:
            _normalizer = create_normalizer_from_dataset(dataset_path_obj)
            print("  ✓ 归一化器已从 episodes_stats.jsonl 创建")
        except Exception as e2:
            print(f"  ✗ 归一化器创建失败: {e2}")
            print("  警告: 将不使用归一化，训练可能不稳定")
            _normalizer = None

    ep_map = build_episode_index(sub_ds)
    replay = build_replay_from_dataset(
        sub_ds,
        ep_map,
        cfg.state_key,
        args.chunk_len,
        args.stride,
        args.gamma,
        device,
        args.replay_capacity,
    )
    rl_token_dim = cfg.raw_config.get("model", {}).get("rl_token", {}).get("rl_token_dim", 256)
    td3_cfg = TD3ChunkConfig(
        gamma=args.gamma,
        tau=args.tau,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        policy_delay=args.policy_delay,
        fixed_std=args.actor_std,
        max_action=args.max_action,
        ref_mask_prob=args.ref_mask_prob,
        policy_constraint_beta=args.policy_constraint_beta,
        use_chunk_return_target=args.use_chunk_return_target,
        hidden_dims=[args.hidden_dim, args.hidden_dim],
    )
    provider = OnlineFeatureProvider(
        sub_ds,
        cfg.image_keys,
        cfg.state_key,
        vla_model,
        rl_encoder,
        device,
        args.chunk_len,
    )
    default_warmup = max(1, int(np.ceil(len(replay) / max(args.batch_size, 1))))
    warmup_n = args.warmup_updates if args.warmup_updates > 0 else default_warmup

    # ---------- 只 warmup ----------
    if args.step == "warmup":
        print("[main] 当前任务：执行 warmup 训练")
        print(
            f"[warmup] episodes={len(FIXED_EPISODE_SLICE)}, replay={len(replay)}, "
            f"warmup_updates={warmup_n}"
        )
        agent = TD3ChunkAgent(
            rl_token_dim, cfg.state_dim, cfg.action_dim, args.chunk_len, td3_cfg, device
        )
        logs = warmup_train(agent, replay, provider, args.batch_size, warmup_n, args.warmup_update_ratio)
        export_q_logs(logs, out_dir, base_name="q_curve_warmup")
        # warmup 结束后默认保存一份固定权重，供后续 online/eval 直接加载
        save_td3_agent(warmup_ckpt_path, agent, args, cfg)
        if args.save_agent:
            save_td3_agent(Path(args.save_agent), agent, args, cfg)
        print(f"[warmup] ✅ 测试成功：warmup 完成，更新步数={len(logs)}")
        print(f"[warmup] 已保存默认 warmup 权重: {warmup_ckpt_path}")
        return

    # ---------- 只 online（无权重则先在同进程里 warmup 一遍）----------
    if args.step == "online":
        print("[main] 当前任务：执行 online 训练")
        if args.load_agent:
            agent = load_td3_agent(Path(args.load_agent), device)
        elif warmup_ckpt_path.exists():
            print(f"[online] 使用默认 warmup 权重: {warmup_ckpt_path}")
            agent = load_td3_agent(warmup_ckpt_path, device)
        else:
            print("[online] 未找到 warmup 权重，先执行一次 warmup 并保存默认权重")
            agent = TD3ChunkAgent(
                rl_token_dim, cfg.state_dim, cfg.action_dim, args.chunk_len, td3_cfg, device
            )
            warmup_train(agent, replay, provider, args.batch_size, warmup_n, args.warmup_update_ratio)
            save_td3_agent(warmup_ckpt_path, agent, args, cfg)
        logs = online_train_loop(
            agent, replay, provider, args.batch_size, args.online_train_episodes
        )
        export_q_logs(logs, out_dir, base_name="q_curve_online")
        if args.save_agent:
            save_td3_agent(Path(args.save_agent), agent, args, cfg)
        print(
            f"[online] ✅ 测试成功：online 训练完成，episode={args.online_train_episodes}，"
            f"更新步数={len(logs)}"
        )
        return

    # ---------- 只评估 ----------
    if args.step == "eval":
        print("[main] 当前任务：执行评估")
        if args.load_agent:
            agent = load_td3_agent(Path(args.load_agent), device)
        elif warmup_ckpt_path.exists():
            print(f"[eval] 使用默认 warmup 权重: {warmup_ckpt_path}")
            agent = load_td3_agent(warmup_ckpt_path, device)
        else:
            raise FileNotFoundError(
                f"[eval] 未找到可用权重：{warmup_ckpt_path}。"
                "请先执行 --step warmup，或通过 --load-agent 指定权重文件。"
            )
        metrics = eval_with_vla_reference(agent, replay, provider, args.max_eval_samples)
        print(
            f"[eval] mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} "
            f"l2={metrics['mean_l2_per_step']:.6f} ref_mse={metrics['ref_mse']:.6f} "
            f"q_mean={metrics['eval_q_mean']:.6f}"
        )
        print("[eval] ✅ 测试成功：评估完成")
        return

    # ---------- 全流程：warmup -> online -> 评估 -> 可选存盘 ----------
    if args.step == "full":
        print("[main] 当前任务：执行 full 全流程（warmup -> online -> eval）")
        print(
            f"[full] episodes={len(FIXED_EPISODE_SLICE)}, replay={len(replay)}, "
            f"warmup={warmup_n}, online_ep={args.online_train_episodes}"
        )
        agent = TD3ChunkAgent(
            rl_token_dim, cfg.state_dim, cfg.action_dim, args.chunk_len, td3_cfg, device
        )
        all_logs: List[Dict[str, float]] = []
        all_logs.extend(
            warmup_train(agent, replay, provider, args.batch_size, warmup_n, args.warmup_update_ratio)
        )
        all_logs.extend(
            online_train_loop(agent, replay, provider, args.batch_size, args.online_train_episodes)
        )
        export_q_logs(all_logs, out_dir, base_name="q_curve")
        metrics = eval_with_vla_reference(agent, replay, provider, args.max_eval_samples)
        print(
            f"[full] mse={metrics['mse']:.6f} mae={metrics['mae']:.6f} "
            f"l2={metrics['mean_l2_per_step']:.6f} ref_mse={metrics['ref_mse']:.6f} "
            f"q_mean={metrics['eval_q_mean']:.6f}"
        )
        if args.save_agent:
            save_td3_agent(Path(args.save_agent), agent, args, cfg)
        print(
            f"[full] ✅ 测试成功：全流程完成（warmup={warmup_n}，online_ep={args.online_train_episodes}）"
        )
        return

    raise RuntimeError(f"未知 step: {args.step}")


def main():
    """
    脚本命令行入口。

    Returns:
        None.
    """
    # 先解析 --config，再用 config 里的 online_rl.td3 直接作为 argparse 默认值
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="config.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    cfg_for_defaults = load_script_config(pre_args.config)
    td3_defaults = cfg_for_defaults.raw_config["training"]["online_rl"]["td3"]

    _execute(build_argparser(td3_defaults).parse_args())


def run_online_td3_sim(args: argparse.Namespace) -> None:
    """
    兼容旧接口：强制跑完整流程（等同 --step full）。

    Args:
        args: 参数命名空间。

    Returns:
        None.
    """
    args = argparse.Namespace(**vars(args))
    args.step = "full"
    _execute(args)


if __name__ == "__main__":
    """
    测试流程：
    python -m test.test_online_td3_sim --step dataset --dataset ./dataset/libero_object
    python -m test.test_online_td3_sim --step models
    python -m test.test_online_td3_sim --step replay

    python -m test.test_online_td3_sim --step warmup 
    python -m test.test_online_td3_sim --step online
    """
    main()
