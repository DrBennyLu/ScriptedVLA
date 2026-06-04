"""
Offline TD3 replay buffer, frozen VLA/RL-token features, and training utilities.

author: Benny Lu
license: MIT
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.ScriptedVLA.model import RLTokenBottleneck, TD3ChunkAgent, TD3ChunkConfig
from src.ScriptedVLA.utils import Normalizer, load_script_config
from test.test_inference import (
    get_test_model_config,
    load_model_from_checkpoint_with_lora_support,
    validate_checkpoint,
)
from train_rl_token import create_delta_timestamps

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    class LeRobotDatasetSubset(LeRobotDataset):
        """Fix episode index mapping when loading an episode subset."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._episode_pos_map = None
            if self.episodes is not None:
                self._episode_pos_map = {int(ep): i for i, ep in enumerate(self.episodes)}

        def _get_query_indices(self, idx: int, ep_idx: int):
            if self._episode_pos_map is not None and ep_idx in self._episode_pos_map:
                ep_idx = self._episode_pos_map[ep_idx]
            return super()._get_query_indices(idx, ep_idx)

except ImportError as exc:
    raise ImportError("lerobot is required: pip install lerobot==0.3.3") from exc


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_pil_image(img_tensor: torch.Tensor):
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
    return int(sample["task_index"].item())


def sample_episode_index(sample: Dict) -> int:
    return int(sample["episode_index"].item())


def load_lerobot_dataset(
    dataset_path: Path,
    action_horizon: int,
    episodes: Optional[List[int]] = None,
    repo_id: Optional[str] = None,
) -> LeRobotDataset:
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


def load_normalizer_from_vla_checkpoint(vla_checkpoint: str) -> Optional[Normalizer]:
    """Load normalizer saved in a VLA checkpoint (for WebSocket eval/collect)."""
    path = Path(vla_checkpoint).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "normalizer" not in ckpt:
        return None
    return Normalizer.from_dict(ckpt["normalizer"])


def build_single_input(
    sample: Dict,
    image_keys: List[str],
    state_key: str,
    device: torch.device,
) -> Dict:
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


def _freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def _rl_token_kwargs_from_checkpoint(
    ckpt: Dict[str, Any],
    vla_hidden_dim: int,
    network_cfg: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    network_cfg = network_cfg or {}
    model_dim = ckpt.get("model_dim") or network_cfg.get("model_dim") or vla_hidden_dim
    rl_token_dim = ckpt.get("rl_token_dim") or network_cfg.get("rl_token_dim") or vla_hidden_dim
    if model_dim is None:
        model_dim = vla_hidden_dim
    if rl_token_dim is None:
        rl_token_dim = vla_hidden_dim
    return {
        "input_dim": vla_hidden_dim,
        "model_dim": int(model_dim),
        "num_encoder_layers": int(network_cfg.get("num_encoder_layers", 2)),
        "num_decoder_layers": int(network_cfg.get("num_decoder_layers", 2)),
        "num_heads": int(network_cfg.get("num_heads", 8)),
        "ffn_dim": network_cfg.get("ffn_dim"),
        "dropout": float(network_cfg.get("dropout", 0.1)),
        "rl_token_dim": int(rl_token_dim),
    }


def load_frozen_vla_and_rl_token(
    config_path: str,
    dataset_path: str,
    vla_checkpoint: str,
    rl_token_checkpoint: str,
    device: torch.device,
    *,
    validate_vla: bool = True,
    rl_token_network_cfg: Optional[Dict[str, Any]] = None,
):
    """
    Load VLA and RL token extractor; freeze all parameters.

    Returns:
        (cfg, vla_model, rl_encoder)
    """
    rl_path = Path(rl_token_checkpoint).expanduser().resolve()
    if not rl_path.is_file():
        raise FileNotFoundError(f"rl_token_checkpoint not found: {rl_path}")

    if validate_vla:
        ok, info = validate_checkpoint(
            vla_checkpoint, config_path, str(device), dataset_path=dataset_path
        )
        if not ok:
            raise RuntimeError(f"VLA checkpoint validation failed: {info.get('errors', [])}")

    test_config_path = get_test_model_config(
        config_path, dataset_path=dataset_path, use_test_config=False
    )
    temp_config_path = test_config_path if test_config_path != config_path else None
    try:
        model, _ = load_model_from_checkpoint_with_lora_support(
            vla_checkpoint, test_config_path, str(device)
        )
    finally:
        if temp_config_path and Path(temp_config_path).exists():
            Path(temp_config_path).unlink()

    _freeze_module(model)

    try:
        ckpt = torch.load(rl_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(rl_path, map_location=device)

    state_dict = ckpt.get("rl_token_state_dict")
    if state_dict is None:
        raise KeyError(f"rl_token_state_dict missing in {rl_path}")

    vla_hidden_dim = model.qwen_vl_interface.get_hidden_dim()
    rl_kwargs = _rl_token_kwargs_from_checkpoint(ckpt, vla_hidden_dim, rl_token_network_cfg)
    rl_module = RLTokenBottleneck(**rl_kwargs).to(device)
    rl_module.load_state_dict(state_dict, strict=True)
    _freeze_module(rl_module)

    cfg = load_script_config(config_path, dataset_path=dataset_path)
    return cfg, model, rl_module


# Backward-compatible alias for test script
def load_trained_modules(
    config_path: str,
    dataset_path: str,
    vla_checkpoint: str,
    rl_checkpoint: str,
    device: torch.device,
):
    return load_frozen_vla_and_rl_token(
        config_path,
        dataset_path,
        vla_checkpoint,
        rl_checkpoint,
        device,
        validate_vla=True,
        rl_token_network_cfg=None,
    )


@dataclass
class ReplayTransition:
    """Offline transition: (x_t, a_t, a_t_hat, r_t, x_{t+C}) with precomputed z_rl."""

    sample_index: int
    next_sample_index: int
    episode_id: int
    z_rl: torch.Tensor
    state: torch.Tensor
    action: torch.Tensor
    ref_action: torch.Tensor
    reward: float
    chunk_return: float
    done: float
    next_z_rl: torch.Tensor
    next_state: torch.Tensor
    next_ref_action: torch.Tensor


REPLAY_CACHE_VERSION = 1


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buf: Deque[ReplayTransition] = deque(maxlen=capacity)

    def add(self, transition: ReplayTransition) -> None:
        self.buf.append(transition)

    def __len__(self) -> int:
        return len(self.buf)

    def sample(self, batch_size: int) -> List[ReplayTransition]:
        return random.sample(list(self.buf), min(batch_size, len(self.buf)))


def sample_mixed_batch(
    online_replay: ReplayBuffer,
    offline_replay: Optional[ReplayBuffer],
    batch_size: int,
    online_sample_ratio: float,
) -> List[ReplayTransition]:
    """
    Build a training batch by Bernoulli mixing per slot:
    P(sample from online) = online_sample_ratio, else offline replay_cache.
    Falls back to the non-empty buffer when the other is empty.
    """
    ratio = float(np.clip(online_sample_ratio, 0.0, 1.0))
    batch_size = max(1, int(batch_size))
    has_online = len(online_replay) > 0
    has_offline = offline_replay is not None and len(offline_replay) > 0

    if not has_online and not has_offline:
        return []
    if not has_offline:
        return online_replay.sample(batch_size)
    if not has_online:
        return offline_replay.sample(batch_size)

    online_list = list(online_replay.buf)
    offline_list = list(offline_replay.buf)
    batch: List[ReplayTransition] = []
    for _ in range(batch_size):
        if random.random() < ratio:
            batch.append(random.choice(online_list))
        else:
            batch.append(random.choice(offline_list))
    return batch


def sample_mixed_batch_with_sources(
    online_replay: ReplayBuffer,
    offline_replay: Optional[ReplayBuffer],
    batch_size: int,
    online_sample_ratio: float,
) -> tuple[List[ReplayTransition], List[str]]:
    """
    Like sample_mixed_batch but tags each slot as 'online' or 'offline' for split metrics.
    """
    ratio = float(np.clip(online_sample_ratio, 0.0, 1.0))
    batch_size = max(1, int(batch_size))
    has_online = len(online_replay) > 0
    has_offline = offline_replay is not None and len(offline_replay) > 0

    if not has_online and not has_offline:
        return [], []
    if not has_offline:
        batch = online_replay.sample(batch_size)
        return batch, ["online"] * len(batch)
    if not has_online:
        batch = offline_replay.sample(batch_size)
        return batch, ["offline"] * len(batch)

    online_list = list(online_replay.buf)
    offline_list = list(offline_replay.buf)
    batch: List[ReplayTransition] = []
    sources: List[str] = []
    for _ in range(batch_size):
        if random.random() < ratio:
            batch.append(random.choice(online_list))
            sources.append("online")
        else:
            batch.append(random.choice(offline_list))
            sources.append("offline")
    return batch, sources


def sample_mixed_batch_multi(
    buffers: List[Tuple[ReplayBuffer, float, str]],
    batch_size: int,
) -> tuple[List[ReplayTransition], List[str]]:
    """
    Sample from multiple replay buffers with normalized weights.

    Args:
        buffers: list of (ReplayBuffer, weight, source_name); zero-weight or empty buffers skipped.
        batch_size: batch size.

    Returns:
        (transitions, source_name per slot)
    """
    active = [(buf, float(w), name) for buf, w, name in buffers if len(buf) > 0 and float(w) > 0]
    batch_size = max(1, int(batch_size))
    if not active:
        return [], []

    total_w = sum(w for _, w, _ in active)
    lists = {name: list(buf.buf) for buf, _, name in active}
    thresholds: List[Tuple[float, str]] = []
    cum = 0.0
    for _, w, name in active:
        cum += w / total_w
        thresholds.append((cum, name))

    batch: List[ReplayTransition] = []
    sources: List[str] = []
    for _ in range(batch_size):
        r = random.random()
        chosen = thresholds[-1][1]
        for threshold, name in thresholds:
            if r <= threshold:
                chosen = name
                break
        batch.append(random.choice(lists[chosen]))
        sources.append(chosen)
    return batch, sources


def online_chunk_reward_fields(
    *,
    episode_done: bool,
    episode_success: bool,
    gamma: float = 0.99,
    chunk_size: int = 10,
) -> tuple[float, float, float]:
    """
    Align online WS rewards with offline replay semantics.

    Offline: reward/chunk_return are nonzero only on terminal chunks (episode end).
    Online (legacy): reward=1 for any step after success with done=0 — inflates Q targets.
    """
    terminal = 1.0 if episode_done else 0.0
    if episode_done and episode_success:
        reward = 1.0
        chunk_return = float(gamma ** 0)
    else:
        reward = 0.0
        chunk_return = 0.0
    return reward, chunk_return, terminal


def transition_action_ref_mse(tr: ReplayTransition) -> float:
    """Mean squared error between stored action chunk and ref_action."""
    if tr.action.numel() == 0 or tr.ref_action.numel() == 0:
        return 0.0
    return float(torch.mean((tr.action - tr.ref_action) ** 2).item())


def summarize_replay_buffer(replay: ReplayBuffer, *, label: str, max_samples: int = 5000) -> Dict[str, float]:
    """Aggregate reward/ref/action stats for offline vs online replay audit."""
    if len(replay) == 0:
        return {"label": label, "size": 0.0}

    buf = list(replay.buf)
    if len(buf) > max_samples:
        buf = random.sample(buf, max_samples)

    rewards = [float(tr.reward) for tr in buf]
    chunk_returns = [float(tr.chunk_return) for tr in buf]
    dones = [float(tr.done) for tr in buf]
    ref_mses = [transition_action_ref_mse(tr) for tr in buf]
    action_norms = [float(tr.action.norm().item()) for tr in buf if tr.action.numel() > 0]

    states = [tr.state for tr in buf if tr.state is not None and tr.state.numel() > 0]
    state_min = state_max = state_mean = float("nan")
    if states:
        stacked = torch.stack(states, dim=0).float()
        state_min = float(stacked.min().item())
        state_max = float(stacked.max().item())
        state_mean = float(stacked.mean().item())

    mid_success = sum(
        1 for tr in buf if float(tr.reward) > 0.0 and float(tr.done) < 0.5
    )

    return {
        "label": label,
        "size": float(len(replay)),
        "sampled": float(len(buf)),
        "positive_reward_rate": float(sum(1 for r in rewards if r > 0.0) / len(buf)),
        "chunk_return_mean": float(np.mean(chunk_returns)),
        "done_rate": float(np.mean(dones)),
        "mid_episode_success_reward_count": float(mid_success),
        "action_ref_mse_mean": float(np.mean(ref_mses)) if ref_mses else 0.0,
        "action_norm_mean": float(np.mean(action_norms)) if action_norms else 0.0,
        "state_min": state_min,
        "state_max": state_max,
        "state_mean": state_mean,
    }


def train_from_batch_with_diagnostics(
    agent: TD3ChunkAgent,
    batch: List[ReplayTransition],
    sources: Optional[List[str]] = None,
    apply_ref_mask: bool = True,
    provider: Optional[OnlineFeatureProvider] = None,
) -> Dict[str, float]:
    """train_from_batch plus per-source action-ref MSE when sources are provided."""
    metrics = train_from_batch(agent, batch, apply_ref_mask=apply_ref_mask, provider=provider)
    if not sources or len(sources) != len(batch):
        return metrics

    online_mses = []
    offline_mses = []
    for tr, src in zip(batch, sources):
        mse = transition_action_ref_mse(tr)
        if src == "online":
            online_mses.append(mse)
        else:
            offline_mses.append(mse)
    if online_mses:
        metrics["online_action_ref_mse"] = float(np.mean(online_mses))
    if offline_mses:
        metrics["offline_action_ref_mse"] = float(np.mean(offline_mses))
    metrics["batch_online_frac"] = float(sum(1 for s in sources if s == "online") / len(sources))
    return metrics


def replay_positive_reward_stats(replay: ReplayBuffer) -> Dict[str, float]:
    """Fraction of transitions with reward > 0 (for logging)."""
    if len(replay) == 0:
        return {"size": 0.0, "positive_reward_rate": 0.0}
    positive = sum(1 for tr in replay.buf if float(tr.reward) > 0.0)
    size = len(replay)
    return {"size": float(size), "positive_reward_rate": float(positive / size)}


def collect_task_episode_ids(dataset: LeRobotDataset, task_index: int) -> List[int]:
    print(f"[dataset] scanning episodes for task_index={task_index}...")
    episode_ids = set()
    for idx in tqdm(range(len(dataset)), desc="scan_task_episodes", leave=False):
        sample = dataset[idx]
        if sample_task_index(sample) != task_index:
            continue
        episode_ids.add(sample_episode_index(sample))
    if not episode_ids:
        raise RuntimeError(f"task_index={task_index} has no episodes")
    return sorted(episode_ids)


def build_episode_index(dataset: LeRobotDataset) -> Dict[int, List[int]]:
    episode_to_indices: Dict[int, List[int]] = defaultdict(list)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        episode_to_indices[sample_episode_index(sample)].append(idx)
    for episode_id in episode_to_indices:
        episode_to_indices[episode_id].sort()
    return dict(episode_to_indices)


def _proprio_from_sample(sample: Dict, state_key: str, fallback: Optional[torch.Tensor] = None) -> torch.Tensor:
    if state_key in sample:
        state = sample[state_key]
    elif fallback is not None:
        state = fallback
    else:
        raise KeyError(f"sample missing state key: {state_key}")
    if isinstance(state, torch.Tensor):
        if state.dim() > 1:
            state = state.reshape(-1)
        return state.to(dtype=torch.float32).detach().cpu()
    return torch.as_tensor(state, dtype=torch.float32)


@dataclass
class WSRolloutFrame:
    """One env step recorded during VLA WebSocket rollout."""

    local_index: int
    obs_msg: dict
    action: np.ndarray
    ref_chunk: np.ndarray


@dataclass
class _ReplayBuildRow:
    sample_index: int
    next_sample_index: int
    episode_id: int
    state: torch.Tensor
    next_state: torch.Tensor
    action: torch.Tensor
    ref_action: torch.Tensor
    next_ref_action: torch.Tensor
    reward: float
    chunk_return: float
    done: float


def _collect_replay_rows(
    dataset: LeRobotDataset,
    episode_to_indices: Dict[int, List[int]],
    state_key: str,
    chunk_len: int,
    stride: int,
    gamma: float,
) -> List[_ReplayBuildRow]:
    rows: List[_ReplayBuildRow] = []
    for episode_id, indices in episode_to_indices.items():
        if len(indices) < chunk_len + 1:
            continue
        for local_start in range(0, len(indices) - chunk_len, stride):
            current_idx = indices[local_start]
            next_local = local_start + chunk_len
            next_idx = indices[next_local]
            sample = dataset[current_idx]
            next_sample = dataset[next_idx]
            actions = sample["action"]
            if actions.shape[0] < chunk_len:
                continue
            action_chunk = actions[:chunk_len].to(dtype=torch.float32).detach().cpu()
            next_actions = next_sample["action"]
            if next_actions.shape[0] < chunk_len:
                continue
            next_action_chunk = next_actions[:chunk_len].to(dtype=torch.float32).detach().cpu()
            state = _proprio_from_sample(sample, state_key)
            next_state = _proprio_from_sample(
                next_sample,
                state_key,
                fallback=state,
            )
            terminal = 1.0 if next_local >= len(indices) - 1 else 0.0
            reward = 1.0 if terminal > 0.5 else 0.0
            episode_last_local = len(indices) - 1
            chunk_end_local = min(local_start + chunk_len - 1, episode_last_local)
            if local_start <= episode_last_local <= chunk_end_local:
                reward_step_offset = episode_last_local - local_start
                chunk_return = float((gamma ** reward_step_offset) * 1.0)
            else:
                chunk_return = 0.0
            rows.append(
                _ReplayBuildRow(
                    sample_index=current_idx,
                    next_sample_index=next_idx,
                    episode_id=episode_id,
                    state=state,
                    next_state=next_state,
                    action=action_chunk,
                    ref_action=action_chunk.clone(),
                    next_ref_action=next_action_chunk,
                    reward=reward,
                    chunk_return=chunk_return,
                    done=terminal,
                )
            )
    return rows


def _state_from_ws_obs(obs_msg: dict, state_dim: int) -> torch.Tensor:
    if obs_msg.get("state"):
        return torch.as_tensor(
            np.asarray(obs_msg["state"], dtype=np.float32).reshape(-1),
            dtype=torch.float32,
        )
    return torch.zeros((state_dim,), dtype=torch.float32)


def collect_replay_rows_from_ws_episode(
    frames: List[WSRolloutFrame],
    episode_id: int,
    chunk_len: int,
    stride: int,
    gamma: float,
    state_dim: int,
    global_index_base: int,
) -> List[_ReplayBuildRow]:
    """Build replay rows from a successful WS episode (mirrors _collect_replay_rows)."""
    if len(frames) < chunk_len + 1:
        return []

    rows: List[_ReplayBuildRow] = []
    num_frames = len(frames)
    for local_start in range(0, num_frames - chunk_len, stride):
        next_local = local_start + chunk_len
        start_frame = frames[local_start]
        next_frame = frames[next_local]

        action_parts = []
        for j in range(chunk_len):
            action_parts.append(
                torch.as_tensor(frames[local_start + j].action, dtype=torch.float32).reshape(-1)
            )
        action_chunk = torch.stack(action_parts, dim=0)

        next_action_parts = []
        if next_local + chunk_len <= num_frames:
            for j in range(chunk_len):
                next_action_parts.append(
                    torch.as_tensor(frames[next_local + j].action, dtype=torch.float32).reshape(-1)
                )
            next_action_chunk = torch.stack(next_action_parts, dim=0)
        else:
            next_action_chunk = torch.as_tensor(start_frame.ref_chunk, dtype=torch.float32).clone()
            if next_action_chunk.dim() == 1:
                next_action_chunk = next_action_chunk.unsqueeze(0).repeat(chunk_len, 1)

        ref_chunk = torch.as_tensor(start_frame.ref_chunk, dtype=torch.float32)
        if ref_chunk.dim() == 1:
            ref_chunk = ref_chunk.unsqueeze(0)
        if ref_chunk.shape[0] < chunk_len:
            pad = ref_chunk[-1:].repeat(chunk_len - ref_chunk.shape[0], 1)
            ref_chunk = torch.cat([ref_chunk, pad], dim=0)
        ref_chunk = ref_chunk[:chunk_len].clone()

        next_ref = torch.as_tensor(next_frame.ref_chunk, dtype=torch.float32)
        if next_ref.dim() == 1:
            next_ref = next_ref.unsqueeze(0)
        if next_ref.shape[0] < chunk_len:
            pad = next_ref[-1:].repeat(chunk_len - next_ref.shape[0], 1)
            next_ref = torch.cat([next_ref, pad], dim=0)
        next_ref = next_ref[:chunk_len].clone()

        state = _state_from_ws_obs(start_frame.obs_msg, state_dim)
        next_state = _state_from_ws_obs(next_frame.obs_msg, state_dim)

        terminal = 1.0 if next_local >= num_frames - 1 else 0.0
        reward = 1.0 if terminal > 0.5 else 0.0
        episode_last_local = num_frames - 1
        chunk_end_local = min(local_start + chunk_len - 1, episode_last_local)
        if local_start <= episode_last_local <= chunk_end_local:
            reward_step_offset = episode_last_local - local_start
            chunk_return = float((gamma ** reward_step_offset) * 1.0)
        else:
            chunk_return = 0.0

        sample_index = global_index_base + local_start
        next_sample_index = global_index_base + next_local
        rows.append(
            _ReplayBuildRow(
                sample_index=sample_index,
                next_sample_index=next_sample_index,
                episode_id=episode_id,
                state=state,
                next_state=next_state,
                action=action_chunk,
                ref_action=ref_chunk,
                next_ref_action=next_action_chunk,
                reward=reward,
                chunk_return=chunk_return,
                done=terminal,
            )
        )
    return rows


@torch.no_grad()
def precompute_ws_frame_features(
    frames: List[WSRolloutFrame],
    frame_indices: List[int],
    vla_model,
    rl_encoder,
    image_keys: List[str],
    image_size: int,
    device: torch.device,
    instruction: str,
    ws_infer=None,
) -> Dict[int, torch.Tensor]:
    unique = sorted(set(frame_indices))
    z_by_index: Dict[int, torch.Tensor] = {}
    index_to_frame = {f.local_index: f for f in frames}
    for idx in tqdm(unique, desc="precompute_ws_z_rl", leave=False):
        frame = index_to_frame[idx]
        if ws_infer is not None:
            from .libero_ws_vla_collect_core import ws_obs_to_normalized_vla_inputs

            inputs = ws_obs_to_normalized_vla_inputs(
                frame.obs_msg,
                image_keys=image_keys,
                image_size=image_size,
                device=device,
                instruction=instruction,
                ws_infer=ws_infer,
            )
        else:
            from .libero_ws_td3_eval_core import ws_obs_to_vla_model_inputs

            inputs, _ = ws_obs_to_vla_model_inputs(
                frame.obs_msg,
                image_keys=image_keys,
                image_size=image_size,
                device=device,
                instruction=instruction,
            )
        z_tokens = vla_model.extract_vla_tokens(inputs)
        z_rl = rl_encoder.encode(z_tokens).float().squeeze(0).detach().cpu()
        z_by_index[idx] = z_rl
    return z_by_index


def append_ws_episode_to_replay(
    replay: ReplayBuffer,
    frames: List[WSRolloutFrame],
    *,
    episode_id: int,
    chunk_len: int,
    stride: int,
    gamma: float,
    state_dim: int,
    global_index_base: int,
    vla_model,
    rl_encoder,
    image_keys: List[str],
    image_size: int,
    device: torch.device,
    instruction: str,
    ws_infer=None,
) -> int:
    """Convert one successful WS episode to ReplayTransitions and append to buffer."""
    rows = collect_replay_rows_from_ws_episode(
        frames,
        episode_id,
        chunk_len,
        stride,
        gamma,
        state_dim,
        global_index_base,
    )
    if not rows:
        return 0

    frame_indices: List[int] = []
    for row in rows:
        frame_indices.append(row.sample_index - global_index_base)
        frame_indices.append(row.next_sample_index - global_index_base)
    z_by_local = precompute_ws_frame_features(
        frames,
        frame_indices,
        vla_model,
        rl_encoder,
        image_keys,
        image_size,
        device,
        instruction,
        ws_infer=ws_infer,
    )

    for row in rows:
        local_curr = row.sample_index - global_index_base
        local_next = row.next_sample_index - global_index_base
        replay.add(
            ReplayTransition(
                sample_index=row.sample_index,
                next_sample_index=row.next_sample_index,
                episode_id=row.episode_id,
                z_rl=z_by_local[local_curr],
                state=row.state,
                action=row.action,
                ref_action=row.ref_action,
                reward=row.reward,
                chunk_return=row.chunk_return,
                done=row.done,
                next_z_rl=z_by_local[local_next],
                next_state=row.next_state,
                next_ref_action=row.next_ref_action,
            )
        )
    return len(rows)


@torch.no_grad()
def precompute_frame_features(
    dataset: LeRobotDataset,
    sample_indices: List[int],
    image_keys: List[str],
    state_key: str,
    vla_model,
    rl_encoder,
    device: torch.device,
    batch_size: int = 8,
) -> Dict[int, torch.Tensor]:
    unique = sorted(set(sample_indices))
    z_by_index: Dict[int, torch.Tensor] = {}
    for start in tqdm(range(0, len(unique), batch_size), desc="precompute_z_rl", leave=False):
        batch_indices = unique[start : start + batch_size]
        for idx in batch_indices:
            sample = dataset[idx]
            inputs = build_single_input(sample, image_keys, state_key, device)
            z_tokens = vla_model.extract_vla_tokens(inputs)
            z_rl = rl_encoder.encode(z_tokens).float().squeeze(0).detach().cpu()
            z_by_index[idx] = z_rl
    return z_by_index


def resolve_rl_token_dim_for_meta(
    rl_token_checkpoint: str | Path,
    raw_config: Dict[str, Any],
    rl_token_network_cfg: Optional[Dict[str, Any]] = None,
    default: int = 256,
) -> int:
    """Resolve RL token dim for replay cache meta (config -> RL checkpoint -> default)."""
    token_block = raw_config.get("train_rl_token") or {}
    network_cfg = rl_token_network_cfg if rl_token_network_cfg is not None else (token_block.get("network") or {})
    dim = network_cfg.get("rl_token_dim")
    if dim is not None:
        return int(dim)

    model_rl = (raw_config.get("model") or {}).get("rl_token") or {}
    dim = model_rl.get("rl_token_dim")
    if dim is not None:
        return int(dim)

    rl_path = Path(rl_token_checkpoint).expanduser().resolve()
    if rl_path.is_file():
        try:
            ckpt = torch.load(rl_path, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(rl_path, map_location="cpu")
        ckpt_dim = ckpt.get("rl_token_dim") or ckpt.get("model_dim")
        if ckpt_dim is not None:
            return int(ckpt_dim)
    return int(default)


def build_replay_cache_meta(
    *,
    chunk_len: int,
    stride: int,
    gamma: float,
    state_dim: int,
    rl_token_dim: int,
    dataset_path: str,
    episodes: Optional[List[int]],
    vla_checkpoint: str,
    rl_token_checkpoint: str,
    config_path: str,
    task_index: Optional[int],
) -> Dict[str, Any]:
    ep_payload = json.dumps(sorted(episodes) if episodes is not None else "all", sort_keys=True)
    return {
        "version": REPLAY_CACHE_VERSION,
        "chunk_len": int(chunk_len),
        "stride": int(stride),
        "gamma": float(gamma),
        "state_dim": int(state_dim),
        "rl_token_dim": int(rl_token_dim),
        "dataset_path": str(Path(dataset_path).resolve()),
        "episodes_hash": hashlib.sha256(ep_payload.encode()).hexdigest()[:16],
        "vla_checkpoint": str(Path(vla_checkpoint).resolve()),
        "rl_token_checkpoint": str(Path(rl_token_checkpoint).resolve()),
        "config_path": str(Path(config_path).resolve()),
        "task_index": task_index,
    }


def build_vla_success_replay_meta(
    *,
    chunk_len: int,
    stride: int,
    gamma: float,
    state_dim: int,
    rl_token_dim: int,
    dataset_path: str,
    vla_checkpoint: str,
    rl_token_checkpoint: str,
    config_path: str,
    task_index: Optional[int],
    num_success_episodes: int,
    total_attempts: int,
) -> Dict[str, Any]:
    base = build_replay_cache_meta(
        chunk_len=chunk_len,
        stride=stride,
        gamma=gamma,
        state_dim=state_dim,
        rl_token_dim=rl_token_dim,
        dataset_path=dataset_path,
        episodes=None,
        vla_checkpoint=vla_checkpoint,
        rl_token_checkpoint=rl_token_checkpoint,
        config_path=config_path,
        task_index=task_index,
    )
    base["source"] = "vla_ws_success"
    base["num_success_episodes"] = int(num_success_episodes)
    base["total_attempts"] = int(total_attempts)
    base["success_rate"] = (
        float(num_success_episodes) / float(total_attempts) if total_attempts > 0 else 0.0
    )
    base["episodes_hash"] = "vla_ws_success"
    return base


def vla_success_cache_compatible(meta: Dict[str, Any], expected: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate VLA success cache against expected TD3 replay hyperparams."""
    keys = (
        "chunk_len",
        "stride",
        "gamma",
        "state_dim",
        "rl_token_dim",
        "vla_checkpoint",
        "rl_token_checkpoint",
    )
    for key in keys:
        if expected.get(key) != meta.get(key):
            return False, f"mismatch on {key}: expected={expected.get(key)!r}, cached={meta.get(key)!r}"
    if meta.get("source") != "vla_ws_success":
        return False, f"not a vla_ws_success cache (source={meta.get('source')!r})"
    return True, "ok"


def replay_cache_matches(expected: Dict[str, Any], cached: Dict[str, Any]) -> Tuple[bool, str]:
    keys = (
        "version",
        "chunk_len",
        "stride",
        "gamma",
        "state_dim",
        "rl_token_dim",
        "dataset_path",
        "episodes_hash",
        "vla_checkpoint",
        "rl_token_checkpoint",
        "config_path",
        "task_index",
    )
    for key in keys:
        if expected.get(key) != cached.get(key):
            return False, f"mismatch on {key}: expected={expected.get(key)!r}, cached={cached.get(key)!r}"
    return True, "ok"


def save_replay_buffer(path: Path, replay: ReplayBuffer, meta: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"meta": meta, "transitions": list(replay.buf)}, path)
    print(f"[replay] saved cache: {path} ({len(replay)} transitions)")


def load_replay_buffer(path: Path, capacity: int) -> Tuple[ReplayBuffer, Dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    meta = payload["meta"]
    replay = ReplayBuffer(capacity=capacity)
    for tr in payload["transitions"]:
        if isinstance(tr, dict):
            tr = ReplayTransition(**tr)
        replay.add(tr)
    print(f"[replay] loaded cache: {path} ({len(replay)} transitions)")
    return replay, meta


def build_replay_from_dataset(
    dataset: LeRobotDataset,
    episode_to_indices: Dict[int, List[int]],
    state_key: str,
    chunk_len: int,
    stride: int,
    gamma: float,
    device: torch.device,
    capacity: int,
    vla_model,
    rl_encoder,
    image_keys: List[str],
    feature_batch_size: int = 8,
) -> ReplayBuffer:
    print("[replay] building ReplayBuffer from dataset...")
    rows = _collect_replay_rows(
        dataset,
        episode_to_indices,
        state_key,
        chunk_len,
        stride,
        gamma,
    )
    if not rows:
        raise RuntimeError("ReplayBuffer is empty: check task, chunk_len, stride, or episode length.")

    frame_indices: List[int] = []
    for row in rows:
        frame_indices.append(row.sample_index)
        frame_indices.append(row.next_sample_index)
    z_by_index = precompute_frame_features(
        dataset,
        frame_indices,
        image_keys,
        state_key,
        vla_model,
        rl_encoder,
        device,
        batch_size=feature_batch_size,
    )

    replay = ReplayBuffer(capacity=capacity)
    for row in tqdm(rows, desc="assemble_replay", leave=False):
        replay.add(
            ReplayTransition(
                sample_index=row.sample_index,
                next_sample_index=row.next_sample_index,
                episode_id=row.episode_id,
                z_rl=z_by_index[row.sample_index],
                state=row.state,
                action=row.action,
                ref_action=row.ref_action,
                reward=row.reward,
                chunk_return=row.chunk_return,
                done=row.done,
                next_z_rl=z_by_index[row.next_sample_index],
                next_state=row.next_state,
                next_ref_action=row.next_ref_action,
            )
        )
    return replay


class OnlineFeatureProvider:
    """Cache z_rl and VLA reference chunks per frame index."""

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
        if sample_index in self.cache:
            z_rl, ref_chunk, state = self.cache[sample_index]
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
        ref_chunk = torch.as_tensor(
            pred[:, : self.chunk_len, :], dtype=torch.float32, device=self.device
        )
        state = sample[self.state_key] if self.state_key in sample else torch.zeros((1, 0), device=self.device)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        state = state.to(device=self.device, dtype=torch.float32)
        self.cache[sample_index] = (z_rl.detach().cpu(), ref_chunk.detach().cpu(), state.detach().cpu())
        return z_rl, ref_chunk, state


def _transition_has_precomputed_features(tr: ReplayTransition) -> bool:
    return tr.z_rl is not None and tr.z_rl.numel() > 0


def apply_online_td3_lr_scale(agent: TD3ChunkAgent, actor_lr_scale: float, critic_lr_scale: float) -> None:
    """Scale TD3 optimizer LRs in-place (e.g. 0.1 for conservative online fine-tuning)."""
    for group in agent.actor_opt.param_groups:
        group["lr"] = float(agent.cfg.actor_lr) * float(actor_lr_scale)
    for group in agent.critic_opt.param_groups:
        group["lr"] = float(agent.cfg.critic_lr) * float(critic_lr_scale)


def train_from_batch(
    agent: TD3ChunkAgent,
    batch: List[ReplayTransition],
    apply_ref_mask: bool = True,
    provider: Optional[OnlineFeatureProvider] = None,
) -> Dict[str, float]:
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
        if _transition_has_precomputed_features(tr):
            z_rl.append(tr.z_rl)
            state.append(tr.state)
            ref.append(tr.ref_action)
            next_z_rl.append(tr.next_z_rl)
            next_state.append(tr.next_state)
            next_ref.append(tr.next_ref_action)
        else:
            if provider is None:
                raise RuntimeError("transition missing z_rl and no OnlineFeatureProvider given")
            curr_z, curr_ref, curr_state = provider.get(tr.sample_index)
            nxt_z, nxt_ref, nxt_state = provider.get(tr.next_sample_index)
            z_rl.append(curr_z.squeeze(0))
            state.append(curr_state.squeeze(0))
            ref.append(curr_ref.squeeze(0))
            next_z_rl.append(nxt_z.squeeze(0))
            next_state.append(nxt_state.squeeze(0))
            next_ref.append(nxt_ref.squeeze(0))
        action.append(tr.action)
        reward.append([tr.reward])
        chunk_return.append([tr.chunk_return])
        done.append([tr.done])

    return agent.train_step(
        z_rl=torch.stack(z_rl, dim=0).to(agent.device),
        state=torch.stack(state, dim=0).to(agent.device),
        ref_actions=torch.stack(ref, dim=0).to(agent.device),
        action=torch.stack(action, dim=0).to(agent.device),
        reward=torch.tensor(reward, dtype=torch.float32, device=agent.device),
        chunk_return=torch.tensor(chunk_return, dtype=torch.float32, device=agent.device),
        next_z_rl=torch.stack(next_z_rl, dim=0).to(agent.device),
        next_state=torch.stack(next_state, dim=0).to(agent.device),
        next_ref_actions=torch.stack(next_ref, dim=0).to(agent.device),
        done=torch.tensor(done, dtype=torch.float32, device=agent.device),
        apply_ref_mask=apply_ref_mask,
    )


def warmup_train(
    agent: TD3ChunkAgent,
    replay: ReplayBuffer,
    batch_size: int,
    warmup_updates: int,
    warmup_update_ratio: int,
    provider: Optional[OnlineFeatureProvider] = None,
) -> List[Dict[str, float]]:
    print(
        f"[warmup] starting warmup: {warmup_updates} outer steps, "
        f"G={warmup_update_ratio} inner gradient steps each..."
    )
    logs: List[Dict[str, float]] = []
    for step in tqdm(range(warmup_updates), desc="warmup_updates", leave=False):
        for g in range(warmup_update_ratio):
            batch = replay.sample(batch_size)
            metrics = train_from_batch(agent, batch, apply_ref_mask=True, provider=provider)
            metrics["phase"] = "warmup"
            metrics["step"] = float(step + 1)
            metrics["warmup_inner_g"] = float(g + 1)
            logs.append(metrics)
    return logs


def online_train_loop(
    agent: TD3ChunkAgent,
    replay: ReplayBuffer,
    batch_size: int,
    online_train_episodes: int,
    provider: Optional[OnlineFeatureProvider] = None,
) -> List[Dict[str, float]]:
    print(f"[online] starting online training for {online_train_episodes} episodes...")
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
            metrics = train_from_batch(agent, batch, apply_ref_mask=True, provider=provider)
            metrics["phase"] = "online"
            metrics["episode"] = float(ep + 1)
            logs.append(metrics)
        if (ep + 1) % max(1, online_train_episodes // 10) == 0:
            print(f"[online] finished episode {ep + 1}/{online_train_episodes}")
    return logs


@torch.no_grad()
def eval_with_vla_reference(
    agent: TD3ChunkAgent,
    replay: ReplayBuffer,
    provider: Optional[OnlineFeatureProvider] = None,
    max_eval_samples: int = 256,
) -> Dict[str, float]:
    print(f"[eval] evaluating up to {max_eval_samples} samples...")
    selected = list(replay.buf)[:max_eval_samples]
    mse_vals = []
    mae_vals = []
    l2_vals = []
    ref_mse_vals = []
    q_vals = []

    for tr in tqdm(selected, desc="eval_samples", leave=False):
        if _transition_has_precomputed_features(tr):
            z_rl = tr.z_rl.unsqueeze(0).to(agent.device)
            state = tr.state.unsqueeze(0).to(agent.device)
        else:
            if provider is None:
                raise RuntimeError("transition missing z_rl and no OnlineFeatureProvider given")
            z_rl, _, state = provider.get(tr.sample_index)
        if provider is not None:
            _, ref_chunk, _ = provider.get(tr.sample_index)
        else:
            ref_chunk = tr.ref_action.unsqueeze(0).to(agent.device)
        gt = tr.action.unsqueeze(0).to(agent.device)
        pred = agent.act(z_rl, state, ref_chunk, deterministic=True, apply_ref_mask=False)
        q1, q2 = agent.critic(z_rl, state, pred)
        q_vals.append(torch.min(q1, q2).mean().item())
        mse_vals.append(F.mse_loss(pred, gt).item())
        mae_vals.append(F.l1_loss(pred, gt).item())
        l2_vals.append((pred - gt).pow(2).sum(dim=-1).sqrt().mean().item())
        ref_mse_vals.append(F.mse_loss(ref_chunk, gt).item())

    if not mse_vals:
        raise RuntimeError("no samples available for evaluation")
    return {
        "mse": float(np.mean(mse_vals)),
        "mae": float(np.mean(mae_vals)),
        "mean_l2_per_step": float(np.mean(l2_vals)),
        "ref_mse": float(np.mean(ref_mse_vals)),
        "eval_q_mean": float(np.mean(q_vals)),
    }


def _plot_curve(
    output_dir: Path,
    prefix: str,
    y_values: List[float],
    ylabel: str,
    title: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{prefix}.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "value"])
        for i, val in enumerate(y_values):
            writer.writerow([i, val if val == val else ""])  # NaN -> empty

    try:
        import matplotlib.pyplot as plt

        x = np.arange(len(y_values))
        y = np.array(y_values, dtype=np.float64)
        mask = np.isfinite(y)
        plt.figure(figsize=(9, 5))
        if mask.any():
            plt.plot(x[mask], y[mask], label=ylabel)
        plt.title(title)
        plt.xlabel("Update Step")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{prefix}.png", dpi=150)
        plt.close()
    except Exception as exc:
        print(f"[warn] failed to write {prefix}.png: {exc}")


def export_training_curves(logs: List[Dict[str, float]], output_dir: Path) -> None:
    """
    Export three loss curves: Q loss, actor BC, policy total loss (eq. 5).
    """
    critic_losses = [float(row.get("critic_loss", float("nan"))) for row in logs]
    actor_bc: List[float] = []
    policy_total: List[float] = []
    for row in logs:
        al = float(row.get("actor_loss", 0.0))
        acl = float(row.get("actor_constraint_loss", 0.0))
        if al == 0.0 and acl == 0.0:
            actor_bc.append(float("nan"))
            policy_total.append(float("nan"))
        else:
            actor_bc.append(acl)
            policy_total.append(al)

    _plot_curve(output_dir, "q_loss_curve", critic_losses, "critic_loss", "Critic TD Loss (L_Q)")
    _plot_curve(
        output_dir,
        "actor_bc_curve",
        actor_bc,
        "actor_constraint_loss",
        "Actor Behavior Clone (beta * ||a - a_ref||^2)",
    )
    _plot_curve(
        output_dir,
        "policy_total_loss_curve",
        policy_total,
        "actor_loss",
        "Policy Total Loss (L_pi, eq. 5)",
    )
    print(f"[curves] saved training curves under {output_dir}")


def export_q_logs(logs: List[Dict[str, float]], output_dir: Path, base_name: str = "q_curve") -> None:
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
        print(f"[warn] failed to write {base_name}.png: {exc}")


def save_td3_agent(
    path: Path,
    agent: TD3ChunkAgent,
    *,
    rl_token_dim: int,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    td3_cfg: TD3ChunkConfig,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "rl_token_dim": rl_token_dim,
        "state_dim": state_dim,
        "action_dim": action_dim,
        "chunk_size": chunk_size,
        "td3_cfg": {
            "gamma": td3_cfg.gamma,
            "tau": td3_cfg.tau,
            "actor_lr": td3_cfg.actor_lr,
            "critic_lr": td3_cfg.critic_lr,
            "policy_noise": td3_cfg.policy_noise,
            "noise_clip": td3_cfg.noise_clip,
            "policy_delay": td3_cfg.policy_delay,
            "fixed_std": td3_cfg.fixed_std,
            "max_action": td3_cfg.max_action,
            "ref_mask_prob": td3_cfg.ref_mask_prob,
            "policy_constraint_beta": td3_cfg.policy_constraint_beta,
            "use_chunk_return_target": td3_cfg.use_chunk_return_target,
            "hidden_dims": td3_cfg.hidden_dims,
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
    print(f"saved TD3 checkpoint: {path}")


def save_td3_agent_from_args(path: Path, agent: TD3ChunkAgent, args, cfg) -> None:
    """Backward-compatible saver for test_online_td3_sim CLI."""
    rl_token_dim = int(agent.actor.rl_token_dim)
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
    save_td3_agent(
        path,
        agent,
        rl_token_dim=int(rl_token_dim),
        state_dim=int(cfg.state_dim),
        action_dim=int(cfg.action_dim),
        chunk_size=int(args.chunk_len),
        td3_cfg=td3_cfg,
    )


def load_td3_agent(path: Path, device: torch.device) -> TD3ChunkAgent:
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
    print(f"loaded TD3 checkpoint: {path}, total_updates={agent.total_updates}")
    return agent


def save_online_step_buffer(
    buffer_dir: Path,
    step: int,
    transition: ReplayTransition,
) -> Path:
    """
    Save one online transition snapshot for debugging/auditing.
    """
    buffer_dir = Path(buffer_dir)
    buffer_dir.mkdir(parents=True, exist_ok=True)
    out_path = buffer_dir / f"step_buffer_{int(step):08d}.pt"
    torch.save(
        {
            "step": int(step),
            "transition": transition,
        },
        out_path,
    )
    return out_path


def prune_old_buffer_files(
    buffer_dir: Path,
    max_files: int,
) -> List[Path]:
    """
    Keep at most `max_files` step buffer files, deleting oldest by filename order.
    """
    if max_files <= 0:
        return []
    buffer_dir = Path(buffer_dir)
    files = sorted(buffer_dir.glob("step_buffer_*.pt"))
    if len(files) <= max_files:
        return []
    removed: List[Path] = []
    for p in files[: len(files) - max_files]:
        p.unlink(missing_ok=True)
        removed.append(p)
    return removed
