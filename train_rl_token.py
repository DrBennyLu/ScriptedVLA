"""
Train RL token bottleneck with frozen VLA token embeddings.

在 libero/config_libero_object.yaml 中编辑顶层 ``train_rl_token`` 后，在仓库根目录执行::

    python train_rl_token.py

author: Benny Lu
license: MIT
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from inference import find_latest_checkpoint
from libero.libero_dataset_replay import get_task_description, resolve_training_episodes
from src.ScriptedVLA.model import RLTokenBottleneck
from src.ScriptedVLA.utils import ScriptConfig, ensure_offline_mode_if_needed, load_script_config
from test.test_inference import (
    get_test_model_config,
    load_model_from_checkpoint_with_lora_support,
    validate_checkpoint,
)

DEFAULT_CONFIG_PATH = "libero/config_libero_object.yaml"

ensure_offline_mode_if_needed(DEFAULT_CONFIG_PATH)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    HAS_LEROBOT = True

    class LeRobotDatasetSubset(LeRobotDataset):
        """修复 episodes=subset 时 episode_data_index 与原始 episode_index 不一致的问题。"""

        def _get_query_indices(self, idx: int, ep_idx: int):
            if self.episodes is not None and ep_idx in self.episodes:
                ep_idx = self.episodes.index(ep_idx)
            return super()._get_query_indices(idx, ep_idx)

except ImportError:
    HAS_LEROBOT = False
    LeRobotDataset = None
    LeRobotDatasetSubset = None


@dataclass
class TrainRLTokenDatasetSettings:
    local_path: str
    task_index: Optional[int]
    episode_slice: Optional[List[int]]


@dataclass
class TrainRLTokenTrainingSettings:
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_steps: int
    save_steps: int
    logging_steps: int
    num_workers: int


@dataclass
class TrainRLTokenNetworkDims:
    input_dim: int
    model_dim: int
    rl_token_dim: int
    num_encoder_layers: int
    num_decoder_layers: int
    num_heads: int
    ffn_dim: int
    dropout: float


@dataclass
class TrainRLTokenSettings:
    vla_checkpoint: Path
    dataset: TrainRLTokenDatasetSettings
    save_dir: Path
    device: str
    seed: int
    validate_vla_checkpoint: bool
    align_dims_to_vla: bool
    network_cfg: Dict[str, Any]
    training: TrainRLTokenTrainingSettings
    resolved_episodes: Optional[List[int]]
    task_description: Optional[str]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def create_delta_timestamps(action_horizon: int, fps: int) -> Dict[str, List[float]]:
    return {"action": [t / fps for t in range(action_horizon)]}


def load_dataset_info(dataset_path: Path) -> dict:
    info_file = dataset_path / "meta" / "info.json"
    if not info_file.exists():
        raise FileNotFoundError(f"missing info.json at {info_file}")
    with open(info_file, "r", encoding="utf-8") as f:
        return json.load(f)


def _tensor_to_pil_image(img_tensor, image_size=None):
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
    img = Image.fromarray(img_array, mode="RGB")
    if image_size is not None and img.size != (image_size, image_size):
        img = img.resize((image_size, image_size), Image.Resampling.LANCZOS)
    return img


def create_collate_fn(image_keys, state_key, image_size=None, use_batch_task=True):
    def collate_fn(batch_list):
        from torch.utils.data._utils.collate import default_collate

        batch_dict = default_collate(batch_list)
        batch_size = len(batch_list)
        images_list = []
        for i in range(batch_size):
            if len(image_keys) == 1:
                images_list.append(_tensor_to_pil_image(batch_dict[image_keys[0]][i], image_size))
            else:
                images_list.append([_tensor_to_pil_image(batch_dict[k][i], image_size) for k in image_keys])

        texts = [""] * batch_size
        if use_batch_task and "task" in batch_dict:
            td = batch_dict["task"]
            texts = [str(t) for t in td] if isinstance(td, list) else [str(td)] * batch_size

        result = {"images": images_list, "text": texts}
        if state_key in batch_dict:
            result["state"] = batch_dict[state_key]
        return result

    return collate_fn


def resolve_vla_checkpoint_path(vla_checkpoint: Optional[str], vla_checkpoint_dir: Optional[str]) -> Path:
    explicit = vla_checkpoint or vla_checkpoint_dir
    if not explicit:
        raise ValueError("train_rl_token.vla_checkpoint or vla_checkpoint_dir must be set in config")
    p = Path(explicit).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"checkpoint path not found: {p}")
    if p.is_dir():
        latest = find_latest_checkpoint(p)
        if latest is None:
            raise FileNotFoundError(f"no checkpoint_step_*.pt under {p}")
        return latest
    return p


def resolve_rl_token_network_dims(
    network_cfg: Dict[str, Any],
    vla_hidden_dim: int,
    align_dims_to_vla: bool,
) -> TrainRLTokenNetworkDims:
    num_heads = int(network_cfg.get("num_heads", 8))
    dropout = float(network_cfg.get("dropout", 0.1))
    num_encoder_layers = int(network_cfg.get("num_encoder_layers", 2))
    num_decoder_layers = int(network_cfg.get("num_decoder_layers", 2))

    cfg_model_dim = network_cfg.get("model_dim")
    cfg_rl_token_dim = network_cfg.get("rl_token_dim")
    cfg_ffn_dim = network_cfg.get("ffn_dim")

    if align_dims_to_vla:
        if cfg_model_dim is not None and int(cfg_model_dim) != vla_hidden_dim:
            raise ValueError(
                f"align_dims_to_vla=true but network.model_dim={cfg_model_dim} != vla_hidden_dim={vla_hidden_dim}"
            )
        if cfg_rl_token_dim is not None and int(cfg_rl_token_dim) != vla_hidden_dim:
            raise ValueError(
                f"align_dims_to_vla=true but network.rl_token_dim={cfg_rl_token_dim} != vla_hidden_dim={vla_hidden_dim}"
            )
        model_dim = vla_hidden_dim
        rl_token_dim = vla_hidden_dim
    else:
        model_dim = int(cfg_model_dim or vla_hidden_dim)
        rl_token_dim = int(cfg_rl_token_dim or model_dim)
        if rl_token_dim != vla_hidden_dim:
            raise ValueError(
                f"rl_token_dim ({rl_token_dim}) must equal vla hidden_dim ({vla_hidden_dim})"
            )

    if model_dim % num_heads != 0:
        raise ValueError(f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})")

    ffn_dim = int(cfg_ffn_dim or (4 * model_dim))

    return TrainRLTokenNetworkDims(
        input_dim=vla_hidden_dim,
        model_dim=model_dim,
        rl_token_dim=rl_token_dim,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_heads=num_heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
    )


def load_train_rl_token_settings(raw: Dict[str, Any], cfg: ScriptConfig) -> TrainRLTokenSettings:
    if "train_rl_token" not in raw:
        raise KeyError(
            f"missing top-level 'train_rl_token' in {cfg.config_path}; "
            "add train_rl_token section (see libero/config_libero_object.yaml)"
        )
    rl_cfg = raw["train_rl_token"]

    ds_cfg = rl_cfg.get("dataset") or {}
    local_path = ds_cfg.get("local_path") or cfg.dataset_path
    task_index = ds_cfg.get("task_index")
    episode_slice = ds_cfg.get("episode_slice")

    dataset_path = Path(local_path).resolve()
    resolved = resolve_training_episodes(str(dataset_path), task_index, episode_slice)
    task_description = None
    if task_index is not None and (episode_slice is None or len(episode_slice) == 0):
        task_description = get_task_description(str(dataset_path), int(task_index))

    train_block = rl_cfg.get("training") or {}
    training = TrainRLTokenTrainingSettings(
        batch_size=int(train_block.get("batch_size", 8)),
        learning_rate=float(train_block.get("learning_rate", 1e-4)),
        weight_decay=float(train_block.get("weight_decay", 0.0)),
        max_steps=int(train_block.get("max_steps", 10000)),
        save_steps=int(train_block.get("save_steps", 2000)),
        logging_steps=int(train_block.get("logging_steps", 50)),
        num_workers=int(train_block.get("num_workers", 0)),
    )

    seed = rl_cfg.get("seed")
    if seed is None:
        seed = cfg.seed

    device = rl_cfg.get("device")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vla_ckpt = resolve_vla_checkpoint_path(
        rl_cfg.get("vla_checkpoint"),
        rl_cfg.get("vla_checkpoint_dir"),
    )

    return TrainRLTokenSettings(
        vla_checkpoint=vla_ckpt,
        dataset=TrainRLTokenDatasetSettings(
            local_path=str(dataset_path),
            task_index=task_index,
            episode_slice=episode_slice,
        ),
        save_dir=Path(rl_cfg.get("save_dir", "./checkpoints/rl_token")).expanduser().resolve(),
        device=str(device),
        seed=int(seed),
        validate_vla_checkpoint=bool(rl_cfg.get("validate_vla_checkpoint", True)),
        align_dims_to_vla=bool(rl_cfg.get("align_dims_to_vla", True)),
        network_cfg=dict(rl_cfg.get("network") or {}),
        training=training,
        resolved_episodes=resolved,
        task_description=task_description,
    )


def _checkpoint_metadata(
    settings: TrainRLTokenSettings,
    net_dims: TrainRLTokenNetworkDims,
    vla_hidden_dim: int,
    step: int,
    loss: Optional[float] = None,
) -> Dict[str, Any]:
    meta = {
        "step": step,
        "vla_hidden_dim": vla_hidden_dim,
        "rl_token_dim": net_dims.rl_token_dim,
        "model_dim": net_dims.model_dim,
        "input_dim": net_dims.input_dim,
        "vla_checkpoint": str(settings.vla_checkpoint),
        "dataset_path": settings.dataset.local_path,
        "task_index": settings.dataset.task_index,
        "task_description": settings.task_description,
        "resolved_episode_count": len(settings.resolved_episodes) if settings.resolved_episodes else None,
        "align_dims_to_vla": settings.align_dims_to_vla,
    }
    if loss is not None:
        meta["loss"] = loss
    return meta


def build_rl_dataset(
    cfg: ScriptConfig,
    settings: TrainRLTokenSettings,
) -> "LeRobotDatasetSubset":
    dataset_path = Path(settings.dataset.local_path)
    info = load_dataset_info(dataset_path)
    fps = info.get("fps", 10)
    episodes_kw: Dict[str, Any] = {}
    if settings.resolved_episodes is not None:
        episodes_kw["episodes"] = settings.resolved_episodes
        ep_slice = settings.dataset.episode_slice
        if ep_slice is not None and len(ep_slice) > 0:
            preview = ep_slice[:10]
            suffix = "..." if len(ep_slice) > 10 else ""
            print(f"  使用 episode_slice: {preview}{suffix}")
        elif settings.dataset.task_index is not None:
            print(
                f"  使用 task_index={settings.dataset.task_index} ({settings.task_description!r})，"
                f"共 {len(settings.resolved_episodes)} 个 episode"
            )
    else:
        print("  使用全部 episode (task_index=null, episode_slice=null)")

    return LeRobotDatasetSubset(
        repo_id=dataset_path.name,
        root=str(dataset_path),
        delta_timestamps=create_delta_timestamps(cfg.action_horizon, fps),
        **episodes_kw,
    )


def train_rl_token(cfg: ScriptConfig, settings: TrainRLTokenSettings) -> None:
    if not HAS_LEROBOT or LeRobotDatasetSubset is None:
        raise ImportError("lerobot is required: pip install lerobot==0.3.3")

    set_seed(settings.seed)
    device = torch.device(settings.device)
    cfg.dataset_path = settings.dataset.local_path

    print(f"Loading frozen VLA from checkpoint: {settings.vla_checkpoint}")
    if settings.validate_vla_checkpoint:
        is_valid, ckpt_info = validate_checkpoint(
            str(settings.vla_checkpoint),
            cfg.config_path,
            str(device),
            dataset_path=settings.dataset.local_path,
        )
        if not is_valid:
            raise RuntimeError(f"checkpoint validation failed: {ckpt_info.get('errors', [])}")

    test_config_path = get_test_model_config(
        cfg.config_path, dataset_path=settings.dataset.local_path, use_test_config=False
    )
    temp_config_path = test_config_path if test_config_path != cfg.config_path else None
    try:
        model, _normalizer = load_model_from_checkpoint_with_lora_support(
            str(settings.vla_checkpoint), test_config_path, str(device)
        )
    finally:
        if temp_config_path and Path(temp_config_path).exists():
            Path(temp_config_path).unlink()

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    vla_hidden_dim = model.qwen_vl_interface.get_hidden_dim()
    net_dims = resolve_rl_token_network_dims(
        settings.network_cfg, vla_hidden_dim, settings.align_dims_to_vla
    )
    print(
        f"VLA hidden_dim={vla_hidden_dim} | "
        f"RLToken input_dim={net_dims.input_dim} model_dim={net_dims.model_dim} "
        f"rl_token_dim={net_dims.rl_token_dim}"
    )

    rl_module = RLTokenBottleneck(
        input_dim=net_dims.input_dim,
        model_dim=net_dims.model_dim,
        num_encoder_layers=net_dims.num_encoder_layers,
        num_decoder_layers=net_dims.num_decoder_layers,
        num_heads=net_dims.num_heads,
        ffn_dim=net_dims.ffn_dim,
        dropout=net_dims.dropout,
        rl_token_dim=net_dims.rl_token_dim,
    ).to(device)

    tr = settings.training
    print(f"\n创建 LeRobot 数据集: {settings.dataset.local_path}")
    ds = build_rl_dataset(cfg, settings)
    cfn = create_collate_fn(cfg.image_keys, cfg.state_key, cfg.image_size, True)
    loader = DataLoader(
        ds,
        batch_size=tr.batch_size,
        shuffle=True,
        num_workers=tr.num_workers,
        collate_fn=cfn,
    )

    optimizer = AdamW(rl_module.parameters(), lr=tr.learning_rate, weight_decay=tr.weight_decay)
    losses: List[float] = []
    settings.save_dir.mkdir(parents=True, exist_ok=True)

    def save_ckpt(path: Path, step: int, loss_val: Optional[float] = None) -> None:
        payload = {
            **_checkpoint_metadata(settings, net_dims, vla_hidden_dim, step, loss_val),
            "rl_token_state_dict": rl_module.state_dict(),
        }
        torch.save(payload, path)

    it = iter(loader)
    pbar = tqdm(range(tr.max_steps), total=tr.max_steps, desc="RLToken Pretrain")
    for step in pbar:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)

        inputs = {"images": batch["images"], "instructions": batch["text"]}
        if "state" in batch:
            inputs["states"] = batch["state"].to(device)
        with torch.no_grad():
            z_tokens = model.extract_vla_tokens(inputs).clone()
        out = rl_module.reconstruction_loss(z_tokens)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val = float(loss.item())
        losses.append(val)
        if (step + 1) % tr.logging_steps == 0:
            pbar.set_postfix(loss=f"{val:.4f}", avg=f"{np.mean(losses):.4f}")
        if (step + 1) % tr.save_steps == 0:
            save_ckpt(settings.save_dir / f"rl_token_step_{step + 1}.pt", step + 1, val)

    final_path = settings.save_dir / "rl_token_final.pt"
    save_ckpt(final_path, tr.max_steps)
    print(f"saved rl-token checkpoint: {final_path}")


def get_train_rl_token_config_from_raw(raw: Dict[str, Any]) -> Dict[str, Any]:
    """供 test / 在线脚本读取 train_rl_token 配置。"""
    if "train_rl_token" not in raw:
        raise KeyError("missing top-level 'train_rl_token' in config")
    return raw["train_rl_token"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train RL token bottleneck (frozen VLA, train RLTokenBottleneck only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            f"  python train_rl_token.py\n"
            f"  python train_rl_token.py --config {DEFAULT_CONFIG_PATH}\n"
            "\n"
            "所有路径、任务编号与超参请在 config 顶层 train_rl_token 段配置。\n"
        ),
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help=f"配置文件路径（默认 {DEFAULT_CONFIG_PATH}）",
    )
    args = parser.parse_args()

    cfg = load_script_config(args.config, dataset_path=None)
    settings = load_train_rl_token_settings(cfg.raw_config, cfg)
    cfg.dataset_path = settings.dataset.local_path
    cfg.seed = settings.seed

    train_rl_token(cfg, settings)


if __name__ == "__main__":
    main()
