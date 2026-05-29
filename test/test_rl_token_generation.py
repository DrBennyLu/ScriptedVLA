"""
使用训练好的 VLA 与 RLToken 权重，对 LeRobot 数据集帧做 RL token 生成稳定性测试。

默认从 libero/config_libero_object.yaml 的 train_rl_token 段读取路径与网络配置。

测试项:
  1. test_rl_token_generation_stability — 同帧重复前向一致性
  2. test_rl_token_rollout_frame_cosine — 单条 task6 rollout 内各帧 vs 首帧余弦相似度并绘图
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ScriptedVLA.model import RLTokenBottleneck
from src.ScriptedVLA.utils import load_script_config
from test.test_inference import (
    get_test_model_config,
    load_model_from_checkpoint_with_lora_support,
    validate_checkpoint,
)
from train_rl_token import (
    DEFAULT_CONFIG_PATH,
    create_delta_timestamps,
    get_train_rl_token_config_from_raw,
    load_train_rl_token_settings,
    resolve_rl_token_network_dims,
)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:
    raise ImportError("请先安装 lerobot: pip install lerobot==0.3.3") from exc

try:
    from train_rl_token import LeRobotDatasetSubset
except ImportError:
    LeRobotDatasetSubset = LeRobotDataset

from libero.libero_dataset_replay import get_episode_ids_for_task_index, resolve_training_episodes


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


def load_single_episode_dataset(
    dataset_path: Path,
    action_horizon: int,
    episode_id: int,
) -> LeRobotDataset:
    """仅加载单个 episode（一条 rollout）。"""
    info_file = dataset_path / "meta" / "info.json"
    if not info_file.exists():
        raise FileNotFoundError(f"missing info.json: {info_file}")
    import json

    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    fps = info.get("fps", 10)
    return LeRobotDatasetSubset(
        repo_id=dataset_path.name,
        root=str(dataset_path),
        delta_timestamps=create_delta_timestamps(action_horizon, fps),
        episodes=[int(episode_id)],
    )


def load_lerobot_dataset_subset(
    dataset_path: Path,
    action_horizon: int,
    task_index: Optional[int],
    episode_slice: Optional[List[int]],
) -> LeRobotDataset:
    info_file = dataset_path / "meta" / "info.json"
    if not info_file.exists():
        raise FileNotFoundError(f"missing info.json: {info_file}")
    import json

    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    fps = info.get("fps", 10)
    resolved = resolve_training_episodes(str(dataset_path), task_index, episode_slice)
    episodes_kw = {"episodes": resolved} if resolved is not None else {}
    return LeRobotDatasetSubset(
        repo_id=dataset_path.name,
        root=str(dataset_path),
        delta_timestamps=create_delta_timestamps(action_horizon, fps),
        **episodes_kw,
    )


def sample_frame_indices(total: int, num_frames: int) -> List[int]:
    if total <= 0:
        return []
    if num_frames >= total:
        return list(range(total))
    if num_frames == 1:
        return [0]
    step = max(1, total // num_frames)
    return list(range(0, total, step))[:num_frames]


def build_single_input(sample: Dict, image_keys: List[str], state_key: str, device: torch.device) -> Dict:
    images = []
    for k in image_keys:
        if k not in sample:
            raise KeyError(f"sample missing image key: {k}")
        images.append(tensor_to_pil_image(sample[k]))

    if len(images) == 1:
        images = [images[0]]
    else:
        images = [images]

    text = str(sample.get("task", ""))
    inputs = {"images": images, "instructions": [text]}
    if state_key in sample:
        state = sample[state_key]
        if isinstance(state, torch.Tensor):
            if state.dim() == 1:
                state = state.unsqueeze(0)
            inputs["states"] = state.to(device=device, dtype=torch.float32)
    return inputs


def resolve_paths_from_config(config_path: str) -> Dict[str, str]:
    cfg = load_script_config(config_path)
    settings = load_train_rl_token_settings(cfg.raw_config, cfg)
    rl_ckpt = settings.save_dir / "rl_token_final.pt"
    return {
        "config_path": config_path,
        "dataset_path": settings.dataset.local_path,
        "vla_checkpoint": str(settings.vla_checkpoint),
        "rl_checkpoint": str(rl_ckpt),
        "task_index": settings.dataset.task_index,
        "episode_slice": settings.dataset.episode_slice,
        "settings": settings,
        "cfg": cfg,
    }


def load_trained_modules(
    config_path: str,
    dataset_path: str,
    vla_checkpoint: str,
    rl_checkpoint: str,
    device: torch.device,
):
    ok, info = validate_checkpoint(vla_checkpoint, config_path, str(device), dataset_path=dataset_path)
    if not ok:
        raise RuntimeError(f"VLA checkpoint验证失败: {info.get('errors', [])}")

    test_config_path = get_test_model_config(config_path, dataset_path=dataset_path, use_test_config=False)
    temp_config_path = test_config_path if test_config_path != config_path else None
    try:
        model, _ = load_model_from_checkpoint_with_lora_support(vla_checkpoint, test_config_path, str(device))
    finally:
        if temp_config_path and Path(temp_config_path).exists():
            Path(temp_config_path).unlink()
    model.eval()

    cfg = load_script_config(config_path, dataset_path=dataset_path)
    rl_cfg = get_train_rl_token_config_from_raw(cfg.raw_config)
    align_dims = bool(rl_cfg.get("align_dims_to_vla", True))
    network_cfg = dict(rl_cfg.get("network") or {})
    vla_hidden_dim = model.qwen_vl_interface.get_hidden_dim()
    net_dims = resolve_rl_token_network_dims(network_cfg, vla_hidden_dim, align_dims)

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

    ckpt = torch.load(rl_checkpoint, map_location=device)
    state_dict = ckpt.get("rl_token_state_dict")
    if state_dict is None:
        raise KeyError(f"{rl_checkpoint} 中缺少 rl_token_state_dict")
    rl_module.load_state_dict(state_dict, strict=True)
    rl_module.eval()

    ckpt_rl_dim = ckpt.get("rl_token_dim")
    if ckpt_rl_dim is not None and int(ckpt_rl_dim) != net_dims.rl_token_dim:
        raise ValueError(
            f"checkpoint rl_token_dim={ckpt_rl_dim} != config/model rl_token_dim={net_dims.rl_token_dim}"
        )
    return cfg, model, rl_module, net_dims


def test_rl_token_generation_stability(
    config_path: str = DEFAULT_CONFIG_PATH,
    vla_checkpoint: Optional[str] = None,
    rl_checkpoint: Optional[str] = None,
    num_frames: int = 5,
    repeats_per_frame: int = 5,
) -> bool:
    print("=" * 80)
    print("RL token 真实权重稳定性测试")
    print("=" * 80)

    resolved = resolve_paths_from_config(config_path)
    if vla_checkpoint is None:
        vla_checkpoint = resolved["vla_checkpoint"]
    if rl_checkpoint is None:
        rl_checkpoint = resolved["rl_checkpoint"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_path = resolved["dataset_path"]
    settings = resolved["settings"]
    set_seed(settings.seed)

    print(f"device: {device}")
    print(f"dataset: {dataset_path}")
    if settings.dataset.task_index is not None:
        print(f"task_index: {settings.dataset.task_index} ({settings.task_description!r})")
    print(f"vla checkpoint: {vla_checkpoint}")
    print(f"rl checkpoint: {rl_checkpoint}")

    cfg, model, rl_module, net_dims = load_trained_modules(
        config_path=config_path,
        dataset_path=dataset_path,
        vla_checkpoint=vla_checkpoint,
        rl_checkpoint=rl_checkpoint,
        device=device,
    )

    dataset = load_lerobot_dataset_subset(
        Path(dataset_path),
        cfg.action_horizon,
        settings.dataset.task_index,
        settings.dataset.episode_slice,
    )
    frame_indices = sample_frame_indices(len(dataset), num_frames)
    if not frame_indices:
        raise RuntimeError("数据集为空，无法测试")
    print(f"subset frames: {len(dataset)}, selected indices: {frame_indices}")

    frame_tokens = []
    max_same_frame_diff = 0.0
    vla_hidden_dim = model.qwen_vl_interface.get_hidden_dim()

    with torch.no_grad():
        for frame_idx in frame_indices:
            sample = dataset[frame_idx]
            inputs = build_single_input(sample, cfg.image_keys, cfg.state_key, device)
            runs = []
            for _ in range(repeats_per_frame):
                z_tokens = model.extract_vla_tokens(inputs).clone()
                z_rl = rl_module.encode(z_tokens).float()
                assert z_rl.shape[-1] == net_dims.rl_token_dim == vla_hidden_dim, (
                    f"z_rl dim {z_rl.shape[-1]} != expected {vla_hidden_dim}"
                )
                runs.append(z_rl)

            base = runs[0]
            diffs = [(r - base).abs().max().item() for r in runs[1:]]
            frame_max_diff = max(diffs) if diffs else 0.0
            max_same_frame_diff = max(max_same_frame_diff, frame_max_diff)
            mean_norm = torch.stack([r.norm(dim=-1) for r in runs]).mean().item()
            print(
                f"frame={frame_idx:6d} | repeats={repeats_per_frame} | "
                f"max_abs_diff={frame_max_diff:.3e} | mean_norm={mean_norm:.4f} | "
                f"z_rl_dim={z_rl.shape[-1]}"
            )
            frame_tokens.append(base.squeeze(0))

    cos_vals = []
    for i in range(len(frame_tokens) - 1):
        a = frame_tokens[i].unsqueeze(0)
        b = frame_tokens[i + 1].unsqueeze(0)
        cos_vals.append(F.cosine_similarity(a, b).item())

    if cos_vals:
        print(
            f"adjacent-frame cosine: min={min(cos_vals):.6f}, "
            f"max={max(cos_vals):.6f}, mean={sum(cos_vals)/len(cos_vals):.6f}"
        )

    assert torch.isfinite(torch.stack(frame_tokens)).all(), "发现 NaN/Inf 的 RL token"
    assert max_same_frame_diff < 1e-6, f"同一帧重复生成不稳定: max_abs_diff={max_same_frame_diff:.3e}"
    print("✓ 稳定性与数值检查通过")
    return True


def _encode_rl_token_for_sample(
    sample: Dict,
    model,
    rl_module: RLTokenBottleneck,
    image_keys: List[str],
    state_key: str,
    device: torch.device,
) -> torch.Tensor:
    inputs = build_single_input(sample, image_keys, state_key, device)
    z_tokens = model.extract_vla_tokens(inputs).clone()
    return rl_module.encode(z_tokens).float().squeeze(0)


def _plot_rollout_cosine(
    cos_to_frame0: np.ndarray,
    episode_id: int,
    task_index: int,
    task_description: Optional[str],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    frame_ids = np.arange(len(cos_to_frame0))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(frame_ids, cos_to_frame0, color="#2563eb", linewidth=1.2, label="cos(z_rl[t], z_rl[0])")
    ax.axhline(1.0, color="#94a3b8", linestyle="--", linewidth=0.9, label="identical (1.0)")
    ax.set_xlabel("Frame index in rollout")
    ax.set_ylabel("Cosine similarity to frame 0")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    title = f"RL token vs frame-0 (task_index={task_index}, episode={episode_id})"
    if task_description:
        title += f"\n{task_description[:80]}{'...' if len(task_description) > 80 else ''}"
    ax.set_title(title)
    rest = cos_to_frame0[1:] if len(cos_to_frame0) > 1 else cos_to_frame0
    stats = (
        f"min={rest.min():.3f}, max={rest.max():.3f}, mean={rest.mean():.3f}"
        if len(rest) > 0
        else "n/a"
    )
    ax.text(
        0.02,
        0.02,
        f"frames={len(cos_to_frame0)} | vs frame0 (excl. t=0): {stats}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"saved cosine plot: {output_path}")


def test_rl_token_rollout_frame_cosine(
    config_path: str = DEFAULT_CONFIG_PATH,
    vla_checkpoint: Optional[str] = None,
    rl_checkpoint: Optional[str] = None,
    episode_id: Optional[int] = None,
    frame_stride: int = 1,
    max_frames: Optional[int] = None,
    output_path: Optional[str] = None,
    max_cosine_to_prove_diversity: float = 0.99,
) -> bool:
    """
    在 task6 的单条 rollout 上，计算各帧 RL token 与首帧的余弦相似度并绘图。

    若不同帧的表征确实不同，则除 t=0 外相似度应明显低于 1.0。
    """
    print("=" * 80)
    print("RL token rollout 帧间余弦相似度测试 (vs 首帧)")
    print("=" * 80)

    resolved = resolve_paths_from_config(config_path)
    if vla_checkpoint is None:
        vla_checkpoint = resolved["vla_checkpoint"]
    if rl_checkpoint is None:
        rl_checkpoint = resolved["rl_checkpoint"]

    settings = resolved["settings"]
    task_index = settings.dataset.task_index
    if task_index is None:
        raise ValueError("train_rl_token.dataset.task_index 未设置，无法进行单任务 rollout 测试")

    dataset_path = Path(resolved["dataset_path"])
    episode_ids = get_episode_ids_for_task_index(str(dataset_path), int(task_index))
    if episode_id is None:
        episode_id = episode_ids[0]
    elif episode_id not in episode_ids:
        raise ValueError(f"episode_id={episode_id} 不属于 task_index={task_index}，可选: {episode_ids[:5]}...")

    if output_path is None:
        output_path = str(
            project_root / "results" / f"rl_token_task{task_index}_ep{episode_id}_rollout_cosine.png"
        )
    out_path = Path(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(settings.seed)

    print(f"device: {device}")
    print(f"dataset: {dataset_path}")
    print(f"task_index: {task_index} ({settings.task_description!r})")
    print(f"episode_id: {episode_id} (task6 共 {len(episode_ids)} 条 rollout)")
    print(f"frame_stride: {frame_stride}, max_frames: {max_frames}")
    print(f"output: {out_path}")

    cfg, model, rl_module, net_dims = load_trained_modules(
        config_path=config_path,
        dataset_path=str(dataset_path),
        vla_checkpoint=vla_checkpoint,
        rl_checkpoint=rl_checkpoint,
        device=device,
    )

    episode_ds = load_single_episode_dataset(dataset_path, cfg.action_horizon, episode_id)
    n_total = len(episode_ds)
    if n_total < 2:
        raise RuntimeError(f"episode {episode_id} 帧数过少 ({n_total})，无法比较")

    frame_indices = list(range(0, n_total, max(1, frame_stride)))
    if max_frames is not None and len(frame_indices) > max_frames:
        step = max(1, len(frame_indices) // max_frames)
        frame_indices = frame_indices[::step][:max_frames]
    if frame_indices[0] != 0:
        frame_indices = [0] + [i for i in frame_indices if i != 0]

    z_rl_list: List[torch.Tensor] = []
    with torch.no_grad():
        for local_idx in tqdm(frame_indices, desc=f"encode ep{episode_id}", leave=False):
            sample = episode_ds[local_idx]
            z_rl = _encode_rl_token_for_sample(
                sample, model, rl_module, cfg.image_keys, cfg.state_key, device
            )
            z_rl_list.append(z_rl)

    z0 = z_rl_list[0].unsqueeze(0)
    cos_vals: List[float] = []
    for z in z_rl_list:
        cos_vals.append(F.cosine_similarity(z0, z.unsqueeze(0)).item())
    cos_arr = np.array(cos_vals, dtype=np.float64)

    print(f"rollout length (sampled): {len(cos_arr)} / {n_total} frames")
    if len(cos_arr) > 1:
        rest = cos_arr[1:]
        print(
            f"cos to frame0 (t>0): min={rest.min():.4f}, max={rest.max():.4f}, "
            f"mean={rest.mean():.4f}, std={rest.std():.4f}"
        )
    print(f"cos at frame0: {cos_arr[0]:.6f}")

    _plot_rollout_cosine(
        cos_arr,
        episode_id=episode_id,
        task_index=int(task_index),
        task_description=settings.task_description,
        output_path=out_path,
    )

    assert np.isfinite(cos_arr).all(), "cosine 含 NaN/Inf"
    assert abs(cos_arr[0] - 1.0) < 1e-5, f"首帧与自身余弦应为 1.0，得到 {cos_arr[0]}"
    if len(cos_arr) > 1:
        assert rest.min() < max_cosine_to_prove_diversity, (
            f"各帧 RL token 与首帧过于相似 (min cosine={rest.min():.4f} >= {max_cosine_to_prove_diversity})，"
            "未能证明帧间差异"
        )
        assert rest.std() > 1e-3, f"余弦曲线几乎无变化 (std={rest.std():.6f})"
    print("✓ rollout 帧间 RL token 差异性检查通过")
    return True


def run_all_tests(
    config_path: str = DEFAULT_CONFIG_PATH,
    skip_stability: bool = False,
    skip_rollout_cosine: bool = False,
    **rollout_kwargs,
) -> bool:
    ok = True
    if not skip_stability:
        ok = test_rl_token_generation_stability(config_path=config_path) and ok
    if not skip_rollout_cosine:
        ok = test_rl_token_rollout_frame_cosine(config_path=config_path, **rollout_kwargs) and ok
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL token 生成与 rollout 差异性测试")
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--skip-stability", action="store_true", help="跳过同帧稳定性测试")
    parser.add_argument(
        "--rollout-cosine-only",
        action="store_true",
        help="仅运行 rollout 余弦相似度测试",
    )
    parser.add_argument("--episode-id", type=int, default=None, help="指定 task6 的 episode_index")
    parser.add_argument("--frame-stride", type=int, default=1, help="rollout 内抽帧步长")
    parser.add_argument("--max-frames", type=int, default=None, help="最多编码帧数（过长 rollout 可限流）")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="余弦曲线图保存路径",
    )
    args = parser.parse_args()

    rollout_kw = dict(
        episode_id=args.episode_id,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        output_path=args.output,
    )
    if args.rollout_cosine_only:
        ok = test_rl_token_rollout_frame_cosine(config_path=args.config, **rollout_kw)
    else:
        ok = run_all_tests(
            config_path=args.config,
            skip_stability=False,
            skip_rollout_cosine=False,
            **rollout_kw,
        )
    raise SystemExit(0 if ok else 1)
