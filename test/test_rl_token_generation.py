"""
使用训练好的 VLA 与 RLToken 权重，对 LeRobot 数据集帧做 RL token 生成稳定性测试。

默认读取：
- VLA checkpoint: ./checkpoints/checkpoint_step_100000.pt
- RL token checkpoint: ./checkpoints/rl_token/rl_token_step_10000.pt
- 数据集路径: config.dataset.local_path（默认 ./dataset/libero_object/）
"""

import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ScriptedVLA.model import RLTokenBottleneck
from src.ScriptedVLA.utils import load_script_config
from test.test_inference import (
    get_test_model_config,
    load_model_from_checkpoint_with_lora_support,
    validate_checkpoint,
)
from train_rl_token import create_delta_timestamps

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:
    raise ImportError("请先安装 lerobot: pip install lerobot==0.3.3") from exc


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


def load_lerobot_dataset(dataset_path: Path, action_horizon: int) -> LeRobotDataset:
    info_file = dataset_path / "meta" / "info.json"
    if not info_file.exists():
        raise FileNotFoundError(f"missing info.json: {info_file}")
    import json

    with open(info_file, "r", encoding="utf-8") as f:
        info = json.load(f)
    fps = info.get("fps", 10)
    return LeRobotDataset(
        repo_id=dataset_path.name,
        root=str(dataset_path),
        delta_timestamps=create_delta_timestamps(action_horizon, fps),
    )


def sample_frame_indices(total: int, num_frames: int) -> List[int]:
    if total <= 0:
        return []
    if num_frames >= total:
        return list(range(total))
    if num_frames == 1:
        return [0]
    step = max(1, total // num_frames)
    idxs = list(range(0, total, step))[:num_frames]
    return idxs


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
        raise KeyError(f"{rl_checkpoint} 中缺少 rl_token_state_dict")
    rl_module.load_state_dict(state_dict, strict=True)
    rl_module.eval()
    return cfg, model, rl_module


def test_rl_token_generation_stability(
    config_path: str = "config.yaml",
    vla_checkpoint: str = "./checkpoints/checkpoint_step_100000.pt",
    rl_checkpoint: str = "./checkpoints/rl_token/rl_token_step_10000.pt",
    num_frames: int = 5,
    repeats_per_frame: int = 5,
) -> bool:
    print("=" * 80)
    print("RL token 真实权重稳定性测试")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg0 = load_script_config(config_path)
    dataset_path = str(Path(cfg0.dataset_path).resolve())
    set_seed(cfg0.seed)

    print(f"device: {device}")
    print(f"dataset: {dataset_path}")
    print(f"vla checkpoint: {vla_checkpoint}")
    print(f"rl checkpoint: {rl_checkpoint}")

    cfg, model, rl_module = load_trained_modules(
        config_path=config_path,
        dataset_path=dataset_path,
        vla_checkpoint=vla_checkpoint,
        rl_checkpoint=rl_checkpoint,
        device=device,
    )

    dataset = load_lerobot_dataset(Path(dataset_path), cfg.action_horizon)
    frame_indices = sample_frame_indices(len(dataset), num_frames)
    if not frame_indices:
        raise RuntimeError("数据集为空，无法测试")
    print(f"total frames: {len(dataset)}, selected: {frame_indices}")

    frame_tokens = []
    max_same_frame_diff = 0.0

    with torch.no_grad():
        for frame_idx in frame_indices:
            sample = dataset[frame_idx]
            inputs = build_single_input(sample, cfg.image_keys, cfg.state_key, device)
            runs = []
            for _ in range(repeats_per_frame):
                z_tokens = model.extract_vla_tokens(inputs).clone()
                z_rl = rl_module.encode(z_tokens).float()
                runs.append(z_rl)

            base = runs[0]
            diffs = [(r - base).abs().max().item() for r in runs[1:]]
            frame_max_diff = max(diffs) if diffs else 0.0
            max_same_frame_diff = max(max_same_frame_diff, frame_max_diff)
            mean_norm = torch.stack([r.norm(dim=-1) for r in runs]).mean().item()
            print(
                f"frame={frame_idx:6d} | repeats={repeats_per_frame} | "
                f"max_abs_diff={frame_max_diff:.3e} | mean_norm={mean_norm:.4f}"
            )
            frame_tokens.append(base.squeeze(0))

    # 跨帧可分性：统计相邻帧 token 余弦相似度
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


if __name__ == "__main__":
    ok = test_rl_token_generation_stability()
    raise SystemExit(0 if ok else 1)
