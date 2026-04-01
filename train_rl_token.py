"""
Train RL token bottleneck with frozen VLA token embeddings.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.ScriptedVLA.cli import add_common_args, parse_common_args
from src.ScriptedVLA.model import QwenGR00TVLAModel, RLTokenBottleneck
from src.ScriptedVLA.utils import ensure_offline_mode_if_needed, load_script_config

ensure_offline_mode_if_needed()

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    HAS_LEROBOT = True
except ImportError:
    HAS_LEROBOT = False
    LeRobotDataset = None


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


def build_vla_model(raw_cfg: dict, cfg) -> QwenGR00TVLAModel:
    model_cfg = raw_cfg.get("model", {})
    vla_cfg = model_cfg.get("vla", {})
    action_head_cfg = model_cfg.get("action_head", {}).copy()
    action_head_cfg["action_horizon"] = cfg.action_horizon
    action_head_cfg["action_dim"] = cfg.action_dim
    model = QwenGR00TVLAModel(
        vlm_config=model_cfg.get("vlm", {}),
        action_head_config=action_head_cfg,
        use_state_vlm=vla_cfg.get("use_state_vlm", vla_cfg.get("use_state", True)),
        use_state_action_head=vla_cfg.get("use_state_action_head", vla_cfg.get("use_state", True)),
        state_dim=cfg.state_dim,
        future_action_window_size=cfg.action_horizon - 1,
    )
    return model


def train_rl_token(cfg) -> None:
    if not HAS_LEROBOT:
        raise ImportError("lerobot is required: pip install lerobot==0.3.3")

    set_seed(cfg.seed)
    raw = cfg.raw_config
    model = build_vla_model(raw, cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    rl_cfg = raw.get("model", {}).get("rl_token", {})
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

    train_cfg = raw.get("training", {}).get("rl_token_pretrain", {})
    lr = train_cfg.get("learning_rate", 1e-4)
    weight_decay = train_cfg.get("weight_decay", 0.0)
    max_steps = train_cfg.get("max_steps", cfg.max_steps)
    save_steps = train_cfg.get("save_steps", cfg.save_steps)
    log_steps = train_cfg.get("logging_steps", cfg.logging_steps)

    dataset_path = Path(cfg.dataset_path).resolve()
    info = load_dataset_info(dataset_path)
    fps = info.get("fps", 10)
    ds = LeRobotDataset(
        repo_id=dataset_path.name,
        root=str(dataset_path),
        delta_timestamps=create_delta_timestamps(cfg.action_horizon, fps),
        episodes=raw.get("dataset", {}).get("episode_slice"),
    )
    cfn = create_collate_fn(cfg.image_keys, cfg.state_key, cfg.image_size, True)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0, collate_fn=cfn)

    optimizer = AdamW(rl_module.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    save_dir = Path(train_cfg.get("save_dir", "./checkpoints/rl_token"))
    save_dir.mkdir(parents=True, exist_ok=True)

    it = iter(loader)
    pbar = tqdm(range(max_steps), total=max_steps, desc="RLToken Pretrain")
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
            z_tokens = model.extract_vla_tokens(inputs)
        out = rl_module.reconstruction_loss(z_tokens)
        loss = out["loss"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        val = float(loss.item())
        losses.append(val)
        if (step + 1) % log_steps == 0:
            pbar.set_postfix(loss=f"{val:.4f}", avg=f"{np.mean(losses):.4f}")
        if (step + 1) % save_steps == 0:
            torch.save(
                {"step": step + 1, "rl_token_state_dict": rl_module.state_dict(), "loss": val},
                save_dir / f"rl_token_step_{step + 1}.pt",
            )

    final_path = save_dir / "rl_token_final.pt"
    torch.save({"step": max_steps, "rl_token_state_dict": rl_module.state_dict()}, final_path)
    print(f"saved rl-token checkpoint: {final_path}")


def main():
    parser = argparse.ArgumentParser(description="Train RL token bottleneck")
    add_common_args(parser, include_config=True, include_device=False, include_seed=True, include_dataset=True)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    args = parser.parse_args()
    common = parse_common_args(args)
    cfg = load_script_config(
        common.config_path,
        dataset_path=common.dataset_path,
        max_steps=args.max_steps,
        save_steps=args.save_steps,
        seed=common.seed,
    )
    train_rl_token(cfg)


if __name__ == "__main__":
    main()
