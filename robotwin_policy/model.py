"""RoboTwin policy server model.

Loads pretrained Qwen2-VL-2B from the local cache and a randomly initialized
flow-matching action head. It does not load install_battery or any other
trained VLA checkpoint.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

USER_ROOT = Path("/data/share/1919650160032350208/users/lxg")
SCRIPTED_VLA_ROOT = USER_ROOT / "ScriptedVLA"
RANDOM_CKPT = SCRIPTED_VLA_ROOT / "checkpoints" / "robotwin_random_init" / "action_head.pt"
TRAINED_DIR = SCRIPTED_VLA_ROOT / "checkpoints" / "robotwin_adjust_bottle"
QWEN_CACHE = SCRIPTED_VLA_ROOT / "cache" / "models"

sys.path.insert(0, str(SCRIPTED_VLA_ROOT / "src"))

from ScriptedVLA.model import QwenGR00TVLAModel  # noqa: E402
from ScriptedVLA.utils.normalization import Normalizer  # noqa: E402

try:
    from XPolicyLab.model_template import ModelTemplate
    from XPolicyLab.utils.process_data import get_action_dim, get_robot_action_dim_info
except ImportError:  # Local editing before XPolicyLab is installed.
    ModelTemplate = object

    def get_action_dim(_env_cfg_type):
        return 14

    def get_robot_action_dim_info(_env_cfg_type):
        return {"arm_dim": [6, 6], "ee_dim": [1, 1]}


CAMERA_KEYS = ("cam_head", "cam_left_wrist", "cam_right_wrist")
STATE_KEYS = (
    "left_arm_joint_state",
    "left_ee_joint_state",
    "right_arm_joint_state",
    "right_ee_joint_state",
)
RAW_CAMERA_KEYS = ("head_camera", "left_camera", "right_camera")


def _as_vector(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float32).reshape(-1)


def _instruction(obs: dict) -> str:
    for key in ("language", "instruction", "instructions"):
        if key not in obs or obs[key] in (None, ""):
            continue
        value = obs[key]
        if isinstance(value, str):
            return value
        if isinstance(value, (list, tuple)) and value:
            return str(value[0])
        if isinstance(value, dict):
            for inner in ("seen", "instruction", "text"):
                if value.get(inner):
                    return str(value[inner])
    return "perform the robot task"


def _rgb_image(camera) -> Image.Image:
    payload = camera.get("rgb", camera.get("color")) if isinstance(camera, dict) else camera
    array = np.asarray(payload)
    if array.ndim == 3 and array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if np.nanmax(array) <= 1.5:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")


class Model(ModelTemplate):
    def __init__(self, model_cfg):
        self.model_cfg = model_cfg
        self.action_type = str(model_cfg.get("action_type", "joint"))
        self.env_cfg_type = str(model_cfg.get("env_cfg_type", "arx_x5"))
        self.seed = int(model_cfg.get("seed", 0))
        self.replan_steps = int(model_cfg.get("replan_steps", 8))
        self.action_horizon = int(model_cfg.get("action_horizon", max(self.replan_steps, 8)))
        self.ckpt_name = str(model_cfg.get("ckpt_name", "random_init"))
        self.normalizer = None
        self.robot_action_dim_info = get_robot_action_dim_info(self.env_cfg_type)
        self.action_dim = int(get_action_dim(self.env_cfg_type))
        arm_dims = [int(v) for v in self.robot_action_dim_info["arm_dim"]]
        ee_dims = [int(v) for v in self.robot_action_dim_info["ee_dim"]]
        if len(arm_dims) != 2 or len(ee_dims) != 2:
            raise ValueError(f"scripted_vla smoke expects a dual-arm profile, got {self.robot_action_dim_info}")
        if self.action_type != "joint":
            raise ValueError("scripted_vla smoke uses joint actions; pass action_type=joint")
        expected = sum(arm_dims) + sum(ee_dims)
        if self.action_dim != expected:
            self.action_dim = expected
        self.arm_dims = arm_dims
        self.ee_dims = ee_dims

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        torch.manual_seed(self.seed)
        horizon = self.action_horizon
        self.vla = QwenGR00TVLAModel(
            vlm_config={
                "model_name": "Qwen/Qwen2-VL-2B-Instruct",
                "image_size": 224,
                "max_seq_length": 512,
                "freeze_vlm": True,
                "cache_dir": str(QWEN_CACHE),
                "use_state": False,
            },
            action_head_config={
                "type": "flow_matching",
                "hidden_dim": 1536,
                "num_layers": 6,
                "num_heads": 12,
                "mlp_ratio": 4.0,
                "action_dim": self.action_dim,
                "action_horizon": horizon,
                "num_inference_timesteps": 10,
                "norm_type": "ada_norm",
                "use_state": True,
            },
            camera_names=list(CAMERA_KEYS),
            use_state_vlm=False,
            use_state_action_head=True,
            state_dim=self.action_dim,
            future_action_window_size=horizon - 1,
        )
        if self.ckpt_name == "adjust_bottle":
            self._load_trained_action_head()
        else:
            self._load_or_create_random_action_head()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.vla.to(self.device).eval()
        self._obs = None
        self._logged_obs = False
        print(
            f"[scripted_vla] random action head ready dim={self.action_dim} "
            f"ckpt={RANDOM_CKPT} device={self.device}"
        )

    def _load_or_create_random_action_head(self) -> None:
        RANDOM_CKPT.parent.mkdir(parents=True, exist_ok=True)
        if RANDOM_CKPT.is_file():
            blob = torch.load(RANDOM_CKPT, map_location="cpu")
            if blob.get("kind") != "scripted_vla_random_action_head":
                raise RuntimeError(f"Refusing to load a non-random checkpoint: {RANDOM_CKPT}")
            self.vla.action_model.load_state_dict(blob["action_head"])
            print(f"[scripted_vla] loaded random action head {RANDOM_CKPT}")
            return
        torch.save(
            {
                "kind": "scripted_vla_random_action_head",
                "seed": self.seed,
                "action_dim": self.action_dim,
                "action_head": self.vla.action_model.state_dict(),
            },
            RANDOM_CKPT,
        )
        print(f"[scripted_vla] created random action head {RANDOM_CKPT}")

    def _load_trained_action_head(self) -> None:
        ckpts = sorted(TRAINED_DIR.glob("checkpoint_step_*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No trained checkpoint in {TRAINED_DIR}")
        ckpt_path = ckpts[-1]
        blob = torch.load(ckpt_path, map_location="cpu")
        state = blob.get("model_state_dict")
        if not isinstance(state, dict):
            raise RuntimeError(f"Checkpoint has no model_state_dict: {ckpt_path}")
        prefix = "action_model."
        action_state = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not action_state:
            raise RuntimeError(f"Checkpoint has no action_model weights: {ckpt_path}")
        self.vla.action_model.load_state_dict(action_state)
        if blob.get("normalizer"):
            self.normalizer = Normalizer.from_dict(blob["normalizer"])
        print(f"[scripted_vla] loaded trained action head {ckpt_path} normalizer={self.normalizer is not None}")

    def update_obs(self, obs):
        self._obs = obs
        if not self._logged_obs and isinstance(obs, dict):
            vision = obs.get("vision", {})
            state = obs.get("state", {})
            print(
                "[scripted_vla] obs keys",
                list(obs.keys()),
                "vision",
                list(vision) if isinstance(vision, dict) else type(vision).__name__,
                "state",
                list(state) if isinstance(state, dict) else type(state).__name__,
            )
            self._logged_obs = True

    def _images_and_state(self):
        obs = self._obs
        if not isinstance(obs, dict):
            raise TypeError(f"Expected observation dict, got {type(obs)}")
        vision = obs.get("vision")
        if isinstance(vision, dict) and all(key in vision for key in CAMERA_KEYS):
            images = [_rgb_image(vision[key]) for key in CAMERA_KEYS]
        else:
            source = obs.get("observation", obs)
            images = [_rgb_image(source[key]) for key in RAW_CAMERA_KEYS]
        state = obs.get("state", {})
        if isinstance(state, dict) and all(key in state for key in STATE_KEYS):
            vector = np.concatenate([_as_vector(state[key]) for key in STATE_KEYS])
        else:
            joint = obs.get("joint_action", {})
            vector = _as_vector(joint["vector"]) if isinstance(joint, dict) and "vector" in joint else None
        if vector is None or vector.shape[0] != self.action_dim:
            raise ValueError(
                f"State dim {None if vector is None else vector.shape} != action_dim {self.action_dim}"
            )
        return images, vector

    def _clip_action(self, action: np.ndarray) -> np.ndarray:
        clipped = action.astype(np.float32).copy()
        offset = 0
        for arm_dim, ee_dim in zip(self.arm_dims, self.ee_dims):
            clipped[offset:offset + arm_dim] = np.clip(clipped[offset:offset + arm_dim], -3.1416, 3.1416)
            offset += arm_dim
            clipped[offset:offset + ee_dim] = np.clip(clipped[offset:offset + ee_dim], 0.0, 1.0)
            offset += ee_dim
        return clipped

    def _pack(self, action: np.ndarray) -> dict:
        offset = 0
        packed = {"action_type": "qpos"}
        names = (
            ("left_arm_joint_state", "left_ee_joint_state"),
            ("right_arm_joint_state", "right_ee_joint_state"),
        )
        for (arm_key, ee_key), arm_dim, ee_dim in zip(names, self.arm_dims, self.ee_dims):
            packed[arm_key] = action[offset:offset + arm_dim]
            offset += arm_dim
            packed[ee_key] = action[offset:offset + ee_dim]
            offset += ee_dim
        return packed

    @torch.inference_mode()
    def get_action(self):
        images, state = self._images_and_state()
        if self.normalizer is not None:
            state = np.asarray(self.normalizer.normalize_state(state), dtype=np.float32)
        prediction = self.vla.predict_action(
            {
                "images": [images],
                "instructions": [_instruction(self._obs)],
                "states": torch.as_tensor(state, dtype=torch.float32, device=self.device).view(1, -1),
            }
        )
        actions = np.asarray(prediction["normalized_actions"][0], dtype=np.float32)
        if self.normalizer is not None:
            actions = np.asarray(self.normalizer.denormalize_action(actions), dtype=np.float32)
        steps = min(self.replan_steps, actions.shape[0])
        packed = []
        for row in actions[:steps]:
            clipped = self._clip_action(row)
            packed.append(self._pack(clipped))
        print(
            f"[scripted_vla] action chunk {actions.shape} -> {steps} steps "
            f"raw0={actions[0, :4].round(3).tolist()} clipped0={packed[0]['left_arm_joint_state'][:4].round(3).tolist()}"
        )
        return packed

    def reset(self):
        self._obs = None
        print("[scripted_vla] reset")
