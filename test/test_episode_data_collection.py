"""
测试一个 episode 的数据采集与保存：执行完整的抓取红色方块并放入盒子流程，
在流程中按固定频率（默认 10Hz）记录双相机图像、机器人关节状态、末端目标 action，并保存到目录。

另含 test_lerobot_dataset_episode_collection：多 episode 采集，通过 LeRobotDataset.create /
new_episode / add_frame / save_episode / save 写入 LeRobot 数据集格式。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from simulator.pick_place_env import PickPlaceEnv

# LeRobotDataset：优先 lerobot.common.datasets（主流 2024–2025 创建/录制 API），否则回退 lerobot.datasets
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore[import-untyped]
except ImportError:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ImportError:
        LeRobotDataset = None


def _get_image_saver():
    """返回 save_image(arr, path) 函数，优先 cv2，其次 PIL，否则 npy。"""
    try:
        import cv2
        has_cv2 = True
    except ImportError:
        has_cv2 = False
    if not has_cv2:
        try:
            from PIL import Image as PILImage
        except ImportError:
            PILImage = None
    else:
        PILImage = None

    def save_image(arr: np.ndarray, path: Path) -> None:
        if has_cv2:
            bgr = np.asarray(arr[:, :, ::-1])
            cv2.imwrite(str(path), bgr)
        elif PILImage is not None:
            PILImage.fromarray(arr).save(path)
        else:
            np.save(path.with_suffix(".npy"), arr)

    return save_image


def test_episode_data_collection(
    use_gui: bool = False,
    output_dir: Path | None = None,
    collect_frequency_hz: float = 10.0,
    image_size: int = 224,
    seed: int = 42,
    steps_per_phase: int = 80,
):
    """
    执行一个完整 episode（抓取红色方块并放入盒子），在过程中按固定频率采集并保存数据。

    采集内容（与 test_data_collection_snapshot 一致）：
    - 固定视角图像 (top)、夹爪相机图像 (wrist)
    - 关节位置 (9 维)
    - 当前步的 action：末端执行器目标位置 + 夹爪 (4 维: x,y,z,gripper)

    数据按 step_000, step_001, ... 保存到 output_dir，每步含 top.png、wrist.png，
    以及 summary.txt 记录关节与 action。

    运行示例:
      python test/test_episode_data_collection.py
      python test/test_episode_data_collection.py --gui
    """
    if output_dir is None:
        output_dir = project_root / "test_output" / "episode_data_collection"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  输出目录: {output_dir}")
    print(f"  采集频率: {collect_frequency_hz} Hz")
    print(f"  图像尺寸: {image_size}x{image_size}")

    env = PickPlaceEnv(render=True, use_gui=use_gui, seed=seed)
    try:
        obs = env.reset()
        collected: list[dict] = []
        # 仿真不按真实时间限速，用步数近似 10Hz：每 collect_interval 步采一帧（40Hz 控制下每 4 步≈10Hz）
        control_hz = 40.0
        collect_interval = max(1, int(control_hz / collect_frequency_hz))
        step_count = [0]

        def on_before_step(action: np.ndarray) -> None:
            step_count[0] += 1
            if step_count[0] % collect_interval == 0:
                snap = env.collect_snapshot(
                    action,
                    image_width=image_size,
                    image_height=image_size,
                )
                collected.append(snap)

        print("\n  执行 execute_pick_place('red')（非实时，尽快跑完）并每 {} 步采集一帧 (~{} Hz)...".format(collect_interval, collect_frequency_hz))
        success, reward, done = env.execute_pick_place(
            "red",
            steps_per_phase=steps_per_phase,
            on_before_step=on_before_step,
            use_real_time=False,
        )
        print(f"\n  Episode 结束: success={success}, reward={reward}, done={done}")
        print(f"  共采集 {len(collected)} 帧 (每 {collect_interval} 步一帧)")

        save_image_fn = _get_image_saver()
        for step, snap in enumerate(collected):
            step_dir = output_dir / f"step_{step:04d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            save_image_fn(snap["observation.images.top"], step_dir / "top.png")
            save_image_fn(snap["observation.images.wrist"], step_dir / "wrist.png")
            if step < 3 or step >= len(collected) - 2:
                print(f"  Step {step}: joint (9) = {snap['observation.state'][:3]}... action = {snap['action']}")
            elif step == 3:
                print("  ...")

        summary_path = output_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"episode_data_collection\n")
            f.write(f"num_frames={len(collected)}\n")
            f.write(f"collect_frequency_hz={collect_frequency_hz}\n")
            f.write(f"image_size={image_size}\n")
            f.write(f"success={success} reward={reward} done={done}\n")
            for step, snap in enumerate(collected):
                f.write(f"\n--- step {step} ---\n")
                f.write(f"joint_positions: {snap['observation.state'].tolist()}\n")
                f.write(f"action: {snap['action'].tolist()}\n")
                f.write(
                    f"camera_top  eye={snap['camera_top_eye']} target={snap['camera_top_target']}\n"
                )
                f.write(
                    f"camera_wrist eye={[round(x, 4) for x in snap['camera_wrist_eye']]} "
                    f"target={[round(x, 4) for x in snap['camera_wrist_target']]}\n"
                )
        print(f"  汇总已写入: {summary_path}")
        print("✓ Episode 数据采集测试完成")
        env.close()
    except Exception as e:
        env.close()
        raise RuntimeError(f"Episode 数据采集测试失败: {e}") from e


# 红蓝方块任务映射（用于语言遵从数据采集）
TASKS = [
    ("red", "Pick up the red cube and place it in the box."),
    ("blue", "Pick up the blue cube and place it in the box."),
]


def test_lerobot_dataset_episode_collection(
    num_episodes: int = 10,
    output_dir: Path | None = None,
    use_gui: bool = True,
    collect_frequency_hz: float = 10.0,
    image_size: int = 224,
    seed: int = 42,
    steps_per_phase: int = 80,
    task_description: str = "Pick the red cube and place it in the box.",
    seed_strategy: str = "fixed",
    task_mode: str = "red_only",
    red_blue_ratio: float = 0.5,
    repo_id: str = "pick_red_lerobot_dataset",
):
    """
    在带 GUI 的仿真下运行多个 episode（抓取红色/蓝色方块），按固定频率采集每帧数据，
    通过 LeRobotDataset.create / add_frame / save_episode / save 写入 LeRobot 格式。

    字段对应：
    - 双相机：observation.images.top_image, observation.images.wrist_image
    - 关节状态：observation.state
    - 动作：action
    - timestamp, frame_index, episode_index, index, task_index 由 LeRobotDataset 维护

    参数：
    - seed_strategy: "fixed" 共用同一 seed，"varying" 每 episode 使用 base_seed+ep 以扩大泛化
    - task_mode: "red_only" | "blue_only" | "red_blue_alternate" | "red_blue_ratio"
    - red_blue_ratio: 当 task_mode="red_blue_ratio" 时，red 任务占比 (0~1)

    运行示例:
      python test/test_episode_data_collection.py --lerobot
      python test/test_episode_data_collection.py --lerobot --task-mode red_blue_alternate --seed-strategy varying
    """
    if LeRobotDataset is None:
        raise ImportError(
            "LeRobotDataset 未找到。请安装: pip install lerobot\n"
            "并确认存在 lerobot.common.datasets.lerobot_dataset 或 lerobot.datasets.lerobot_dataset"
        )

    if output_dir is None:
        output_dir = project_root / "test_output" / "lerobot_pick_red_dataset"
    root = Path(output_dir)
    # 不在此处 mkdir：LeRobotDataset.create() 会自行创建 root，且要求目录不存在 (exist_ok=False)

    print("=" * 60)
    print("测试: LeRobot 格式多 Episode 数据采集 (LeRobotDataset.create / add_frame / save)")
    print("=" * 60)
    print(f"  输出目录: {root}")
    print(f"  repo_id: {repo_id}")
    print(f"  Episode 数: {num_episodes}")
    print(f"  seed_strategy: {seed_strategy}")
    print(f"  task_mode: {task_mode}")
    if task_mode == "red_blue_ratio":
        print(f"  red_blue_ratio: {red_blue_ratio}")
    print(f"  GUI: {use_gui}")
    print(f"  采集频率: {collect_frequency_hz} Hz")
    print(f"  图像尺寸: {image_size}x{image_size}")

    control_hz = 40.0
    collect_interval = max(1, int(control_hz / collect_frequency_hz))
    fps = collect_frequency_hz

    # ---------- 1. 创建数据集（明确 features + fps，便于 VLA/π0 训练） ----------
    # dataset = LeRobotDataset.create(
    #     repo_id=repo_id,
    #     fps=int(fps),
    #     features={
    #         "observation.state": {"dtype": "float32", "shape": (9,)},
    #         "observation.images.top_image": {"dtype": "image", "shape": (image_size, image_size, 3)},
    #         "observation.images.wrist_image": {"dtype": "image", "shape": (image_size, image_size, 3)},
    #         "action": {"dtype": "float32", "shape": (4,)},
    #     },
    #     root=root,
    # )
    dataset = LeRobotDataset.create(
    repo_id=repo_id,
    fps=int(fps),
    features={
        "observation.state": {"dtype": "float32", "shape": (9,)},
        "observation.images.top_image": {
            "dtype": "video",                 # 修改点 1: image -> video
            "shape": (3, image_size, image_size), # 修改点 2: 变为 (C, H, W)
            "names": ["color"],
            "video_codec": "libx264",         # 添加编码器配置
            "fps": int(fps),                  # 视频帧率需与数据集一致
        },
        "observation.images.wrist_image": {
            "dtype": "video",                 # 修改点 1
            "shape": (3, image_size, image_size), # 修改点 2
            "names": ["color"],
            "video_codec": "libx264",
            "fps": int(fps),
        },
        "action": {"dtype": "float32", "shape": (4,)},
    },
    use_videos=True, # 修改点 3: 必须显式开启视频支持
    root=root,
)

    env = PickPlaceEnv(render=True, use_gui=use_gui, seed=seed)
    total_frames = 0

    try:
        for ep in range(num_episodes):
            # 抓取泛化：每个 episode 使用不同 seed 以增加方块位置多样性
            if seed_strategy == "varying":
                np.random.seed(seed + ep)

            obs = env.reset()
            collected: list[dict] = []
            step_count = [0]

            # 语言遵从：根据 task_mode 选择本 episode 抓取红/蓝方块及任务描述
            if task_mode == "red_only":
                color, ep_task_desc = TASKS[0]
            elif task_mode == "blue_only":
                color, ep_task_desc = TASKS[1]
            elif task_mode == "red_blue_alternate":
                color, ep_task_desc = TASKS[ep % 2]
            elif task_mode == "red_blue_ratio":
                n_red = int(num_episodes * red_blue_ratio)
                if ep < n_red:
                    color, ep_task_desc = TASKS[0]
                else:
                    color, ep_task_desc = TASKS[1]
            else:
                color, ep_task_desc = TASKS[0]

            def on_before_step(action: np.ndarray) -> None:
                step_count[0] += 1
                if step_count[0] % collect_interval == 0:
                    snap = env.collect_snapshot(
                        action,
                        image_width=image_size,
                        image_height=image_size,
                    )
                    collected.append(snap)

            success, reward, done = env.execute_pick_place(
                color,
                steps_per_phase=steps_per_phase,
                on_before_step=on_before_step,
                use_real_time=False,
            )

            # ---------- 2. 逐帧 add_frame（frame=全量特征 dict，task=任务描述）；无 new_episode，由 save_episode 后自动重置 buffer ----------
            for fi, snap in enumerate(collected):
                # t = total_frames / fps
                # frame = {
                #     "observation.state": snap["observation.state"].astype(np.float32),
                #     "observation.images.top_image": snap["observation.images.top"],
                #     "observation.images.wrist_image": snap["observation.images.wrist"],
                #     "action": snap["action"].astype(np.float32),
                # }
                # dataset.add_frame(frame=frame, task=ep_task_desc, timestamp=t)
                t = float(total_frames) / fps
    
                # 确保图片从 (H, W, C) 转换为 (C, H, W)
                # 如果 snap 中的数据已经是 (3, H, W) 则不需要这一步
                top_img = snap["observation.images.top"]
                if top_img.shape[-1] == 3: # 如果最后一维是通道
                    top_img = top_img.transpose(2, 0, 1)
                    
                wrist_img = snap["observation.images.wrist"]
                if wrist_img.shape[-1] == 3:
                    wrist_img = wrist_img.transpose(2, 0, 1)
            
                frame = {
                    "observation.state": snap["observation.state"].astype(np.float32),
                    "observation.images.top_image": top_img,
                    "observation.images.wrist_image": wrist_img,
                    "action": snap["action"].astype(np.float32),
                }
                total_frames += 1

            # ---------- 3. 保存当前 episode（并自动为下一 episode 创建新 buffer） ----------
            dataset.save_episode()

            print(f"  Episode {ep + 1}/{num_episodes}: color={color}, {len(collected)} 帧, success={success}, reward={reward}")

        print(f"\n  总帧数: {total_frames}")
        print(f"  已写入 LeRobot 格式: {root}")
        print("✓ LeRobot 数据集采集测试完成")
        env.close()
    except Exception as e:
        env.close()
        raise RuntimeError(f"LeRobot 数据集采集测试失败: {e}") from e


def _parse_lerobot_args():
    """解析 LeRobot 采集命令行参数"""
    import argparse
    ap = argparse.ArgumentParser(description="LeRobot 格式多 episode 数据采集")
    ap.add_argument("--lerobot", action="store_true", help="使用 LeRobot 格式采集")
    ap.add_argument("--no-gui", action="store_true", help="禁用 GUI（DIRECT 仿真）")
    ap.add_argument("--num-episodes", type=int, default=10, help="Episode 数量")
    ap.add_argument("--task-mode", type=str, default="red_only",
                    choices=["red_only", "blue_only", "red_blue_alternate", "red_blue_ratio"],
                    help="任务模式：red_only/blue_only/red_blue_alternate/red_blue_ratio")
    ap.add_argument("--red-blue-ratio", type=float, default=0.5,
                    help="当 task_mode=red_blue_ratio 时 red 任务占比 (0~1)")
    ap.add_argument("--seed-strategy", type=str, default="fixed",
                    choices=["fixed", "varying"],
                    help="seed 策略：fixed 共用同一 seed，varying 每 episode 使用 base_seed+ep")
    ap.add_argument("--repo-id", type=str, default="pick_red_lerobot_dataset",
                    help="LeRobot 数据集 repo_id（红蓝任务建议 pick_red_blue_lerobot_dataset）")
    ap.add_argument("--output-dir", type=str, default=None, help="输出目录")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_lerobot_args()
    if args.lerobot:
        use_gui = not args.no_gui
        out_dir = Path(args.output_dir) if args.output_dir else None
        if out_dir is None and args.task_mode in ("red_blue_alternate", "red_blue_ratio"):
            out_dir = project_root / "test_output" / "lerobot_pick_red_blue_dataset"
        test_lerobot_dataset_episode_collection(
            num_episodes=args.num_episodes,
            output_dir=out_dir,
            use_gui=use_gui,
            collect_frequency_hz=10.0,
            image_size=224,
            task_mode=args.task_mode,
            red_blue_ratio=args.red_blue_ratio,
            seed_strategy=args.seed_strategy,
            repo_id=args.repo_id,
        )
        sys.exit(0)

    use_gui = "--gui" in sys.argv
    test_episode_data_collection(
        use_gui=use_gui,
        collect_frequency_hz=10.0,
        image_size=224,
    )
