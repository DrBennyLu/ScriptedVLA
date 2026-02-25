"""
测试一个 episode 的数据采集与保存：执行完整的抓取红色方块并放入盒子流程，
在流程中按固定频率（默认 10Hz）记录双相机图像、机器人关节状态、末端目标 action，并保存到目录。
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from simulator.pick_place_env import PickPlaceEnv


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


if __name__ == "__main__":
    use_gui = "--gui" in sys.argv
    test_episode_data_collection(
        use_gui=use_gui,
        collect_frequency_hz=10.0,
        image_size=224,
    )
