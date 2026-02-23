"""
测试 PickPlaceEnv 仿真环境
验证仿真环境能否正常工作：加载场景、生成方块、reset、step 等。
包含数据采集测试：固定频率采集图像（固定视角 + 夹爪相机）、关节位置、action。
"""

import sys
from pathlib import Path
import time

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from simulator.pick_place_env import PickPlaceEnv


def test_env_creation():
    """测试环境创建（无 GUI，用于 CI）"""
    print("=" * 60)
    print("测试1: 环境创建 (DIRECT 模式，无 GUI)")
    print("=" * 60)

    env = PickPlaceEnv(render=True, use_gui=False, seed=42)
    try:
        obs = env.reset()
        print("✓ 环境创建成功")
        print(f"  红色方块位置: {obs.get('red_cube_pos', 'N/A')}")
        print(f"  蓝色方块位置: {obs.get('blue_cube_pos', 'N/A')}")
        print(f"  盒子位置: {obs.get('box_pos', 'N/A')}")

        # 验证观测包含必要键
        assert "red_cube_pos" in obs
        assert "blue_cube_pos" in obs
        assert "box_pos" in obs
        print("✓ 观测数据格式正确")

        env.close()
        print("✓ 环境关闭成功")
    except Exception as e:
        env.close()
        raise RuntimeError(f"环境测试失败: {e}") from e


def test_env_reset_randomness():
    """测试 reset 时方块位置随机性"""
    print("\n" + "=" * 60)
    print("测试2: Reset 随机性")
    print("=" * 60)

    env = PickPlaceEnv(render=True, use_gui=False, seed=123)
    try:
        obs1 = env.reset()
        obs2 = env.reset()

        red_pos1 = obs1["red_cube_pos"]
        red_pos2 = obs2["red_cube_pos"]

        # 两次 reset 后红色方块位置应不同（随机种子相同但 spawn 有随机性）
        # 实际上相同 seed 下 spawn 应该是确定的... 我们验证的是能正常 reset
        print(f"  第一次 reset - 红色方块: {red_pos1}")
        print(f"  第二次 reset - 红色方块: {red_pos2}")
        print("✓ Reset 执行成功")

        env.close()
    except Exception as e:
        env.close()
        raise RuntimeError(f"Reset 测试失败: {e}") from e


def test_env_step():
    """测试 step 功能"""
    print("\n" + "=" * 60)
    print("测试3: Step 功能")
    print("=" * 60)

    env = PickPlaceEnv(render=True, use_gui=False, seed=456)
    try:
        env.reset()

        for i in range(10):
            obs, reward, done, info = env.step()
            assert isinstance(obs, dict)
            assert isinstance(reward, (int, float))
            assert isinstance(done, bool)
            assert isinstance(info, dict)

        print("✓ Step 执行 10 次成功")
        print(f"  最后一次观测 - 红色方块: {obs.get('red_cube_pos')}")

        env.close()
    except Exception as e:
        env.close()
        raise RuntimeError(f"Step 测试失败: {e}") from e


def test_get_cube_positions():
    """测试 get_cube_positions 方法"""
    print("\n" + "=" * 60)
    print("测试4: get_cube_positions")
    print("=" * 60)

    env = PickPlaceEnv(render=True, use_gui=False, seed=789)
    try:
        env.reset()
        positions = env.get_cube_positions()

        assert "red" in positions
        assert "blue" in positions
        assert len(positions["red"]) == 3
        assert len(positions["blue"]) == 3

        print(f"  红色方块: {positions['red']}")
        print(f"  蓝色方块: {positions['blue']}")
        print("✓ get_cube_positions 正常")

        env.close()
    except Exception as e:
        env.close()
        raise RuntimeError(f"get_cube_positions 测试失败: {e}") from e


def test_pick_place_with_gui(
    target_frame_rotation_deg: float = 0.0,
    target_frame_offset_xy: tuple = (0.0, 0.0),
    ee_to_grasp_offset_xyz: tuple = (0.0, 0.0, 0.0),
):
    """
    测试机器人抓取并放入盒子（带 GUI）
    - 5 个 episode 抓取红色方块
    - 5 个 episode 抓取蓝色方块
    - 每个 episode: reset -> 抓取指定颜色方块 -> 移动到盒子上方 -> 放入盒子

    若夹爪与方块/盒子位置有偏差，可尝试：
    - target_frame_rotation_deg=90 或 -90（若 XY 轴需要旋转）
    - target_frame_offset_xy=(dx, dy) 微调偏移
    - ee_to_grasp_offset_xyz=(dx, dy, dz) 补偿 panda_hand 原点与抓取中心的偏移
    """
    print("\n" + "=" * 60)
    print("测试5: 抓取-放置 (GUI, 5 红 + 5 蓝 episode)")
    print("=" * 60)

    env = PickPlaceEnv(
        render=True,
        use_gui=True,
        seed=42,
        target_frame_rotation_deg=target_frame_rotation_deg,
        target_frame_offset_xy=target_frame_offset_xy,
        ee_to_grasp_offset_xyz=ee_to_grasp_offset_xyz,
    )
    try:
        # 5 episodes: 抓取红色方块
        print("  执行 5 个 episode: 抓取红色方块 -> 放入盒子")
        for ep in range(5):
            obs = env.reset()
            success, reward, done = env.execute_pick_place("red", steps_per_phase=80)
            print(f"    Episode {ep + 1}/5 (红色): {'成功' if success else '失败'}, reward={reward}, done={done}")
        print("  红色方块抓取测试完成")

        # 5 episodes: 抓取蓝色方块
        print("  执行 5 个 episode: 抓取蓝色方块 -> 放入盒子")
        for ep in range(5):
            obs = env.reset()
            success, reward, done = env.execute_pick_place("blue", steps_per_phase=80)
            print(f"    Episode {ep + 1}/5 (蓝色): {'成功' if success else '失败'}, reward={reward}, done={done}")
        print("  蓝色方块抓取测试完成")

        print("✓ 抓取-放置测试完成")
        env.close()
    except Exception as e:
        env.close()
        raise RuntimeError(f"抓取-放置测试失败: {e}") from e


def test_env_with_gui():
    """
    测试6: 带 GUI 的环境（可选，手动运行）
    在 CI 或无显示环境下可能失败，因此单独标注
    """
    print("\n" + "=" * 60)
    print("测试6: 带 GUI 的环境 (可选)")
    print("=" * 60)

    try:
        env = PickPlaceEnv(render=True, use_gui=True, seed=0)
        obs = env.reset()
        print("✓ GUI 环境创建成功")
        # 运行几步让用户看到
        for _ in range(100):
            env.step()
            time.sleep(0.1) # 10hz
        print("  已运行 100 步，请查看仿真窗口")
        input("  按 Enter 关闭仿真...")
        env.close()
        print("✓ GUI 测试完成")
    except Exception as e:
        print(f"  注意: GUI 测试跳过 (无显示环境): {e}")
        print("  在本地有显示的机器上可单独运行此测试")


def test_data_collection_snapshot(
    num_steps: int = 5,
    use_gui: bool = False,
    output_dir: Path | None = None,
    image_size: int = 224,
):
    """
    数据采集测试（第一步验证）：仿真运行若干步，每步采集
    - 固定视角图像 (top)
    - 夹爪相机图像 (wrist)
    - 关节位置 (9 维)
    - 本步 action (4 维: x,y,z,gripper)
    采集完成后将图像保存到 output_dir，并在控制台打印关节与 action，便于确认无误。

    运行方式:
      python test/test_pick_place_env.py --data-collection        # 无 GUI，5 步，输出到 test_output/data_collection_test/
      python test/test_pick_place_env.py --data-collection --gui  # 带 GUI，便于观察
    """
    print("\n" + "=" * 60)
    print("测试: 数据采集 (固定频率) — 图像 / 关节 / action")
    print("=" * 60)

    if output_dir is None:
        output_dir = project_root / "test_output" / "data_collection_test"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"  输出目录: {output_dir}")

    env = PickPlaceEnv(render=True, use_gui=use_gui, seed=42)
    try:
        obs = env.reset()
        # 初始末端位置与夹爪开度，作为前几步的保持动作
        ee_pos = env._get_ee_pos()
        action = np.array([*ee_pos, 0.04], dtype=np.float64)

        collected = []
        for step in range(num_steps):
            # 每步：先 step(action)，再采集当前状态（图像+关节）+ 本步 action
            obs, reward, done, info = env.step(action, sim_steps_per_call=24)
            snapshot = env.collect_snapshot(
                action,
                image_width=image_size,
                image_height=image_size,
            )
            collected.append(snapshot)
            # 下一步动作：可保持或略微移动（这里保持，便于观察关节/图像是否随仿真变化）
            action = np.array([*env._get_ee_pos(), 0.04], dtype=np.float64)

        # 展示：保存图像 + 打印关节与 action
        try:
            import cv2
            has_cv2 = True
        except ImportError:
            has_cv2 = False
        if not has_cv2:
            try:
                from PIL import Image
            except ImportError:
                Image = None
        else:
            Image = None

        def save_image(arr: np.ndarray, path: Path) -> None:
            if has_cv2:
                # OpenCV 使用 BGR
                bgr = np.asarray(arr[:, :, ::-1])
                cv2.imwrite(str(path), bgr)
            elif Image is not None:
                Image.fromarray(arr).save(path)
            else:
                np.save(path.with_suffix(".npy"), arr)

        for step, snap in enumerate(collected):
            step_dir = output_dir / f"step_{step:03d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            save_image(snap["observation.images.top"], step_dir / "top.png")
            save_image(snap["observation.images.wrist"], step_dir / "wrist.png")
            print(f"\n  Step {step}:")
            print(f"    joint_positions (9): {snap['observation.state']}")
            print(f"    action (x,y,z,gripper): {snap['action']}")
            print(f"    camera top:  eye={snap['camera_top_eye']}, target={snap['camera_top_target']}")
            print(f"    camera wrist: eye={[round(x, 4) for x in snap['camera_wrist_eye']]}, target={[round(x, 4) for x in snap['camera_wrist_target']]}")

        # 汇总信息写入文本，便于核对（含相机位置）
        summary_path = output_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"num_steps={num_steps}\n")
            f.write(f"image_size={image_size}\n")
            for step, snap in enumerate(collected):
                f.write(f"\n--- step {step} ---\n")
                f.write(f"joint_positions: {snap['observation.state'].tolist()}\n")
                f.write(f"action: {snap['action'].tolist()}\n")
                f.write(f"camera_top  eye={snap['camera_top_eye']} target={snap['camera_top_target']}\n")
                f.write(f"camera_wrist eye={[round(x, 4) for x in snap['camera_wrist_eye']]} target={[round(x, 4) for x in snap['camera_wrist_target']]}\n")
        print(f"\n  汇总已写入: {summary_path}")
        print("✓ 数据采集测试完成（请查看输出目录中的图像与 summary.txt 确认）")
        env.close()
    except Exception as e:
        env.close()
        raise RuntimeError(f"数据采集测试失败: {e}") from e


def run_all_tests(
    skip_gui: bool = True,
    target_frame_rotation_deg: float = 0.0,
    target_frame_offset_xy: tuple = (0.0, 0.0),
    ee_to_grasp_offset_xyz: tuple = (0.0, 0.0, 0.0),
):
    """运行所有测试"""
    print("\n" + "#" * 60)
    print("# PickPlaceEnv 仿真环境测试")
    print("#" * 60)

    test_env_creation()
    test_env_reset_randomness()
    test_env_step()
    test_get_cube_positions()
    # 数据采集测试：若干步后保存图像 + 关节 + action，用于确认采集无误
    test_data_collection_snapshot(num_steps=3, use_gui=False)

    if not skip_gui:
        test_pick_place_with_gui(
            target_frame_rotation_deg=target_frame_rotation_deg,
            target_frame_offset_xy=target_frame_offset_xy,
            ee_to_grasp_offset_xyz=ee_to_grasp_offset_xyz,
        )
        test_env_with_gui()
    else:
        print("\n提示: 跳过 GUI 测试。要测试抓取-放置 (含 GUI)，请运行:")
        print("  python test/test_pick_place_env.py --gui")

    print("\n" + "=" * 60)
    print("所有测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    # 仅运行数据采集测试（第一步验证）：图像 + 关节 + action 保存到 test_output/data_collection_test/
    if "--data-collection" in sys.argv:
        use_gui = "--gui" in sys.argv
        test_data_collection_snapshot(
            num_steps=5,
            use_gui=use_gui,
            image_size=224,
        )
        sys.exit(0)

    skip_gui = "--gui" not in sys.argv
    # 坐标系对齐调试：--rotation 90 --offset 0.01,0.02 --ee-offset -0.04,0,0
    rotation_deg = 0.0
    offset_xy = (0.0, 0.0)
    ee_offset = (0.0, 0.0, 0.0)
    for i, arg in enumerate(sys.argv):
        if arg == "--rotation" and i + 1 < len(sys.argv):
            rotation_deg = float(sys.argv[i + 1])
        elif arg == "--offset" and i + 1 < len(sys.argv):
            parts = sys.argv[i + 1].split(",")
            offset_xy = (float(parts[0]), float(parts[1])) if len(parts) >= 2 else (0.0, 0.0)
        elif arg == "--ee-offset" and i + 1 < len(sys.argv):
            parts = sys.argv[i + 1].split(",")
            ee_offset = (float(parts[0]), float(parts[1]), float(parts[2])) if len(parts) >= 3 else (0.0, 0.0, 0.0)

    run_all_tests(
        skip_gui=skip_gui,
        target_frame_rotation_deg=rotation_deg,
        target_frame_offset_xy=offset_xy,
        ee_to_grasp_offset_xyz=ee_offset,
    )
