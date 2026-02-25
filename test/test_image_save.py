"""
测试仿真环境图片保存功能。
调用 PickPlaceEnv，输出一帧第三视角与腕部相机下的环境照片。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from simulator.pick_place_env import PickPlaceEnv


def capture_full_scene_direct(
    env: PickPlaceEnv,
    width: int = 640,
    height: int = 640,
) -> np.ndarray:
    """
    第三视角相机：委托到 env._capture_fixed_camera，与 collect_snapshot 使用同一套渲染逻辑。
    """
    return env._capture_fixed_camera(width, height)


def capture_wrist_direct(
    env: PickPlaceEnv,
    width: int = 640,
    height: int = 640,
) -> np.ndarray:
    """
    腕部/夹爪相机：委托到 env._capture_wrist_camera，与 collect_snapshot 使用同一套渲染逻辑。
    """
    return env._capture_wrist_camera(width, height)


def save_image(arr: np.ndarray, path: Path) -> None:
    """将 RGB 数组保存为图片（优先 cv2，其次 PIL，否则 npy）"""
    try:
        import cv2
        bgr = np.asarray(arr[:, :, ::-1])
        cv2.imwrite(str(path), bgr)
    except ImportError:
        try:
            from PIL import Image
            Image.fromarray(arr).save(path)
        except ImportError:
            np.save(path.with_suffix(".npy"), arr)


def test_image_save(
    use_gui: bool = False,
    output_dir: Path | None = None,
    image_size: int = 640,
    include_wrist: bool = True,
):
    """
    测试仿真环境图片保存功能。
    创建环境 -> reset -> 渲染第三视角与腕部图像 -> 保存。
    """
    if output_dir is None:
        output_dir = project_root / "test_output" / "image_save_test"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("测试: 仿真环境图片保存")
    print("=" * 60)
    print(f"  输出目录: {output_dir}")
    print(f"  图像尺寸: {image_size}x{image_size}")
    print(f"  GUI: {use_gui}")

    env = PickPlaceEnv(render=True, use_gui=use_gui, seed=42)
    try:
        obs = env.reset()

        # 与 collect_snapshot 一致：使用 env 内置的 _capture_fixed_camera / _capture_wrist_camera
        img_fixed = env._capture_fixed_camera(image_size, image_size)
        path_fixed = output_dir / "env_frame_fixed.png"
        save_image(img_fixed, path_fixed)
        print(f"  ✓ 第三视角已保存: {path_fixed}")

        if include_wrist:
            img_wrist = env._capture_wrist_camera(image_size, image_size)
            path_wrist = output_dir / "env_frame_wrist.png"
            save_image(img_wrist, path_wrist)
            print(f"  ✓ 腕部视角已保存: {path_wrist}")

        print("\n✓ 图片保存测试完成")
        env.close()
    except Exception as e:
        env.close()
        raise RuntimeError(f"图片保存测试失败: {e}") from e


if __name__ == "__main__":
    use_gui = "--gui" in sys.argv
    output = None
    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i + 1 < len(sys.argv):
            output = Path(sys.argv[i + 1])

    include_wrist = "--no-wrist" not in sys.argv

    test_image_save(
        use_gui=use_gui,
        output_dir=output,
        image_size=640,
        include_wrist=include_wrist,
    )
