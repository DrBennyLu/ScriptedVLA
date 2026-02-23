"""
测试仿真环境图片保存功能。
调用 PickPlaceEnv，输出一帧第三视角与腕部相机下的环境照片。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pybullet as p
from simulator.pick_place_env import PickPlaceEnv


def capture_full_scene_direct(
    env: PickPlaceEnv,
    width: int = 640,
    height: int = 640,
) -> np.ndarray:
    """
    第三视角相机：与 pick_place_env 的 resetDebugVisualizerCamera 保持一致。
    GUI 模式：reset 后取 debug 的 view 和 projection；DIRECT 模式：computeViewMatrix。
    """
    cid = env._client_id
    table_height = env.table_height

    cam_target = [0.0, 0.0, table_height + 0.1]
    cam_distance = 0.8
    cam_yaw = 180.0
    cam_pitch = -35.0
    fov_deg = 60.0

    if env.use_gui:
        p.resetDebugVisualizerCamera(
            cameraDistance=cam_distance,
            cameraYaw=cam_yaw,
            cameraPitch=cam_pitch,
            cameraTargetPosition=cam_target,
            physicsClientId=cid,
        )
        p.stepSimulation(physicsClientId=cid)
        cam = p.getDebugVisualizerCamera(physicsClientId=cid)
        view_matrix = cam[2]
        projection_matrix = cam[3]
        cap_width = int(cam[0])
        cap_height = int(cam[1])
    else:
        yaw_rad = np.radians(cam_yaw)
        pitch_rad = np.radians(cam_pitch)
        dx = cam_distance * np.cos(pitch_rad) * np.cos(yaw_rad)
        dy = cam_distance * np.cos(pitch_rad) * np.sin(yaw_rad)
        dz = -cam_distance * np.sin(pitch_rad)
        eye = [
            cam_target[0] + dx,
            cam_target[1] + dy,
            cam_target[2] + dz,
        ]
        view_matrix = p.computeViewMatrix(eye, cam_target, [0, 0, 1])
        aspect = width / float(height)
        projection_matrix = p.computeProjectionMatrixFOV(
            np.radians(fov_deg), aspect, 0.02, 10.0
        )
        cap_width = width
        cap_height = height

    p.stepSimulation(physicsClientId=cid)

    renderer = (
        p.ER_BULLET_HARDWARE_OPENGL if env.use_gui else p.ER_TINY_RENDERER
    )
    result = p.getCameraImage(
        cap_width,
        cap_height,
        view_matrix,
        projection_matrix,
        shadow=False,
        renderer=renderer,
        physicsClientId=cid,
    )

    w_actual, h_actual = int(result[0]), int(result[1])
    rgb = np.array(result[2], dtype=np.uint8)
    rgb = rgb.reshape((h_actual, w_actual, 4))[:, :, :3]

    if env.use_gui and (w_actual != width or h_actual != height):
        try:
            import cv2
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image
            rgb = np.array(Image.fromarray(rgb).resize((width, height)))

    return rgb


def capture_wrist_direct(
    env: PickPlaceEnv,
    width: int = 640,
    height: int = 640,
) -> np.ndarray:
    """
    腕部/夹爪相机：与手指朝向一致，位于夹爪横向约 0.04m（由 env._get_camera_view_params 决定）。
    腕部相机使用更大视场角以看清被抓物体。
    """
    cid = env._client_id
    fov_deg = 105.0  # 视野调大，便于清晰看见被抓物体

    eye, target, up = env._get_camera_view_params("gripper")
    view_matrix = p.computeViewMatrix(eye, target, up)

    if env.use_gui:
        cam = p.getDebugVisualizerCamera(physicsClientId=cid)
        cap_width = int(cam[0])
        cap_height = int(cam[1])
        aspect = cap_width / float(cap_height)
        projection_matrix = p.computeProjectionMatrixFOV(
            np.radians(fov_deg), aspect, 0.02, 10.0
        )
    else:
        aspect = width / float(height)
        projection_matrix = p.computeProjectionMatrixFOV(
            np.radians(fov_deg), aspect, 0.02, 10.0
        )
        cap_width = width
        cap_height = height

    p.stepSimulation(physicsClientId=cid)

    renderer = (
        p.ER_BULLET_HARDWARE_OPENGL if env.use_gui else p.ER_TINY_RENDERER
    )
    result = p.getCameraImage(
        cap_width,
        cap_height,
        view_matrix,
        projection_matrix,
        shadow=False,
        renderer=renderer,
        physicsClientId=cid,
    )

    w_actual, h_actual = int(result[0]), int(result[1])
    rgb = np.array(result[2], dtype=np.uint8)
    rgb = rgb.reshape((h_actual, w_actual, 4))[:, :, :3]

    if env.use_gui and (w_actual != width or h_actual != height):
        try:
            import cv2
            rgb = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            from PIL import Image
            rgb = np.array(Image.fromarray(rgb).resize((width, height)))

    return rgb


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

        img_fixed = capture_full_scene_direct(env, width=image_size, height=image_size)
        path_fixed = output_dir / "env_frame_fixed.png"
        save_image(img_fixed, path_fixed)
        print(f"  ✓ 第三视角已保存: {path_fixed}")

        if include_wrist:
            img_wrist = capture_wrist_direct(env, width=image_size, height=image_size)
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
