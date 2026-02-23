"""
PyBullet simulation environment for pick-and-place task.
Robotic arm picks up red or blue cubes from a table and places them in a box.

Environment setup:
- Franka Panda robot mounted ON the table
- Red and blue cubes on the table (positions randomized)
- A box on the table for placing cubes
- Task: Pick up cubes and place them in the box
"""

import time

import numpy as np
import pybullet as p
import pybullet_data


class PickPlaceEnv:
    """
    Pick-and-place simulation environment using PyBullet.
    
    The robot arm picks up red or blue cubes from the table and places them in a box.
    Cube positions are randomized within a specified range on each reset.
    """

    def __init__(
        self,
        render: bool = True,
        use_gui: bool = True,
        table_height: float = 0.5,
        table_half_extents: tuple = (0.7, 0.55, 0.04),
        cube_size: float = 0.015,
        cube_mass: float = 0.02,
        box_position: tuple | None = None,
        cube_spawn_range_x: tuple = (-0.1, 0.1),
        cube_spawn_range_y: tuple = (-0.12, 0.12),
        robot_base_xy: tuple = (-0.45, 0.0),
        robot_base_z_offset: float = 0.0,
        robot_initial_joints: tuple | None = None,
        target_frame_rotation_deg: float = 0.0,
        target_frame_offset_xy: tuple = (0.0, 0.0),
        ee_to_grasp_offset_xyz: tuple = (0.0, 0.0, 0.0),
        robot_base_orientation: tuple | None = None,
        arm_position_gain: float = 0.3,
        arm_velocity_gain: float = 1.0,
        seed: int | None = None,
    ):
        """
        Initialize the pick-and-place environment.

        Args:
            render: Whether to enable rendering.
            use_gui: Whether to show the GUI (only applies when render=True).
            table_height: Height of the table surface (z of table top).
            table_half_extents: (x, y, z) half-extents of the table.
            cube_size: Size of each cube (half-extents).
            cube_mass: Mass of each cube.
            box_position: (x, y, z) position of the box center. If None, auto-computed on table.
            cube_spawn_range_x: (min, max) x range for cube spawning on table.
            cube_spawn_range_y: (min, max) y range for cube spawning on table.
            robot_base_xy: (x, y) position of robot base on table (z = table_height).
            robot_base_z_offset: Extra z offset for robot base (if URDF origin != bottom).
            robot_initial_joints: (j0..j6, finger1, finger2) 共9个关节，用于手工调整初始姿态。
            target_frame_rotation_deg: 目标坐标系旋转（绕 Z 轴，度）。用于对齐机器人与方块/盒子坐标系。
            target_frame_offset_xy: (dx, dy) 目标偏移，用于微调夹爪与方块/盒子的对齐。
            ee_to_grasp_offset_xyz: (dx, dy, dz) 补偿 panda_hand 链接原点与视觉抓取中心的偏移。
                若夹爪在 XY 上相对方块有固定偏移，可据此微调。例如夹爪在方块右侧 4cm 则用 (-0.04, 0, 0)。
            robot_base_orientation: 机器人 base 四元数 [x,y,z,w]。若机器人朝向与预期不符可设置，如 [0,0,1,0] 表示绕 Z 转 180°。
            arm_position_gain: 手臂位置控制 positionGain，越大收敛越快（原 0.05 偏小）。
            arm_velocity_gain: 手臂位置控制 velocityGain。
            seed: Random seed for reproducibility.
        """
        self.render = render
        self.use_gui = use_gui
        self.table_height = table_height
        self.table_half_extents = np.array(table_half_extents)
        self.cube_size = cube_size
        self.cube_mass = cube_mass
        self.cube_spawn_range_x = cube_spawn_range_x
        self.cube_spawn_range_y = cube_spawn_range_y
        self.robot_base_xy = np.array(robot_base_xy)
        self.robot_base_z_offset = robot_base_z_offset
        # Box (open container) on table: center z = table_top + half_height
        if box_position is None:
            self.box_position = np.array(
                [0.0, 0.22, table_height + 0.04]
            )  # Closer to robot, within workspace; z = table_top + half container height
        else:
            self.box_position = np.array(box_position)

        if seed is not None:
            np.random.seed(seed)

        self._client_id: int | None = None
        self._robot_id: int | None = None
        self._table_id: int | None = None
        self._box_id: int | None = None
        self._box_body_ids: list[int] = []
        self._red_cube_id: int | None = None
        self._blue_cube_id: int | None = None

        # Colors (RGBA)
        self._red_color = [1.0, 0.0, 0.0, 1.0]
        self._blue_color = [0.0, 0.0, 1.0, 1.0]

        # 机器人初始关节角，手工调整姿态
        # [j0, j1, j2, j3, j4, j5, j6, finger1, finger2]
        # 夹爪: 0.04=开, 0.01=闭
        # 参考姿态：0.98, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32, 0.04, 0.04,
        if robot_initial_joints is None:
            self._robot_initial_joints = [
                -0.60, 0.458, 0.31, -2.24, -0.30, 2.66, 2.32,
                0.04, 0.04,
            ]
        else:
            self._robot_initial_joints = list(robot_initial_joints)

        self._target_frame_rotation_deg = float(target_frame_rotation_deg)
        self._target_frame_offset_xy = np.array(target_frame_offset_xy, dtype=np.float64)
        self._ee_to_grasp_offset_xyz = np.array(ee_to_grasp_offset_xyz, dtype=np.float64)
        self._robot_base_orientation = robot_base_orientation
        self._arm_position_gain = float(arm_position_gain)
        self._arm_velocity_gain = float(arm_velocity_gain)

    def reset(self) -> dict:
        """
        Reset the environment: new cube positions, reset robot.

        Returns:
            dict: Observation containing state info (for future use).
        """
        if self._client_id is None:
            self._connect()

        self._remove_cubes()
        self._spawn_cubes()
        self._reset_robot()
        self._step_simulation(steps=50)

        return self._get_observation()

    def _connect(self) -> None:
        """Connect to PyBullet and set up the simulation."""
        if self.render and self.use_gui:
            self._client_id = p.connect(p.GUI)
        else:
            self._client_id = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self._load_scene()

    def _load_scene(self) -> None:
        """Load table, robot, and box."""
        cid = self._client_id

        # Load plane (ground)
        p.loadURDF(
            "plane.urdf",
            [0, 0, 0],
            useFixedBase=True,
            physicsClientId=cid,
        )

        # Load table (1.4m x 1.1m surface, 8cm thick; robot/cubes/box all on top)
        th = self.table_half_extents  # [x, y, z] half-extents
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=list(th),
            rgbaColor=[0.6, 0.4, 0.2, 1.0],
        )
        table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(th))
        table_pos = [0, 0, self.table_height - th[2]]
        self._table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=table_pos,
            physicsClientId=cid,
        )

        # Load robot ON the table (base bottom at table_height)
        rx, ry = self.robot_base_xy[0], self.robot_base_xy[1]
        robot_base_z = self.table_height + self.robot_base_z_offset
        robot_base_pos = [rx, ry, robot_base_z]
        load_kw = dict(
            basePosition=robot_base_pos,
            useFixedBase=True,
            physicsClientId=cid,
        )
        if self._robot_base_orientation is not None:
            load_kw["baseOrientation"] = list(self._robot_base_orientation)
        self._robot_id = p.loadURDF(
            "franka_panda/panda.urdf",
            **load_kw,
        )
        # Franka Panda: ee link 9 (panda_hand), arm 0-6, finger 9-10
        self._ee_link = 8
        self._num_arm_joints = 7
        self._finger_joints = [9, 10]
        self._ik_joint_limits = (
            [-2.97, -1.83, -2.97, -3.14, -2.97, -0.09, -2.97],
            [2.97, 1.83, 2.97, 0.0, 2.97, 3.82, 2.97],
        )

        # Create open container (bowl-like): bottom + 4 walls, open top
        self._box_id = self._create_open_container(cid)

        # Set camera view (look at table center)
        # 与 _get_camera_view_params / render_camera_image 使用相同参数
        self._camera_distance = 0.8
        self._camera_yaw_deg = 180
        self._camera_pitch_deg = -35
        self._camera_target = [0.0, 0, self.table_height + 0.1]
        if self.render and self.use_gui:
            p.resetDebugVisualizerCamera(
                cameraDistance=self._camera_distance,
                cameraYaw=self._camera_yaw_deg,
                cameraPitch=self._camera_pitch_deg,
                cameraTargetPosition=self._camera_target,
                physicsClientId=cid,
            )

    def _solve_ik(
        self,
        target_pos: np.ndarray,
        target_orn: np.ndarray | None = None,
        rest_poses: list[float] | None = None,
        max_iter: int = 80,
    ) -> list[float]:
        """Solve IK，target_orn 为 None 时保持当前末端姿态."""
        cid = self._client_id
        # if target_orn is None:
        #     ls = p.getLinkState(
        #         self._robot_id, self._ee_link, physicsClientId=cid
        #     )
        #     target_orn = ls[5]
        target_orn = p.getQuaternionFromEuler([3.1415, 0, 0])
        if rest_poses is None:
            rest_poses = self._robot_initial_joints[:7]
        ll, ul = self._ik_joint_limits
        jr = [ul[i] - ll[i] for i in range(self._num_arm_joints)]
        for _ in range(max_iter):
            joint_poses = p.calculateInverseKinematics(
                self._robot_id,
                self._ee_link,
                target_pos.tolist(),
                target_orn,
                lowerLimits=ll,
                upperLimits=ul,
                jointRanges=jr,
                restPoses=rest_poses,
                physicsClientId=cid,
            )
            for i in range(min(self._num_arm_joints, len(joint_poses))):
                p.resetJointState(
                    self._robot_id, i, joint_poses[i],
                    targetVelocity=0, physicsClientId=cid
                )
            ls = p.getLinkState(
                self._robot_id, self._ee_link, physicsClientId=cid
            )
            ee_pos = np.array(ls[4])
            if np.linalg.norm(ee_pos - target_pos) < 0.002:
                break
        return joint_poses[: self._num_arm_joints]

    def _move_ee_to(
        self,
        target_pos: np.ndarray,
        steps: int = 60,
        use_control: bool = True,
        step_delay: float = 0.0,
        gripper_width: float | None = None,
        num_waypoints: int = 15,
    ) -> None:
        """Move end effector to target with Cartesian trajectory interpolation."""
        cid = self._client_id
        ls = p.getLinkState(
            self._robot_id, self._ee_link, physicsClientId=cid
        )
        start_pos = np.array(ls[4])
        # 当前关节角作为 rest，保证轨迹连续、避免配置跳变
        rest = [
            p.getJointState(self._robot_id, i, physicsClientId=cid)[0]
            for i in range(self._num_arm_joints)
        ]
        steps_per_waypoint = max(1, steps // num_waypoints)
        for k in range(1, num_waypoints + 1):
            t = k / num_waypoints
            waypoint = start_pos + t * (target_pos - start_pos)
            joint_poses = self._solve_ik(waypoint, rest_poses=rest)
            rest = joint_poses[: self._num_arm_joints]
            for _ in range(steps_per_waypoint):
                if use_control:
                    for i in range(min(self._num_arm_joints, len(joint_poses))):
                        p.setJointMotorControl2(
                            self._robot_id, i, p.POSITION_CONTROL,
                            targetPosition=joint_poses[i], targetVelocity=0,
                            force=300, positionGain=self._arm_position_gain, velocityGain=self._arm_velocity_gain,
                            physicsClientId=cid,
                        )
                    if gripper_width is not None:
                        self._set_gripper(gripper_width)
                self._step_simulation(1)
                if step_delay > 0:
                    time.sleep(step_delay)

    def _set_gripper(self, width: float) -> None:
        """Set gripper width: 0.04=open, 0=closed. Uses position control."""
        cid = self._client_id
        for j in self._finger_joints:
            p.setJointMotorControl2(
                self._robot_id, j, p.POSITION_CONTROL,
                targetPosition=width, targetVelocity=0,
                force=20, positionGain=0.1, velocityGain=0.5,
                physicsClientId=cid,
                maxVelocity=0.05
            )

    def _open_gripper(self) -> None:
        """Open gripper (width 0.04)."""
        self._set_gripper(0.04)

    def _close_gripper(self) -> None:
        """Close gripper to grasp (width ~0.01 for 3cm cube)."""
        self._set_gripper(0.01)

    def _transform_world_to_target(self, pos: np.ndarray) -> np.ndarray:
        """
        将世界坐标下的目标位姿变换为机器人 IK 使用的目标位姿。
        用于对齐机器人与方块/盒子的坐标系（旋转 + 偏移）。
        """
        x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
        # angle = np.radians(self._target_frame_rotation_deg)
        # if abs(angle) > 1e-6:
        #     c, s = np.cos(angle), np.sin(angle)
        #     x, y = x * c - y * s, x * s + y * c
        # x += self._target_frame_offset_xy[0]
        # y += self._target_frame_offset_xy[1]
        return np.array([x, y, z])

    def _is_cube_in_box(self, cube_id: int) -> bool:
        """判断方块是否在盒子内."""
        pos, _ = p.getBasePositionAndOrientation(
            cube_id, physicsClientId=self._client_id
        )
        cx, cy, cz = self.box_position[0], self.box_position[1], self.box_position[2]
        # 盒子内腔约 10cm x 10cm x 8cm
        margin = 0.02
        return (
            cx - 0.05 + margin <= pos[0] <= cx + 0.05 - margin
            and cy - 0.05 + margin <= pos[1] <= cy + 0.05 - margin
            and cz - 0.05 <= pos[2] <= cz + 0.05
        )

    def execute_pick_place(
        self,
        cube_color: str,
        steps_per_phase: int = 80,
        step_delay: float = 0.012,
        sim_steps_per_call: int = 1,
        position_tolerance: float = 0.005,
        max_wait_steps: int = 500,
        control_frequency: float = 40.0,
    ) -> tuple[bool, float, bool]:
        """
        执行抓取-放置：方块上方 -> 垂直下降 -> 夹取 -> 垂直上抬 -> 平移到盒子上方 -> 松手.
        全部通过 step(action) 控制，不直接调用 _move_ee_to 或 _step_simulation。
        调用前需先执行 reset()，由 reset 负责机器人重置。
        position_tolerance: 末端到位误差阈值（米），达到后才进入下一步。默认 5mm。
        max_wait_steps: 等待到位的最大步数，超时后强制进入下一步。
        返回 (success, reward, done): 方块入盒则 reward=1, done=True.
        """
        class SyncTimer:
            def __init__(self, frequency):
                self.dt = 1.0 / frequency
                self.last_time = time.perf_counter()

            def wait(self):
                current_time = time.perf_counter()
                elapsed = current_time - self.last_time
                sleep_time = self.dt - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)

                self.last_time = time.perf_counter()

        timer = SyncTimer(control_frequency)
        cube_id = self._red_cube_id if cube_color == "red" else self._blue_cube_id
        if cube_id is None:
            return False, 0.0, False

        cid = self._client_id
        pos, _ = p.getBasePositionAndOrientation(cube_id, physicsClientId=cid)
        cube_pos = np.array(pos)
        box_center = self.box_position.copy()
        safe_height = 0.25
        lift_h = 0.30
        # 应用坐标变换，使目标位姿与机器人坐标系对齐
        grasp_pos = self._transform_world_to_target(cube_pos + np.array([0, 0, 0.10]))
        above_cube_safe = self._transform_world_to_target(cube_pos + np.array([0, 0, safe_height]))
        lift_pos = self._transform_world_to_target(cube_pos + np.array([0, 0, lift_h]))
        above_box = self._transform_world_to_target(
            np.array([box_center[0], box_center[1], cube_pos[2] + lift_h])
        )

        # 打印方块位置与夹爪目标位置，用于调试坐标系对齐
        print(f"[抓取调试] 方块颜色: {cube_color}")
        print(f"  方块位置 (world):     x={cube_pos[0]:.4f}, y={cube_pos[1]:.4f}, z={cube_pos[2]:.4f}")
        print(f"  夹爪目标-方块上方:    x={above_cube_safe[0]:.4f}, y={above_cube_safe[1]:.4f}, z={above_cube_safe[2]:.4f}")
        print(f"  夹爪目标-抓取高度:    x={grasp_pos[0]:.4f}, y={grasp_pos[1]:.4f}, z={grasp_pos[2]:.4f}")
        print(f"  夹爪目标-盒子上方:    x={above_box[0]:.4f}, y={above_box[1]:.4f}, z={above_box[2]:.4f}")
        print(f"  盒子中心 (world):     x={box_center[0]:.4f}, y={box_center[1]:.4f}, z={box_center[2]:.4f}")

        def _interpolate_and_step(
            start: np.ndarray,
            target: np.ndarray,
            gripper: float,
            steps: int,
        ) -> dict:
            """沿直线插值并调用 step(action)，到位后等待直到 EE 在 position_tolerance 内或超时。"""
            obs = {}
            action_target = np.array([*target, gripper])
            for k in range(1, steps + 1):
                t = k / steps
                interp = start + t * (target - start)
                action = np.array([*interp, gripper])
                obs, _, _, _ = self.step(
                    action,
                    sim_steps_per_call=sim_steps_per_call,
                    step_delay=step_delay,
                )
                timer.wait()
            max_wait_steps = 50
            for _ in range(max_wait_steps):
                current_ee = self._get_ee_pos()
                error = np.linalg.norm(current_ee - target)

                if error < 0.001:
                    break

                self.step(np.array([*target, gripper]), sim_steps_per_call=sim_steps_per_call)
                timer.wait()
            return obs

        def _stabilize(target_pos, gripper, duration_steps=40):
            """在目标位置悬停一段时间，消除物理惯性和 PID 滞后"""
            action = np.array([*target_pos, gripper])
            for _ in range(duration_steps):
                self.step(action, sim_steps_per_call=sim_steps_per_call)
                timer.wait()

            # 调试：打印稳定后的实际误差
            actual_pos = self._get_ee_pos()
            diff = np.linalg.norm(actual_pos[:2] - target_pos[:2])  # 只看 XY 平面
            if diff > 0.005:  # 如果误差大于 5mm
                print(f"  [警告] 稳定后 XY 偏差仍有: {diff * 1000:.2f}mm (可能需要增加 P gain 或迭代次数)")

        # 假设 reset() 已调用，机器人在初始位姿；用 step 保持位姿一段时间
        init_pos = self._get_ee_pos()
        hold_action = np.array([*init_pos, 0.04])
        # for _ in range(30):
        #     obs, _, _, _ = self.step(
        #         hold_action,
        #         sim_steps_per_call=sim_steps_per_call,
        #         step_delay=step_delay,
        #     )
        
        _stabilize(init_pos, 0.04, duration_steps=30)    

        # 1. 移动到方块正上方（夹爪打开）
        current = self._get_ee_pos()
        _interpolate_and_step(current, above_cube_safe, 0.04, steps_per_phase)
        actual_ee = self._get_ee_pos()
        print(f"  [对比] 到达方块上方后 - 目标: ({above_cube_safe[0]:.4f}, {above_cube_safe[1]:.4f}, {above_cube_safe[2]:.4f}), "
              f"实际 panda_hand 位置: ({actual_ee[0]:.4f}, {actual_ee[1]:.4f}, {actual_ee[2]:.4f})")

        # 2. 垂直下降至抓取高度
        current = self._get_ee_pos()
        _interpolate_and_step(current, grasp_pos, 0.04, steps_per_phase)
        actual_ee = self._get_ee_pos()
        print(f"  [对比] 到达抓取高度后 - 目标: ({grasp_pos[0]:.4f}, {grasp_pos[1]:.4f}, {grasp_pos[2]:.4f}), "
              f"实际 panda_hand 位置: ({actual_ee[0]:.4f}, {actual_ee[1]:.4f}, {actual_ee[2]:.4f})")

        # 3. 闭合夹爪，保持位姿等待夹紧
        hold_pos = self._get_ee_pos()
        hold_action = np.array([*hold_pos, 0.01])
        for _ in range(150):
            obs, _, _, _ = self.step(
                hold_action,
                sim_steps_per_call=sim_steps_per_call,
                step_delay=step_delay,
            )

        # 4. 垂直上抬
        current = self._get_ee_pos()
        _interpolate_and_step(current, lift_pos, 0.01, steps_per_phase)

        # 5. 平移到盒子上方
        current = self._get_ee_pos()
        _interpolate_and_step(current, above_box, 0.01, steps_per_phase)

        # 6. 松开夹爪
        hold_pos = self._get_ee_pos()
        hold_action = np.array([*hold_pos, 0.04])
        # for _ in range(100):
        #     obs, _, _, _ = self.step(
        #         hold_action,
        #         sim_steps_per_call=sim_steps_per_call,
        #         step_delay=step_delay,
        #     )
        for _ in range(60):
            self.step(hold_action, sim_steps_per_call=sim_steps_per_call)
            timer.wait()
        
        in_box = self._is_cube_in_box(cube_id)
        reward = 1.0 if in_box else 0.0
        done = in_box
        return True, reward, done

    def _create_open_container(self, cid: int) -> int:
        """Create an open-top container (bowl-like): bottom + 4 walls, open top."""
        # Container: 10cm x 10cm interior, 8cm tall, 4mm wall thickness
        w, h = 0.05, 0.04  # half width, half height
        t = 0.004  # wall thickness
        cx, cy, cz = self.box_position[0], self.box_position[1], self.box_position[2]
        color = [0.35, 0.28, 0.2, 0.95]

        def _make_box(hx, hy, hz, px, py, pz):
            vs = p.createVisualShape(
                p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=color
            )
            cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
            return p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=cs,
                baseVisualShapeIndex=vs,
                basePosition=[cx + px, cy + py, cz + pz],
                physicsClientId=cid,
            )

        # Bottom plate (on table)
        self._box_body_ids.append(_make_box(w, w, t / 2, 0, 0, -h + t / 2))
        # 4 walls (on top of bottom plate, open top)
        wall_z = t  # walls sit on top of bottom plate
        self._box_body_ids.append(_make_box(w, t / 2, h, 0, -w + t / 2, wall_z))
        self._box_body_ids.append(_make_box(w, t / 2, h, 0, w - t / 2, wall_z))
        self._box_body_ids.append(_make_box(t / 2, w, h, -w + t / 2, 0, wall_z))
        self._box_body_ids.append(_make_box(t / 2, w, h, w - t / 2, 0, wall_z))
        return self._box_body_ids[0]  # return first as "main" id for compatibility

    def _create_cube(
        self, position: tuple[float, float, float], color: list[float]
    ) -> int:
        """Create a cube at the given position with the given color."""
        half_extents = [self.cube_size] * 3
        visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
        )
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
        cube_id = p.createMultiBody(
            baseMass=self.cube_mass,
            baseCollisionShapeIndex=collision,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=self._client_id,
        )
        return cube_id

    def _spawn_cubes(self) -> None:
        """Spawn red and blue cubes at random positions on the table."""
        z = self.table_height + self.cube_size + 0.001

        # Red cube - random position
        x_red = np.random.uniform(*self.cube_spawn_range_x)
        y_red = np.random.uniform(*self.cube_spawn_range_y)
        self._red_cube_id = self._create_cube(
            (x_red, y_red, z), self._red_color
        )

        # Blue cube - random position (ensure not too close to red)
        for _ in range(20):
            x_blue = np.random.uniform(*self.cube_spawn_range_x)
            y_blue = np.random.uniform(*self.cube_spawn_range_y)
            dist = np.sqrt((x_blue - x_red) ** 2 + (y_blue - y_red) ** 2)
            if dist > self.cube_size * 4:
                break
        self._blue_cube_id = self._create_cube(
            (x_blue, y_blue, z), self._blue_color
        )

    def _remove_cubes(self) -> None:
        """Remove existing cubes from the simulation."""
        cid = self._client_id
        for cube_id in [self._red_cube_id, self._blue_cube_id]:
            if cube_id is not None:
                p.removeBody(cube_id, physicsClientId=cid)
        self._red_cube_id = None
        self._blue_cube_id = None

    def _reset_robot(self) -> None:
        """Reset robot: 直接设置 robot_initial_joints [j0..j6, finger1, finger2].
        同时将电机目标设为初始位置，避免 reset 后电机仍瞄准上一 episode 的位姿导致弹回。
        """
        if self._robot_id is None:
            return

        cid = self._client_id
        j = self._robot_initial_joints
        for i in range(self._num_arm_joints):
            p.resetJointState(
                self._robot_id, i, j[i], targetVelocity=0, physicsClientId=cid
            )
            p.setJointMotorControl2(
                self._robot_id, i, p.POSITION_CONTROL,
                targetPosition=j[i], targetVelocity=0,
                force=300, positionGain=self._arm_position_gain, velocityGain=self._arm_velocity_gain,
                physicsClientId=cid,
            )
        for idx, ji in enumerate(self._finger_joints):
            p.resetJointState(
                self._robot_id, ji, j[7 + idx], targetVelocity=0, physicsClientId=cid
            )
            p.setJointMotorControl2(
                self._robot_id, ji, p.POSITION_CONTROL,
                targetPosition=j[7 + idx], targetVelocity=0,
                force=20, positionGain=0.1, velocityGain=0.5,
                physicsClientId=cid,
            )

    def _get_ee_pos(self) -> np.ndarray:
        """Get current end-effector position."""
        ls = p.getLinkState(
            self._robot_id, self._ee_link, physicsClientId=self._client_id
        )
        return np.array(ls[4])

    def _get_ik_joint_targets(
        self,
        target_pos: np.ndarray,
        target_orn: np.ndarray | None = None,
    ) -> list[float]:
        """
        Solve IK and return joint targets (no state change). Used for motor control.
        PyBullet calculateInverseKinematics 的 targetPosition 使用世界坐标系 (world frame)，
        与 getBasePositionAndOrientation/getLinkState 返回的坐标系一致。
        """
        cid = self._client_id
        if target_orn is None:
            ls = p.getLinkState(
                self._robot_id, self._ee_link, physicsClientId=cid
            )
            target_orn = ls[5]
        rest = self._robot_initial_joints[: self._num_arm_joints]
        ll, ul = self._ik_joint_limits
        jr = [ul[i] - ll[i] for i in range(self._num_arm_joints)]
        joint_poses = p.calculateInverseKinematics(
            self._robot_id,
            self._ee_link,
            target_pos.tolist(),
            target_orn,
            lowerLimits=ll,
            upperLimits=ul,
            jointRanges=jr,
            restPoses=rest,
            physicsClientId=cid,
        )
        return list(joint_poses[: self._num_arm_joints])

    def _apply_action(self, action: np.ndarray) -> None:
        """
        Apply robot action: [target_x, target_y, target_z, gripper_width].
        Sets motor targets via IK for arm and gripper.
        应用 ee_to_grasp_offset 补偿 panda_hand 原点与视觉抓取中心的偏移。
        """
        if action is None or len(action) < 4:
            return
        target_pos = np.array(action[:3], dtype=np.float64) + self._ee_to_grasp_offset_xyz
        gripper_width = float(action[3])
        joint_targets = self._get_ik_joint_targets(target_pos)
        cid = self._client_id
        for i in range(min(self._num_arm_joints, len(joint_targets))):
            p.setJointMotorControl2(
                self._robot_id, i, p.POSITION_CONTROL,
                targetPosition=joint_targets[i], targetVelocity=0,
                force=300, positionGain=self._arm_position_gain, velocityGain=self._arm_velocity_gain,
                physicsClientId=cid,
            )
        self._set_gripper(gripper_width)

    def _step_simulation(self, steps: int = 1) -> None:
        """Step the physics simulation."""
        for _ in range(steps):
            p.stepSimulation(physicsClientId=self._client_id)

    def _get_observation(self) -> dict:
        """Get current observation (cube positions, robot state, ee pos, etc.)."""
        obs = {}
        cid = self._client_id

        if self._red_cube_id is not None:
            pos, _ = p.getBasePositionAndOrientation(
                self._red_cube_id, physicsClientId=cid
            )
            obs["red_cube_pos"] = np.array(pos)
        if self._blue_cube_id is not None:
            pos, _ = p.getBasePositionAndOrientation(
                self._blue_cube_id, physicsClientId=cid
            )
            obs["blue_cube_pos"] = np.array(pos)
        obs["box_pos"] = self.box_position.copy()
        if self._robot_id is not None:
            obs["ee_pos"] = self._get_ee_pos()

        return obs

    # -------------------------------------------------------------------------
    # 数据采集：固定频率采集图像、关节位置、action（供 LeRobot 等使用）
    # -------------------------------------------------------------------------

    def get_joint_positions(self) -> np.ndarray:
        """
        返回当前机器人关节位置，用于数据采集。
        顺序: [j0..j6 (7 个臂关节), finger1, finger2]，共 9 维。
        """
        if self._robot_id is None:
            return np.zeros(9, dtype=np.float64)
        cid = self._client_id
        positions = []
        for i in range(self._num_arm_joints):
            positions.append(
                p.getJointState(self._robot_id, i, physicsClientId=cid)[0]
            )
        for ji in self._finger_joints:
            positions.append(p.getJointState(self._robot_id, ji, physicsClientId=cid)[0])
        return np.array(positions, dtype=np.float64)

    def _get_camera_view_params(self, camera_type: str) -> tuple[list, list, list]:
        """
        返回 (eye, target, up) 用于 computeViewMatrix。
        固定相机：球坐标约定与 computeViewMatrixFromYawPitchRoll 一致，
        相机位于目标上方俯视（pitch 负值 → 相机 z 高于 target）。
        """
        cid = self._client_id
        if camera_type == "fixed":
            target = list(self._camera_target)
            dist = self._camera_distance
            yaw_rad = np.radians(self._camera_yaw_deg)
            pitch_rad = np.radians(self._camera_pitch_deg)
            dx = dist * np.cos(pitch_rad) * np.cos(yaw_rad)
            dy = dist * np.cos(pitch_rad) * np.sin(yaw_rad)
            dz = -dist * np.sin(pitch_rad)
            eye = [
                target[0] + dx,
                target[1] + dy,
                target[2] + dz,
            ]
            up = [0.0, 0.0, 1.0]
            return eye, target, up
        elif camera_type == "gripper":
            # 腕部相机：位于夹爪横向约 0.04m，向下观看；旋转对齐世界 Y 轴（up=Y）
            ls = p.getLinkState(
                self._robot_id,
                self._ee_link,
                physicsClientId=cid,
            )
            ee_pos = np.array(ls[4])
            ee_orn = np.array(ls[5])
            R = np.array(p.getMatrixFromQuaternion(ee_orn)).reshape(3, 3)
            forward = R @ np.array([1.0, 0.0, 0.0])
            forward = forward / (np.linalg.norm(forward) + 1e-8)
            lateral = R @ np.array([0.0, 1.0, 0.0])
            lateral = lateral / (np.linalg.norm(lateral) + 1e-8)
            lateral_offset = 0.04
            look_down = 0.12
            eye = (ee_pos + lateral_offset * lateral).tolist()
            target = (ee_pos - look_down * np.array([0.0, 0.0, 1.0])).tolist()
            up = [0.0, 1.0, 0.0]  # 相机 up 对齐世界 Y 轴
            return eye, target, up
        raise ValueError(
            f"camera_type 必须是 'fixed' 或 'gripper'，得到: {camera_type}"
        )

    def get_camera_eye_target(self, camera_type: str) -> tuple[list, list]:
        """返回 (eye, target) 世界坐标 [x,y,z]，用于确认相机位置。固定相机在 GUI 下为调试窗口当前视角。"""
        if camera_type == "fixed" and self.use_gui:
            cam = p.getDebugVisualizerCamera(physicsClientId=self._client_id)
            if len(cam) >= 12:
                # cam[10]=distance, cam[11]=target; cam[8,9]=yaw,pitch (度)
                dist, target = cam[10], list(cam[11])
                yaw, pitch = cam[8], cam[9]
                pitch_rad = np.radians(pitch)
                yaw_rad = np.radians(yaw)
                dx = dist * np.cos(pitch_rad) * np.cos(yaw_rad)
                dy = dist * np.cos(pitch_rad) * np.sin(yaw_rad)
                dz = -dist * np.sin(pitch_rad)
                eye = [target[0] + dx, target[1] + dy, target[2] + dz]
                return eye, target
        eye, target, _ = self._get_camera_view_params(camera_type)
        return eye, target

    def render_camera_image(
        self,
        camera_type: str = "fixed",
        width: int = 224,
        height: int = 224,
        fov: float = 60.0,
        near: float = 0.02,
        far: float = 5.0,
    ) -> np.ndarray:
        """
        渲染指定相机的 RGB 图像。

        Args:
            camera_type: "fixed" = 固定场景视角（类似当前 debug 相机），
                        "gripper" = 安装在夹爪上的相机，随末端一起移动。
            width, height: 图像分辨率。
            fov: 垂直方向视场角（度），默认 60 与 debug 视图一致。
            near, far: 近/远裁剪面（米）。

        Returns:
            RGB 图像 (height, width, 3)，dtype uint8，取值 0-255。
        """
        cid = self._client_id
        aspect = width / float(height)
        fov_rad = np.radians(fov)
        projection_matrix = p.computeProjectionMatrixFOV(
            fov_rad, aspect, near, far
        )

        # 固定相机：使用 computeViewMatrixFromYawPitchRoll 与 resetDebugVisualizerCamera 一致
        # 注意：yaw/pitch 传入角度（度），与 Bullet debug 相机相同
        if camera_type == "fixed":
            view_matrix = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self._camera_target,
                distance=self._camera_distance,
                yaw=self._camera_yaw_deg,
                pitch=self._camera_pitch_deg,
                roll=0,
                upAxisIndex=2,
            )
        else:
            eye, target, up = self._get_camera_view_params(camera_type)
            view_matrix = p.computeViewMatrix(eye, target, up)

        # GUI 下用 OpenGL 与窗口一致；DIRECT 下用 Tiny 软件渲染
        renderer = (
            p.ER_BULLET_HARDWARE_OPENGL
            if self.use_gui
            else p.ER_TINY_RENDERER
        )
        result = p.getCameraImage(
            width,
            height,
            view_matrix,
            projection_matrix,
            shadow=True,
            renderer=renderer,
            physicsClientId=cid,
        )
        # result: (width, height, rgb, depth, seg)；rgb 可能为扁平 (height*width*4,)
        rgb = np.array(result[2], dtype=np.uint8)
        if rgb.ndim == 1:
            rgb = np.reshape(rgb, (height, width, 4))
        if rgb.shape[-1] == 4:
            rgb = rgb[:, :, :3]
        assert rgb.shape == (height, width, 3), rgb.shape
        return rgb

    def collect_snapshot(
        self,
        action: np.ndarray | None,
        image_width: int = 224,
        image_height: int = 224,
    ) -> dict:
        """
        采集当前时刻的一帧数据：固定视角图像、夹爪视角图像、关节位置、以及本步的 action。
        用于固定频率采集：在每步 step() 之后调用，传入该步的 action。

        Returns:
            dict:
                - "observation.images.top": (H,W,3) uint8，固定场景相机
                - "observation.images.wrist": (H,W,3) uint8，夹爪相机
                - "observation.state": (9,) float64，关节位置
                - "action": (4,) float64，[x,y,z,gripper]，若 action 为 None 则为 zeros
        """
        img_fixed = self.render_camera_image(
            "fixed", width=image_width, height=image_height
        )
        img_gripper = self.render_camera_image(
            "gripper", width=image_width, height=image_height
        )
        joint_pos = self.get_joint_positions()
        action_arr = (
            np.asarray(action, dtype=np.float64)
            if action is not None and len(action) >= 4
            else np.zeros(4, dtype=np.float64)
        )
        if action_arr.shape[0] > 4:
            action_arr = action_arr[:4]
        eye_top, tgt_top = self.get_camera_eye_target("fixed")
        eye_wrist, tgt_wrist = self.get_camera_eye_target("gripper")
        return {
            "observation.images.top": img_fixed,
            "observation.images.wrist": img_gripper,
            "observation.state": joint_pos,
            "action": action_arr,
            "camera_top_eye": eye_top,
            "camera_top_target": tgt_top,
            "camera_wrist_eye": eye_wrist,
            "camera_wrist_target": tgt_wrist,
        }

    def step(
        self,
        action: np.ndarray | None = None,
        sim_steps_per_call: int = 24,
        step_delay: float = 0.0,
    ) -> tuple[dict, float, bool, dict]:
        """
        统一的控制入口。应用 action，运行仿真步，返回 obs、reward、done、info。

        Args:
            action: [target_x, target_y, target_z, gripper_width] 末端目标位置与夹爪宽度。
                    None 则不更新电机目标，仅运行仿真。
            sim_steps_per_call: 每次调用运行多少仿真步。
            step_delay: 每步延时（秒），用于可视化。

        Returns:
            observation, reward, done, info
        """
        if action is not None:
            self._apply_action(np.asarray(action))
        for _ in range(sim_steps_per_call):
            self._step_simulation(1)
            if step_delay > 0:
                time.sleep(step_delay)
        obs = self._get_observation()
        reward = 0.0
        done = False
        info = {}
        return obs, reward, done, info

    def get_debug_positions(self) -> dict[str, np.ndarray]:
        """
        返回 ee_pos、方块、盒子位置，用于调试坐标系对齐。
        若夹爪与方块/盒子有偏差，可尝试调整 target_frame_rotation_deg 或 target_frame_offset_xy。
        """
        out = {"box_pos": self.box_position.copy()}
        if self._robot_id is not None:
            out["ee_pos"] = self._get_ee_pos()
        if self._red_cube_id is not None:
            pos, _ = p.getBasePositionAndOrientation(
                self._red_cube_id, physicsClientId=self._client_id
            )
            out["red_cube_pos"] = np.array(pos)
        if self._blue_cube_id is not None:
            pos, _ = p.getBasePositionAndOrientation(
                self._blue_cube_id, physicsClientId=self._client_id
            )
            out["blue_cube_pos"] = np.array(pos)
        return out

    def get_cube_positions(self) -> dict[str, np.ndarray]:
        """Get current positions of red and blue cubes."""
        positions = {}
        if self._red_cube_id is not None:
            pos, _ = p.getBasePositionAndOrientation(
                self._red_cube_id, physicsClientId=self._client_id
            )
            positions["red"] = np.array(pos)
        if self._blue_cube_id is not None:
            pos, _ = p.getBasePositionAndOrientation(
                self._blue_cube_id, physicsClientId=self._client_id
            )
            positions["blue"] = np.array(pos)
        return positions

    def close(self) -> None:
        """Disconnect from PyBullet and clean up."""
        if self._client_id is not None:
            try:
                p.disconnect(physicsClientId=self._client_id)
            except p.error:
                # 如果物理服务器已经断开连接，忽略错误
                pass
            self._client_id = None
        self._robot_id = None
        self._table_id = None
        self._box_id = None
        self._box_body_ids = []
        self._red_cube_id = None
        self._blue_cube_id = None

    def __enter__(self) -> "PickPlaceEnv":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
