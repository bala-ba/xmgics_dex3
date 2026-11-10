"""
MuJoCo机器人控制脚本
实现位置和力矩混合控制，读取传感器数据
"""
import mujoco
import mujoco.viewer
import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading
import rclpy
from rclpy.node import Node
# import message_filters
from bodyctrl_msgs.msg import MotorCtrl, MotorStatus,MotorStatusMsg, CmdMotorCtrl, Imu, Euler
from sensor_msgs.msg import JointState, Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import glfw
from queue import Queue
from scipy.spatial.transform import Rotation
import os
import sys
from elastic_band import ElasticBand

# 缩短 GIL 切换间隔, 降低物理线程被 400Hz 发布线程阻塞后的唤醒延迟
sys.setswitchinterval(0.0005)

def quaternion_to_euler_scipy(w, x, y, z, degrees=False):
    """
    使用SciPy进行四元数转欧拉角
    """
    # 创建旋转对象 (注意顺序: [x, y, z, w])
    rotation = Rotation.from_quat([x, y, z, w])

    # 转换为欧拉角 (顺序: 'xyz' 对应 roll, pitch, yaw)
    euler_angles = rotation.as_euler('xyz', degrees=degrees)

    return euler_angles[0], euler_angles[1], euler_angles[2]

def sleep_until(deadline: float):
    """睡眠到绝对截止时间: 先 sleep 到剩 0.2ms, 其余自旋等待,
    规避 time.sleep 的定时器余量(中位约 60us)"""
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return
        if remaining > 2e-4:
            time.sleep(remaining - 2e-4)


class RobotSimulator:

    def __init__(self, model_path, node: Node | None = None, debug=False,
                 # depth=True, depth_width=640, depth_height=360):
                 depth=True, depth_width=320, depth_height=180, headless=False,
                 pd_mode='auto'):
                 # depth=True, depth_width=64, depth_height=36):
        self.debug = debug
        self.headless = headless
        self.thread_error = None
        # 加载模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 初始化控制模式 (0=(位置控制(仅上半身)和力控都存在), 1=全身仅力矩控制)
        self.control_mode = 1

        # 记录数据用于绘图
        record_length = 10000
        self.time_history = deque(maxlen=record_length)
        self.joint_pos_history = deque(maxlen=record_length)
        self.joint_vel_history = deque(maxlen=record_length)
        self.joint_torque_history = deque(maxlen=record_length)
        self.imu_orient_history = deque(maxlen=record_length)
        self.imu_accel_history = deque(maxlen=record_length)
        self.imu_pos_history = deque(maxlen=record_length)

        # 机器人电机ID到关节映射
        self.motor_id_to_joint = {
            1: 'head_roll_joint',
            2: 'head_pitch_joint',
            3: 'head_yaw_joint',
            11: 'shoulder_pitch_l_joint',
            12: 'shoulder_roll_l_joint',
            13: 'shoulder_yaw_l_joint',
            14: 'elbow_pitch_l_joint',
            15: 'elbow_yaw_l_joint',
            16: 'wrist_pitch_l_joint',
            17: 'wrist_roll_l_joint',
            21: 'shoulder_pitch_r_joint',
            22: 'shoulder_roll_r_joint',
            23: 'shoulder_yaw_r_joint',
            24: 'elbow_pitch_r_joint',
            25: 'elbow_yaw_r_joint',
            26: 'wrist_pitch_r_joint',
            27: 'wrist_roll_r_joint',
            # 31: 'waist_yaw_joint',
            # 32: 'waist_roll_joint',
            # 33: 'waist_pitch_joint',
            33: 'waist_yaw_joint',
            32: 'waist_roll_joint',
            31: 'waist_pitch_joint',
            51: 'hip_pitch_l_joint',
            52: 'hip_roll_l_joint',
            53: 'hip_yaw_l_joint',
            54: 'knee_pitch_l_joint',
            55: 'ankle_pitch_l_joint',
            56: 'ankle_roll_l_joint',
            61: 'hip_pitch_r_joint',
            62: 'hip_roll_r_joint',
            63: 'hip_yaw_r_joint',
            64: 'knee_pitch_r_joint',
            65: 'ankle_pitch_r_joint',
            66: 'ankle_roll_r_joint'
        }
        # 关节到电机ID的映射
        self.joint_to_motor_id = {
            joint: id
            for id, joint in self.motor_id_to_joint.items()
        }

        #双臂关节名称
        self.arm_joint_names = [
            'shoulder_pitch_l_joint', 'shoulder_roll_l_joint',
            'shoulder_yaw_l_joint', 'elbow_pitch_l_joint', 'elbow_yaw_l_joint',
            'wrist_pitch_l_joint', 'wrist_roll_l_joint',
            'shoulder_pitch_r_joint', 'shoulder_roll_r_joint',
            'shoulder_yaw_r_joint', 'elbow_pitch_r_joint', 'elbow_yaw_r_joint',
            'wrist_pitch_r_joint', 'wrist_roll_r_joint'
        ]
        #腰部关节名
        self.waist_joint_names = [
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint'
        ]
        #腿部关节名称
        self.leg_joint_names = [
            'hip_roll_l_joint', 'hip_pitch_l_joint', 'hip_yaw_l_joint',
            'knee_pitch_l_joint', 'ankle_pitch_l_joint', 'ankle_roll_l_joint',
            'hip_roll_r_joint', 'hip_pitch_r_joint', 'hip_yaw_r_joint',
            'knee_pitch_r_joint', 'ankle_pitch_r_joint', 'ankle_roll_r_joint'
        ]
        # 获取传感器/执行器索引
        self.get_all_robot_info()
        if pd_mode not in ('auto', 'explicit', 'implicit'):
            raise ValueError(f'Unknown PD mode: {pd_mode}')
        self.pd_mode = ('explicit' if self.pos_actuator_indices_map else 'implicit') if pd_mode == 'auto' else pd_mode

        self.sim_view_lock = threading.Lock()
        
        # 升降带
        self.elastic_band = ElasticBand(enable=True)
        self.band_attached_link = self.model.body("waist_pitch_link").id

        # 深度相机：必须在 launch_passive 之前于主线程创建 GL 上下文并立即
        # 释放（GLFW 仅允许主线程建窗口）。渲染线程逐帧 make-current/
        # release，与 viewer 各自的上下文互不冲突，可同进程共存。
        # 做法移植自 bitbot-ovinf src/device/mujoco_depth_camera.cc
        self.depth_renderer = None
        self.depth_enabled = False
        self.depth_width = depth_width
        self.depth_height = depth_height
        if depth:
            cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA,
                                       'depth_camera')
            if cam_id == -1:
                print("警告: 模型中未定义 'depth_camera' 相机，深度渲染已禁用")
            else:
                self.depth_renderer = mujoco.Renderer(
                    self.model, height=depth_height, width=depth_width)
                self.depth_renderer.enable_depth_rendering()
                glfw.make_context_current(None)
                self.depth_enabled = True
                self._depth_cam_id = cam_id
                # 深度渲染裁剪面 (bitbot-ovinf dex_mj.xml: znear=0.1, zfar=50)。
                # vis.map 为相对 extent 的比值; znear=0.1 同时遮挡相机外壳。
                extent = float(self.model.stat.extent)
                self.model.vis.map.znear = 0.20 / extent
                self.model.vis.map.zfar = 50.0 / extent

        # 启动可视化器
        self.paused = False
        # self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer = None
        if not headless:
            self.viewer = mujoco.viewer.launch_passive(
                self.model, self.data,
                key_callback=self.elastic_band.MujuocoKeyCallback)
        self.view_dt = 0.03
        # 线程控制变量
        self.joint_cmd_lock = threading.Lock()
        self.stop_event = threading.Event()

        #创建ROS相关话题
        self.node = node
        if self.node:
            #创建发布者
            self.arm_status_pub = self.node.create_publisher(
                MotorStatusMsg, 'arm/status', 20)
            self.leg_status_pub = self.node.create_publisher(
                MotorStatusMsg, 'leg/status', 20)
            self.waist_status_pub = self.node.create_publisher(
                MotorStatusMsg, 'waist/status', 10)
            self.imu_status_pub = self.node.create_publisher(
                Imu, 'imu/status', 10)
            # 供外部渲染/可视化进程使用的标准话题
            self.joint_states_pub = self.node.create_publisher(
                JointState, 'joint_states', 10)
            self.pelvis_pose_pub = self.node.create_publisher(
                PoseStamped, 'pelvis/pose', 10)
            if self.depth_enabled:
                self.depth_img_pub = self.node.create_publisher(
                    Image, '/camera/depth/image_raw', 10)
                self.depth_info_pub = self.node.create_publisher(
                    CameraInfo, 'depth_camera/camera_info', 10)
                fovy = float(self.model.cam_fovy[self._depth_cam_id])
                fy = 0.5 * self.depth_height / np.tan(np.radians(fovy) / 2.0)
                cx = (self.depth_width - 1) / 2.0
                cy = (self.depth_height - 1) / 2.0
                info = CameraInfo()
                info.header.frame_id = 'depth_camera'
                info.height = self.depth_height
                info.width = self.depth_width
                info.distortion_model = 'plumb_bob'
                info.d = [0.0] * 5
                info.k = [fy, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
                info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                info.p = [fy, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
                self.depth_info = info
        # 传感器返回的机器人状态
        self.joint_positions = {}
        self.joint_velocities = {}
        self.joint_torques = {}
        self.imu_orientation = None
        self.imu_position = None
        self.imu_angular_velocity = None
        self.imu_linear_velocity = None
        self.imu_linear_acceleration = None
        self.imu_magnetometer = None

        # 关节命令
        self.raw_joint_commands = Queue()
        self.joint_commands = {}
        self.last_raw_joint_commands = {}

        # 发布线程相关
        self.pub_thread = None
        self.pub_thread_lock = threading.Lock()
        self.pub_thread_running = False
        self.last_sensor_data = None
        self.sensor_data_lock = threading.Lock()

        # 初始化控制器
        self.init_controller()
        self.configure_pd()
        # 首条状态必须包含有效四元数，而不是尚未更新的全零 sensordata。
        mujoco.mj_forward(self.model, self.data)
        self.read_sensors()
        # 订阅放在状态初始化完成后创建，避免回调早于属性就绪
        if self.node:
            self.node.create_subscription(CmdMotorCtrl, 'arm/cmd_ctrl',
                                          self.joint_cmd_callback, 10)
            self.node.create_subscription(CmdMotorCtrl, 'leg/cmd_ctrl',
                                          self.joint_cmd_callback, 10)
            self.node.create_subscription(CmdMotorCtrl, 'waist/cmd_ctrl',
                                          self.joint_cmd_callback, 10)

    def configure_pd(self):
        """把 PD 的速度项交给隐式积分器，避免轻关节上的显式阻尼数值振荡。

        单位齿比电机：force = ctrl - kp*q - kd*dq；
        ctrl = kp*q_target + kd*dq_target + feedforward。
        限制的是最终 force，不能提前裁剪含 kp*q_target 的 ctrl。
        """
        if self.pd_mode == 'implicit':
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
            for name, idx in self.motor_actuator_indices_map.items():
                if (self.model.actuator_trntype[idx] != mujoco.mjtTrn.mjTRN_JOINT
                        or self.model.actuator_dyntype[idx] != mujoco.mjtDyn.mjDYN_NONE
                        or self.model.actuator_gaintype[idx] != mujoco.mjtGain.mjGAIN_FIXED
                        or self.model.actuator_biastype[idx] != mujoco.mjtBias.mjBIAS_NONE
                        or self.model.actuator_gainprm[idx, 0] != 1
                        or not np.array_equal(self.model.actuator_gear[idx], [1, 0, 0, 0, 0, 0])):
                    raise ValueError(f'Implicit PD requires a unit-gear torque motor: {name}')
                if self.model.actuator_ctrllimited[idx]:
                    limits = self.model.actuator_ctrlrange[idx].copy()
                    if self.model.actuator_forcelimited[idx]:
                        limits[0] = max(limits[0], self.model.actuator_forcerange[idx, 0])
                        limits[1] = min(limits[1], self.model.actuator_forcerange[idx, 1])
                    if limits[0] >= limits[1]:
                        raise ValueError(f'Invalid force limits: {name}')
                    self.model.actuator_forcelimited[idx] = True
                    self.model.actuator_forcerange[idx] = limits
                    self.model.actuator_ctrllimited[idx] = False
                self.model.actuator_biastype[idx] = mujoco.mjtBias.mjBIAS_AFFINE
                self.model.actuator_biasprm[idx, :3] = 0
        print(f'PD integration: {self.pd_mode} (feedback from current physics state)')

    def init_controller(self):
        if self.control_mode == 1:
            # 将位置控制器的增益设为0
            for idx in self.pos_actuator_indices_map.values():
                self.model.actuator_gainprm[idx, 0] = 0  # kp = 0
                self.model.actuator_biasprm[idx, 1] = 0  # kv = 0 (biasprm[1]通常是kv)
            self.control_mode = 1
            print("切换到力矩控制模式：位置控制器已禁用")
    def switch_to_torque_control(self):
        """切换到力矩控制：禁用位置控制器"""
        if self.control_mode == 0:
            # 将位置控制器的增益设为0
            for idx in self.pos_actuator_indices_map.values():
                self.model.actuator_gainprm[idx, 0] = 0  # kp = 0
                self.model.actuator_biasprm[idx, 1] = 0  # kv = 0 (biasprm[1]通常是kv)
            self.control_mode = 1
            print("切换到力矩控制模式：位置控制器已禁用")
    
    def switch_to_position_torque_control(self):
        """切换到位置力控混合控制模式：启用位置控制器"""
        if not self.pos_actuator_indices_map:
            print("当前模型仅有力矩执行器，保持力矩控制模式")
            return
        if self.control_mode == 1:
            # 恢复位置控制器的原始增益
            for idx in self.pos_actuator_indices_map.values():
                self.model.actuator_gainprm[idx] = self.original_gains[idx]
                self.model.actuator_biasprm[idx] = self.original_biases[idx]
            self.control_mode = 0
            print("切换到位置力控混合控制模式：位置控制器已启用")

    def start_keyboard_listener(self):
        """使用 pynput 监听键盘"""
        from pynput import keyboard

        def on_press(key):
            try:
                if key == keyboard.Key.num_lock:
                    self.paused = not self.paused
                    print(f"仿真已{'暂停' if self.paused else '继续'}")
                if key == keyboard.Key.end:
                    if self.control_mode == 1:
                        self.switch_to_position_torque_control()
                    else:
                        self.switch_to_torque_control()
            except Exception as e:
                print(f"键盘监听错误: {e}")

        def listen():
            with keyboard.Listener(on_press=on_press) as listener:
                while not self.stop_event.is_set():
                    time.sleep(0.1)
                listener.stop()

        thread = threading.Thread(target=listen, daemon=True)
        thread.start()

    def _publish_joint_states(self, sensor_data):
        """发布 /joint_states 与 /pelvis/pose（供外部渲染/可视化用）"""
        if not sensor_data:
            return
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = 'pelvis'
        joint_positions = sensor_data['joint_positions']
        for name, pos in joint_positions.items():
            msg.name.append(name)
            msg.position.append(float(pos))
        self.joint_states_pub.publish(msg)

        pose = PoseStamped()
        pose.header.stamp = msg.header.stamp
        pose.header.frame_id = 'world'
        quat = sensor_data['imu_orientation']  # [w, x, y, z]
        pos = sensor_data['imu_position']      # imu site 即骨盆原点
        pose.pose.position.x = float(pos[0])
        pose.pose.position.y = float(pos[1])
        pose.pose.position.z = float(pos[2])
        pose.pose.orientation.w = float(quat[0])
        pose.pose.orientation.x = float(quat[1])
        pose.pose.orientation.y = float(quat[2])
        pose.pose.orientation.z = float(quat[3])
        self.pelvis_pose_pub.publish(pose)

    def _render_and_publish_depth(self):
        """同进程离屏渲染深度图并发布。

        update_scene 仅读 MjData，需持有 sim_view_lock；GL 渲染只依赖已
        填充的 MjvScene，可在锁外进行。渲染前 make-current（render 内部
        完成）、渲染后释放，是 viewer 与本上下文共存的关键。
        """
        try:
            with self.sim_view_lock:
                self.depth_renderer.update_scene(self.data,
                                                 camera='depth_camera')
            depth = self.depth_renderer.render()
            glfw.make_context_current(None)
            depth = self._apply_depth_noise(depth)
        except Exception as e:
            print(f"深度渲染失败，已禁用: {e}")
            self.depth_enabled = False
            return
        now = self.node.get_clock().now().to_msg()
        img = Image()
        img.header.stamp = now
        img.header.frame_id = 'depth_camera'
        img.height = self.depth_height
        img.width = self.depth_width
        img.encoding = '32FC1'
        img.is_bigendian = 0
        img.step = self.depth_width * 4
        img.data = np.ascontiguousarray(depth, dtype=np.float32).tobytes()
        self.depth_img_pub.publish(img)
        self.depth_info.header.stamp = now
        self.depth_info_pub.publish(self.depth_info)

    def _apply_depth_noise(self, depth):
        """深度加噪流水线, 移植自 instinctlab dex23dof_depth_cfg.py 的
        noise_pipeline (取其前四级): 高斯噪声 → 亮点/阴影伪影 → 天空伪影"""
        # 高斯噪声: probability=1.0, noise_mean=0.0, noise_std=0.015
        depth = depth + np.random.normal(0.0, 0.015, depth.shape)
        # 亮点伪影(镜面过曝, 值超量程) / 阴影伪影(双目失配归零)
        self._stamp_artifact_patches(depth, 1e-4, (2, 0.5), [4.0])
        self._stamp_artifact_patches(depth, 1e-4, (2, 0.5), [0.0])
        # 天空伪影: 像素及其上方整列均过远(>2m)视为天空,
        # 天空区域内随机小块替换为典型天空量测值
        sky = np.cumsum(depth > 2.0, axis=0) > np.arange(depth.shape[0])[:, None]
        self._stamp_artifact_patches(depth, 1e-4, (2, 3.2),
                                     [0.6, 1.0, 1.2, 1.5, 1.8], region_mask=sky)
        return depth

    @staticmethod
    def _stamp_artifact_patches(depth, prob, size_mean_std, values,
                                region_mask=None):
        """以每像素 prob 概率撒种子, 种子处长出边长 ~ N(mean, std) 的小方块
        伪影, 值从 values 随机取; region_mask 限制伪影只落在掩码区域内"""
        h, w = depth.shape
        seeds = np.random.random(depth.shape) < prob
        for y, x in zip(*np.nonzero(seeds)):
            value = np.random.choice(values)
            ph = int(np.clip(np.random.normal(*size_mean_std), 0, h))
            pw = int(np.clip(np.random.normal(*size_mean_std), 0, w))
            y0, y1 = max(y - ph // 2, 0), min(y + (ph + 1) // 2, h)
            x0, x1 = max(x - pw // 2, 0), min(x + (pw + 1) // 2, w)
            patch = depth[y0:y1, x0:x1]
            if region_mask is None:
                patch[:] = value
            else:
                patch[region_mask[y0:y1, x0:x1]] = value

    def start_publishing(self):
        """启动发布线程"""
        if not self.node:
            return
        with self.pub_thread_lock:
            if not self.pub_thread_running:
                self.pub_thread_running = True
                self.pub_thread = threading.Thread(target=self._publish_thread,
                                                   daemon=True)
                self.pub_thread.start()

    def stop_publishing(self):
        """停止发布线程"""
        with self.pub_thread_lock:
            if self.pub_thread_running:
                self.pub_thread_running = False
                if self.pub_thread:
                    self.pub_thread.join(timeout=1.0)

    def _publish_thread(self):
        """发布状态的独立线程"""
        pub_freq = 400  # 100Hz发布频率
        pub_interval = 1.0 / pub_freq
        sensor_data = None
        cycle = 0
        while self.pub_thread_running and not self.stop_event.is_set():
            start_time = time.perf_counter()

            # 获取最新的传感器数据
            with self.sensor_data_lock:
                if self.last_sensor_data is not None:
                    sensor_data = self.last_sensor_data.copy()
            # 为每个关节组发布状态
            self._publish_joint_group(self.arm_joint_names,
                                      self.arm_status_pub, sensor_data)
            self._publish_joint_group(self.waist_joint_names,
                                      self.waist_status_pub, sensor_data)
            self._publish_joint_group(self.leg_joint_names,
                                      self.leg_status_pub, sensor_data)
            # 发布Imu
            self._publish_imu_status(self.imu_status_pub, sensor_data)
            # 100Hz 发布 joint_states / pelvis_pose（400Hz 中每4次1次）
            if cycle % 4 == 0:
                self._publish_joint_states(sensor_data)
            # 50Hz 深度渲染（400Hz 中每8次1次, 对齐 ovinf depth frequency）
            if self.depth_enabled and cycle % 8 == 0:
                self._render_and_publish_depth()
            cycle += 1
            # 控制发布频率
            next_step_time = start_time + pub_interval
            now = time.perf_counter()
            if next_step_time < now - 0.01:
                next_step_time = now  # 落后过多(查看器卡顿等)时重同步, 不补跑
            sleep_until(next_step_time)

            # # 控制发布频率
            # elapsed = time.perf_counter() - start_time
            # sleep_time = pub_interval - elapsed
            # if sleep_time > 0:
            #     time.sleep(sleep_time)

    def _publish_imu_status(self, publisher, sensor_data):
        if publisher and sensor_data:
            orientaion = sensor_data['imu_orientation']
            # position = sensor_data['imu_position']
            imu_angular_velocity = sensor_data['imu_angular_velocity']
            imu_linear_acceleration = sensor_data['imu_linear_acceleration']
            imu_msg: Imu = Imu()
            imu_msg.header.stamp = self.node.get_clock().now().to_msg()
            imu_msg.orientation.x = orientaion[1]
            imu_msg.orientation.y = orientaion[2]
            imu_msg.orientation.z = orientaion[3]
            imu_msg.orientation.w = orientaion[0]
            # 四元数转换为欧拉角
            r, p, y = quaternion_to_euler_scipy(*list(orientaion))
            imu_msg.euler = Euler()
            imu_msg.euler.roll = r
            imu_msg.euler.pitch = p
            imu_msg.euler.yaw = y
            imu_msg.angular_velocity.x = imu_angular_velocity[0]
            imu_msg.angular_velocity.y = imu_angular_velocity[1]
            imu_msg.angular_velocity.z = imu_angular_velocity[2]
            imu_msg.linear_acceleration.x = imu_linear_acceleration[0]
            imu_msg.linear_acceleration.y = imu_linear_acceleration[1]
            imu_msg.linear_acceleration.z = imu_linear_acceleration[2]

            try:
                publisher.publish(imu_msg)
            except Exception as e:
                print(f"发布 imu 状态时出错: {e}")

    def _publish_joint_group(self, joint_names, publisher, sensor_data):
        """发布指定关节组的状态"""
        if publisher and sensor_data:
            joint_positions = sensor_data['joint_positions']
            joint_velocities = sensor_data['joint_velocities']
            joint_torques = sensor_data['joint_torques']
            #构造关节状态消息
            motor_status_msg = MotorStatusMsg()
            motor_status_msg.header.stamp = self.node.get_clock().now().to_msg()
            for joint_name in joint_names:
                if joint_name in joint_positions and joint_name in joint_velocities:
                    motor_status = self.construct_motor_status(
                        joint_name, joint_positions[joint_name],
                        joint_velocities[joint_name], joint_torques[joint_name])
                    motor_status_msg.status.append(motor_status)
            try:
                publisher.publish(motor_status_msg)
            except Exception as e:
                print(f"发布状态{motor_status_msg}时出错: {e}")

    def get_all_robot_info(self):
        self.joint_names = self.arm_joint_names + self.waist_joint_names + self.leg_joint_names
        self.joint_qpos_addresses = {name: int(self.model.joint(name).qposadr[0]) for name in self.joint_names}
        self.joint_dof_addresses = {name: int(self.model.joint(name).dofadr[0]) for name in self.joint_names}
        self.pos_actuator_names = [f"pos_{name}" for name in self.joint_names]
        self.motor_actuator_names = [f"motor_{name}" for name in self.joint_names]

        def required_id(kind, name):
            idx = mujoco.mj_name2id(self.model, kind, name)
            if idx < 0:
                raise ValueError(f"模型缺少仿真接口必需的 {kind.name}: {name}")
            return idx

        def sensor_id(name, dimension):
            idx = required_id(mujoco.mjtObj.mjOBJ_SENSOR, name)
            if self.model.sensor_dim[idx] != dimension:
                raise ValueError(f"传感器 {name} 应为 {dimension} 维")
            return idx

        # Tiangong3 只有 torque motor；缺失的 position actuator 不能以 -1 索引数组。
        self.pos_actuator_indices_map = {}
        for name in self.pos_actuator_names:
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if idx >= 0:
                self.pos_actuator_indices_map[name] = idx
        self.motor_actuator_indices_map = {
            name: required_id(mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.motor_actuator_names
        }
        self.original_gains = self.model.actuator_gainprm.copy()
        self.original_biases = self.model.actuator_biasprm.copy()

        for kind in ('pos', 'vel', 'torque'):
            names = [f"{name}_{kind}" for name in self.joint_names]
            setattr(self, f"joint_{kind}_sensor_names", names)
            setattr(self, f"joint_{kind}_sensor_indices_map",
                    {name: sensor_id(name, 1) for name in names})

        for attr, name, dimension in (
            ('orient', 'orientation', 4), ('pos', 'position', 3),
            ('gyro', 'angular-velocity', 3), ('vel', 'linear-velocity', 3),
            ('accel', 'linear-acceleration', 3), ('mag', 'magnetometer', 3),
        ):
            idx = sensor_id(name, dimension)
            setattr(self, f"imu_{attr}_idx", idx)
            setattr(self, f"imu_{attr}_address", self.model.sensor_adr[idx])

    def pvd_controller(self, joint_name, pos_d, spd_d, tor_d, kp, kd):
        """
        PVD控制器实现函数，用于计算关节的输出扭矩
        
        参数:
        joint_name: 关节名称，用于获取关节状态
        pos_d: 期望位置
        spd_d: 期望速度
        tor_d: 期望扭矩
        kp: 位置比例增益系数
        kd: 速度比例增益系数
        
        返回值:
        output_torque: 计算得到的输出扭矩值
        """

        # mj_step 完成积分后，sensordata 仍对应积分前；发布缓存不能用作内环反馈。
        pos_a = self.data.qpos[self.joint_qpos_addresses[joint_name]]
        spd_a = self.data.qvel[self.joint_dof_addresses[joint_name]]
        output_torque = kp * (pos_d - pos_a) + kd * (spd_d - spd_a) + tor_d
        return output_torque

    def joint_cmd_callback(self, msg: CmdMotorCtrl):
        cmd: MotorCtrl = None
        # 每一组命令全部接受完再释放锁
        with self.joint_cmd_lock:

            for cmd in msg.cmds:
                self.raw_joint_commands.put(cmd)
                

    def calc_motor_cmd(self, joint_name, pos, spd, tor, kp, kd):
        # joint_name = self.motor_id_to_joint[name_id]
        # 计算下发力矩
        target_torque = self.pvd_controller(joint_name, pos, spd, tor, kp, kd)
        self.joint_commands[joint_name] = target_torque
        if self.debug:
            print(f"{joint_name} torque : {target_torque}\n")

    def set_motor_cmd(self):
        with self.joint_cmd_lock:
            while not self.raw_joint_commands.empty():
                cmd: MotorCtrl = self.raw_joint_commands.get()
                name = cmd.name
                pos = cmd.pos
                spd = cmd.spd
                tor = cmd.tor
                kp = cmd.kp
                kd = cmd.kd
                joint_name = self.motor_id_to_joint[name]
                self.last_raw_joint_commands[joint_name] = { "pos": pos, "spd": spd, "tor": tor, "kp": kp, "kd": kd}
                # self.calc_motor_cmd(name, pos, spd, tor, kp, kd)


        for joint_name in self.joint_names :
            if joint_name in self.last_raw_joint_commands:
                pos = self.last_raw_joint_commands[joint_name].get("pos")
                spd = self.last_raw_joint_commands[joint_name].get("spd")
                tor = self.last_raw_joint_commands[joint_name].get("tor")
                kp = self.last_raw_joint_commands[joint_name].get("kp")
                kd = self.last_raw_joint_commands[joint_name].get("kd")
                self.calc_motor_cmd(joint_name, pos, spd, tor, kp, kd)
                motor_name = f"motor_{joint_name}"
                motor_index = self.motor_actuator_indices_map[motor_name]
                if motor_index != -1:
                    # 确保电机存在且已收到命令
                    if self.pd_mode == 'implicit':
                        self.model.actuator_biasprm[motor_index, 1] = -kp
                        self.model.actuator_biasprm[motor_index, 2] = -kd
                        self.data.ctrl[motor_index] = kp * pos + kd * spd + tor
                    else:
                        self.data.ctrl[motor_index] = self.joint_commands[joint_name]
    def is_running(self):
        return (not self.stop_event.is_set()
                and (self.viewer is None or self.viewer.is_running()))

    def viewer_thread(self):
        try:
            while self.is_running():
                with self.sim_view_lock:
                    self.viewer.sync()
                time.sleep(self.view_dt)
        except Exception as e:
            self.thread_error = e
            self.stop_event.set()
            print(f"Viewer thread error: {e}")
        finally:
            print("Viewer thread stopped")

    def read_sensors(self):
        """读取所有传感器数据"""
        # 读取关节传感器数据

        for name in self.joint_names:
            pos_sensor_name = f"{name}_pos"
            vel_sensor_name = f"{name}_vel"
            torque_sensor_name = f"{name}_torque"
            pos_idx = self.joint_pos_sensor_indices_map[pos_sensor_name]
            vel_idx = self.joint_vel_sensor_indices_map[vel_sensor_name]
            torque_idx = self.joint_torque_sensor_indices_map[
                torque_sensor_name]

            self.joint_positions[name] = self.data.sensordata[self.model.sensor_adr[pos_idx]]
            self.joint_velocities[name] = self.data.sensordata[self.model.sensor_adr[vel_idx]]
            self.joint_torques[name] = self.data.sensordata[self.model.sensor_adr[torque_idx]]

        # 复制快照，避免物理线程下一步修改发布线程正在使用的数组。
        for attr, address, size in (
            ('orientation', self.imu_orient_address, 4),
            ('position', self.imu_pos_address, 3),
            ('angular_velocity', self.imu_gyro_address, 3),
            ('linear_velocity', self.imu_vel_address, 3),
            ('linear_acceleration', self.imu_accel_address, 3),
            ('magnetometer', self.imu_mag_address, 3),
        ):
            setattr(self, f"imu_{attr}", self.data.sensordata[address:address + size].copy())
        # 更新传感器数据供发布线程使用
        plot_data = {
            'joint_positions': np.array(list(self.joint_positions.values())),
            'joint_velocities': np.array(list(self.joint_velocities.values())),
            'joint_torques': np.array(list(self.joint_torques.values())),
            'imu_orientation': self.imu_orientation,
            'imu_position': self.imu_position,
            'imu_angular_velocity': self.imu_angular_velocity,
            'imu_linear_velocity': self.imu_linear_velocity,
            'imu_linear_acceleration': self.imu_linear_acceleration,
            'imu_magnetometer': self.imu_magnetometer
        }
        with self.sensor_data_lock:
            self.last_sensor_data = {
                'joint_positions': self.joint_positions.copy(),
                'joint_velocities': self.joint_velocities.copy(),
                'joint_torques': self.joint_torques.copy(),
                'imu_orientation': self.imu_orientation,
                'imu_position': self.imu_position,
                'imu_angular_velocity': self.imu_angular_velocity,
                'imu_linear_velocity': self.imu_linear_velocity,
                'imu_linear_acceleration': self.imu_linear_acceleration,
                'imu_magnetometer': self.imu_magnetometer
            }

        return plot_data

    def construct_motor_status(self, joint_name, pos, speed, torque):
        """
        构建电机状态消息
        
        Args:
            joint_name (str): 关节名称
            pos (float): 位置值
            speed (float): 速度值
            
        Returns:
            MotorStatus: 电机状态消息
        """
        motor_status = MotorStatus()
        motor_status.name = self.joint_to_motor_id.get(joint_name, int())
        motor_status.pos = pos
        motor_status.speed = speed
        motor_status.current = torque

        return motor_status

    def get_status(self, joint_name):
        # Todo: 处理还未收到传感器状态的情况
        with self.sensor_data_lock:
            if (self.last_sensor_data is None
                    or joint_name not in self.last_sensor_data['joint_positions']):
                return {'position': 0.0, 'velocity': 0.0, 'torque': 0.0}
            joint_positions = self.last_sensor_data['joint_positions'].copy()
            joint_velocities = self.last_sensor_data['joint_velocities'].copy()
            joint_torques = self.last_sensor_data['joint_torques'].copy()
        pos = joint_positions[joint_name]
        vel = joint_velocities[joint_name]
        torque = joint_torques[joint_name]
        return {'position': pos, 'velocity': vel, 'torque': torque}

    def simulate_thread(self):
        """运行控制循环"""
        # 控制参数
        time_step = 0
        control_frequency = 1000  # 100Hz控制频率
        sensor_frequency = 1000  # 100Hz传感器读取频率
        control_interval = 1 / control_frequency

        target_time = 0
        # last_control_time = 0
        rate_t0 = time.perf_counter()
        rate_count = 0
        next_step_time = time.perf_counter()
        try:
            while self.is_running():
                step_start = time.perf_counter()

                rate_count += 1
                if step_start - rate_t0 >= 2.0:
                    print(f"sim rate: {rate_count / (step_start - rate_t0):.1f} Hz "
                          f"(target {1.0 / self.model.opt.timestep:.0f} Hz)")
                    rate_t0 = step_start
                    rate_count = 0

                target_time += self.model.opt.timestep
                with self.sim_view_lock:
                    # 读取传感器数据
                    if time_step % max(1, round(
                            1.0 / self.model.opt.timestep / sensor_frequency)) == 0:
                        sensor_data = self.read_sensors()
                        # 存储数据用于绘图
                        self.time_history.append(target_time)
                        self.joint_pos_history.append(
                            sensor_data['joint_positions'].copy())
                        self.joint_vel_history.append(
                            sensor_data['joint_velocities'].copy())
                        self.joint_torque_history.append(
                            sensor_data['joint_torques'].copy())
                        self.imu_orient_history.append(
                            sensor_data['imu_orientation'].copy())
                        self.imu_accel_history.append(
                            sensor_data['imu_linear_acceleration'].copy())
                        self.imu_pos_history.append(
                            sensor_data['imu_position'].copy())
                        # 打印部分传感器数据
                        if time_step % 100 == 0 and self.debug:
                            print(f"Time: {target_time:.3f}s")
                            print(
                                f"Control Mode: {'Position' if self.control_mode == 0 else 'Torque'}"
                            )
                            print(
                                f"Joint 0 Pos: {sensor_data['joint_positions'][0]:.3f}, "
                                f"Vel: {sensor_data['joint_velocities'][0]:.3f}, "
                                f"Torque: {sensor_data['joint_torques'][0]:.3f}"
                            )
                            print(
                                f"IMU Orientation: [{sensor_data['imu_orientation'][0]:.3f}, "
                                f"{sensor_data['imu_orientation'][1]:.3f}, "
                                f"{sensor_data['imu_orientation'][2]:.3f}, "
                                f"{sensor_data['imu_orientation'][3]:.3f}]")
                            print(
                                f"IMU Acceleration: [{sensor_data['imu_linear_acceleration'][0]:.3f}, "
                                f"{sensor_data['imu_linear_acceleration'][1]:.3f}, "
                                f"{sensor_data['imu_linear_acceleration'][2]:.3f}]"
                            )
                            print("-" * 50)
                    # 添加虚拟力
                    self.add_virtual_force()
                    # 控制命令写入
                    self.set_motor_cmd()
                    # 前进仿真一步
                    if not self.paused:
                        mujoco.mj_step(self.model, self.data)
                    else:
                        mujoco.mj_forward(self.model, self.data)
                time_step += 1

                # 控制仿真速度: 绝对截止时间步进。sleep 只睡到剩余 0.3ms,
                # 其余自旋等待, 消除 time.sleep 的定时器余量(~60us/次);
                # deadline 按固定周期累加, 单次超支会在后续步补偿回来,
                # 不会像相对 step_start 步进那样无限漂移。
                next_step_time += self.model.opt.timestep
                now = time.perf_counter()
                if next_step_time < now - 0.01:
                    next_step_time = now  # 落后过多(查看器卡顿等)时重同步, 不补跑
                sleep_until(next_step_time)

                # 控制仿真速度
                # time_until_next_step = self.model.opt.timestep - (time.time() -
                #                                                   step_start)
                # print(f'sim timestep: {time_until_next_step}')
                # if time_until_next_step > 0:
                #     time.sleep(time_until_next_step)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.thread_error = e
            self.stop_event.set()
            print(f"Simulation thread error: {e}")
        finally:
            print("Simulation thread stopped")

    def add_virtual_force(self):
        if self.elastic_band.enable:
            self.data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
            self.data.qpos[0:3], self.data.qvel[0:3]
        )
        else:
            # 禁用时必须清零, 否则最后一次的拉力会永久残留在 xfrc_applied
            self.data.xfrc_applied[self.band_attached_link, :3] = 0.0

    def plot_sensor_data(self):
        """绘制传感器数据"""
        if len(self.time_history) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Sensor Data')

        time_array = np.array(list(self.time_history))

        # 绘制关节位置
        if len(self.joint_pos_history) > 0:
            pos_array = np.array(list(self.joint_pos_history))
            axes[0, 0].plot(time_array, pos_array[:, 0], label='Joint 0')
            axes[0, 0].plot(time_array, pos_array[:, 1], label='Joint 1')
            axes[0, 0].set_title('Joint Positions')
            axes[0, 0].set_xlabel('Time (s)')
            axes[0, 0].set_ylabel('Position (rad)')
            axes[0, 0].legend()
            axes[0, 0].grid(True)

        # 绘制关节速度
        if len(self.joint_vel_history) > 0:
            vel_array = np.array(list(self.joint_vel_history))
            axes[0, 1].plot(time_array, vel_array[:, 0], label='Joint 0')
            axes[0, 1].plot(time_array, vel_array[:, 1], label='Joint 1')
            axes[0, 1].set_title('Joint Velocities')
            axes[0, 1].set_xlabel('Time (s)')
            axes[0, 1].set_ylabel('Velocity (rad/s)')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

        # 绘制IMU方向
        if len(self.imu_orient_history) > 0:
            orient_array = np.array(list(self.imu_orient_history))
            axes[1, 0].plot(time_array, orient_array[:, 0], label='w')
            axes[1, 0].plot(time_array, orient_array[:, 1], label='x')
            axes[1, 0].plot(time_array, orient_array[:, 2], label='y')
            axes[1, 0].plot(time_array, orient_array[:, 3], label='z')
            axes[1, 0].set_title('IMU Orientation (Quaternion)')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Quaternion')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # # 绘制IMU加速度
        # if len(self.imu_accel_history) > 0:
        #     accel_array = np.array(list(self.imu_accel_history))
        #     axes[1, 1].plot(time_array, accel_array[:, 0], label='X')
        #     axes[1, 1].plot(time_array, accel_array[:, 1], label='Y')
        #     axes[1, 1].plot(time_array, accel_array[:, 2], label='Z')
        #     axes[1, 1].set_title('IMU Linear Acceleration')
        #     axes[1, 1].set_xlabel('Time (s)')
        #     axes[1, 1].set_ylabel('Acceleration (m/s²)')
        #     axes[1, 1].legend()
        #     axes[1, 1].grid(True)

        # 绘制IMU位置
        if len(self.imu_pos_history) > 0:
            pos_array = np.array(list(self.imu_pos_history))
            axes[1, 1].plot(time_array, pos_array[:, 0], label='X')
            axes[1, 1].plot(time_array, pos_array[:, 1], label='Y')
            axes[1, 1].plot(time_array, pos_array[:, 2], label='Z')
            axes[1, 1].set_title('IMU Linear Position')
            axes[1, 1].set_xlabel('Time (s)')
            axes[1, 1].set_ylabel('Position (m)')
            axes[1, 1].legend()
            axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def start(self, duration=None):
        """运行到窗口关闭、Ctrl+C 或 duration 秒墙钟时间结束。"""
        threads = [threading.Thread(target=self.simulate_thread, daemon=True)]
        if self.viewer is not None:
            threads.append(threading.Thread(target=self.viewer_thread, daemon=True))
        started_threads = []
        deadline = None if duration is None else time.perf_counter() + duration
        try:
            if self.viewer is not None:
                self.start_keyboard_listener()
            for thread in threads:
                thread.start()
                started_threads.append(thread)
            self.start_publishing()
            while self.is_running() and all(t.is_alive() for t in threads):
                if deadline is not None and time.perf_counter() >= deadline:
                    break
                time.sleep(0.05)
        except KeyboardInterrupt:
            print("\nReceived KeyboardInterrupt, stopping simulation...")
        finally:
            self.stop_event.set()
            for thread in started_threads:
                thread.join(timeout=2.0)
            self.stop_publishing()
            # 深度上下文先于 viewer 关闭。
            if self.depth_renderer is not None:
                self.depth_renderer.close()
            if self.viewer is not None:
                self.viewer.close()
            if self.debug:
                self.plot_sensor_data()
            print(f"Simulation finished (simulated {self.data.time:.3f} s)")
        if self.thread_error is not None:
            raise RuntimeError("Simulation worker failed") from self.thread_error

    def set_keyframe(self, key_name):
        """
        将机器人设置为指定的关键帧姿态
        
        参数:
            key_name: 关键帧名称 (例如 "zero", "zero_height", "zero_standup")
        """
        # 查找关键帧的索引
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY,
                                   key_name)

        if key_id == -1:
            print(f"关键帧 '{key_name}' 未找到")
            return False
        # # # 将关键帧的qpos值复制到当前状态
        self.data.qpos[:] = self.model.key_qpos[key_id]

        # 将关键帧的ctrl值复制到控制向量
        if self.model.key_ctrl is not None:
            self.data.ctrl[:] = self.model.key_ctrl[key_id]

        mujoco.mj_forward(self.model, self.data)

        # 更新系统状态
        # mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        return True

    def get_available_keyframes(self):
        """
        获取所有可用的关键帧名称
        """
        keyframes = []
        for i in range(self.model.nkey):
            # 获取关键帧名称
            key_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_KEY,
                                         i)
            if key_name:
                keyframes.append(key_name)
        return keyframes

def _spin_wrapper(node, stop_event):
    """ROS2 spin wrapper for graceful shutdown."""
    try:
        while rclpy.ok() and not stop_event.is_set():
            rclpy.spin_once(node, timeout_sec=0.001)
    except Exception as e:
        if rclpy.ok() and not stop_event.is_set():
            print(f"Error in ROS2 spin thread: {e}")
    finally:
        print("ROS2 spin thread terminated.")


def main():
    parser = argparse.ArgumentParser(description="xmujoco ROS2 异步仿真器")
    parser.add_argument('--robot', choices=('dex', 'tiangong3'), default='tiangong3',
                        help="内置 tiangong3；dex 需要通过 --model 指定外部 XML")
    parser.add_argument('--model', help="自定义 XML 路径，优先于 --robot")
    parser.add_argument('--headless', action='store_true', help="不打开窗口或监听桌面键盘")
    parser.add_argument('--duration', type=float, help="运行指定墙钟秒数后退出")
    parser.add_argument('--no-band', action='store_true', help="关闭默认启用的虚拟吊带")
    parser.add_argument('--pd-mode', choices=('auto', 'implicit', 'explicit'), default='auto',
                        help='auto: 纯力矩模型使用隐式 PD，DEX 使用最新反馈的显式 PD')
    args = parser.parse_args(rclpy.utilities.remove_ros_args(args=sys.argv)[1:])
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error('--duration 必须是有限正数')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.robot == 'dex' and not args.model:
        parser.error('本仓库仅内置 Tiangong3；--robot dex 需要 --model /path/to/dex.xml')
    model_path = os.path.abspath(os.path.expanduser(args.model)) if args.model else os.path.abspath(
        os.path.join(script_dir, '..', 'assets', 'tiangong3', 'tiangong3_29dof_dex_armature.xml'))
    print(f"MuJoCo model: {model_path}")
    rclpy.init()
    node = None
    spin_thread = None
    spin_stop = threading.Event()
    try:
        node = rclpy.create_node(f'mujoco_simulator_{args.robot}')
        simulator = RobotSimulator(model_path, node, debug=False, depth=False,
                                   headless=args.headless, pd_mode=args.pd_mode)
        simulator.elastic_band.enable = not args.no_band
        spin_thread = threading.Thread(target=_spin_wrapper,
                                       args=(node, spin_stop), daemon=True,
                                       name="ROS2_Spin_Thread")
        spin_thread.start()
        simulator.start(duration=args.duration)
    finally:
        spin_stop.set()
        if spin_thread is not None:
            spin_thread.join(timeout=2.0)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
