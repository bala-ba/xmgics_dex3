# Tiangong3 资产与旧 SONIC 策略动作生成

## 资产来源

部署所需的官方 URDF、39 个网格及 torque XML 已放入仓库的 `assets/tiangong3/`，
不再需要旁边的 `TienKung-Lab-dev` 项目。原始下载目录为
`/home/lzp/projects/TienKung-Lab-dev/legged_lab/assets/tiangong3`（仅来源记录）。

来源：<https://github.com/Open-X-Humanoid/TienKung-Lab/tree/dev/legged_lab/assets/tiangong3>

本次固定提交：`ffea8bde32fcb632b5ef2a700e452f99e907ea40`。
原始目录含 URDF、Xacro、MJCF、Python 配置及全部网格，未修改上游文件。

独立生成的 MuJoCo 资产：`assets/tiangong3/tiangong3_29dof.xml`。
对应 `.json` 记录源文件、提交、SHA-256、每个网格的校验和和转换约定。

转换以官方 `urdf/tiangong3.urdf` 为机械参数来源，保留全部 link 的质量、质心、
完整惯量矩阵、关节安装变换、轴、硬限位、URDF 碰撞几何和视觉网格。
解锁双侧 `elbow_yaw`、`wrist_pitch`、`wrist_roll` 共 6 个关节；头部 yaw/pitch
保持固定且保留质量。模型为浮动基座 + 29 关节：`nq=36, nv=35, nu=29`，总质量 63.272147 kg。

关节阻尼和摩擦来自官方 `tiangong3_torq.xml`；未声明的 armature 为零。
力矩上限采用 Tiangong3 URDF，关闭自碰撞与官方 `tiangong3.py` 一致。
生成器没有直接使用官方 MJCF 的惯量表示：其 pelvis 未显式填写惯量，其他 link 使用
对角惯量，因此从 URDF 保留完整惯量来避免丢失机械参数。
FK 已与 MuJoCo 原生 URDF 导入器在零位及随机姿态下对照验证。

## 策略和动作

继续使用 `policy/sonic_track/config/sonic_track.yaml` 指定的旧导出：

`policy/sonic_track/model/export_100000_20261103_111558`

encoder、decoder、manifest、PD、动作缩放、观测和历史定义均未修改，也没有重新训练。
IDLE 为 `idle_loop_001__A028.pkl`；当前播放动作是
`walk_forward_loop_001__A022.pkl`。二者均位于 `policy/sonic_track/motions/`。
原始动作参考未重新重定向到 Tiangong3；这是旧策略与旧参考的跨机器人迁移实验。

默认运行过程：1 秒准备，自动跟踪 IDLE，6 秒时 PLAY，动作结束后 RETURNING → IDLE，
再运行 5 秒。仅前 2 秒启用与旧验证相同的虚拟吊绳，之后无外部支撑。
只在初始化时把足底放到地面；运行中不覆盖机器人的 qpos 或根位姿。

默认频率：MuJoCo 和 PD 内环 **1 kHz**，命令更新 **100 Hz**，SONIC 推理 **50 Hz**。
目标在推理间保持；没有新增动作线性插值。PD 使用每个物理步的最新位置/速度。
策略目标额外限制在 Tiangong3 硬限位内，力矩按 Tiangong3 执行器上限裁剪，
这些执行器约束不会改写网络上一拍动作的历史。

## 运行

离线生成视频和轨迹，不需要 DISPLAY，也不会连接 ROS 控制话题：

默认直接加载内置 XML。需要重新构建时可指定 `--source assets/tiangong3`；
通常的交互式仿真请使用 [SIMULATION.md](SIMULATION.md) 中的 DEX armature 配置。

```bash
cd /home/lzp/projects/xmigcs_dex3
./scripts/run_tiangong3_sonic.sh
```

在有 DISPLAY 的桌面终端打开实时 MuJoCo 窗口：

```bash
./scripts/run_tiangong3_sonic.sh --viewer --no-video --output outputs/tiangong3_live
```

60 秒 IDLE 检查：

```bash
./scripts/run_tiangong3_sonic.sh --idle-only --no-video --output outputs/tiangong3_idle_fresh
```

修改动作后重跑：编辑 `policy/sonic_track/config/sonic_track.yaml` 的 `motion_files`，
或用 `--config /path/to/sonic_track.yaml` 指定另一份配置。
`--output` 可指定新的输出目录；重用相同目录会覆盖对应产物。

本入口独立运行 MuJoCo，不需要同时启动 `xmujoco` 或 `rl_control_node_sim.py`。
ROS + xmujoco 入口也可通过 `simulator_view_asyn.py --robot tiangong3` 加载
`xmujoco/urdf/tiangong3_29dof.xml`；不指定模型时仍使用原 DEX。
Tiangong3 默认使用隐式 PD，修复轻关节在显式阻尼及旧传感器缓存反馈下的数值振荡。
启动说明和闭环验证条件见 `../xmujoco/TIANGONG3.md`；
`scripts/check_tiangong3_control.py` 可复现 ZERO → InstantTrack → 释放吊带的持续控制检查。

## 产物

默认位于 `outputs/tiangong3_sonic_step100000/`：

- `simulation.mp4`：左侧实际物理仿真，右侧运动学参考；相机分别跟随骨盆，HUD 保留实际根位置。
- `trajectory.npz`：100 Hz 的实际 qpos/qvel、目标、参考、网络 action、最后一个物理步力矩与 phase。
  `raw_action_policy` 按 policy 顺序；带 `_serial` 的字段按 `serial_joint_names` 顺序。
- `actual_motion.csv`：时间、根位置、根四元数 wxyz、29 个实际关节角，100 Hz。
- `generated_motion.pkl`：实际仿真轨迹降采样到 50 Hz 的 MotionLib 格式；
  `dof` 按导出 mapping 的 `mjcf` 顺序，`root_rot` 为 xyzw；metadata 标明机器人为 Tiangong3。
  包含准备、IDLE、动作和返回全过程，并非原参考文件的复制。
- `report.json`：条件、模型来源、频率、动作事件、误差、漂移和停止原因。

## 本次结果与限制

2026-11-02，原 step 100000 策略：

| 场景 | 仿真时长 | 最低根高度 | 关节 RMSE | 最大根 XY 位移 |
|---|---:|---:|---:|---:|
| IDLE，1 kHz 最新反馈 PD | 60.00 s | 0.936 m | 0.329 rad | 6.777 m |
| 双手动作及返回，1 kHz 最新反馈 PD | 21.38 s | 0.881 m | 0.283 rad | 1.552 m |

两场景均完成且没有触发跌倒/非有限/超速停止，但**存在明显位移漂移和跟踪误差**。
报告里的 `passed` 仅表示该场景完成并满足这些基本运行检查，不表示精确跟踪或实机可用。
策略未使用根位置反馈，视频跟随相机不代表机器人保持在原地。

复刻旧仿真的 100 Hz 缓存反馈 PD（`--pd-feedback cached`）时，Tiangong3
在约 2.50 秒跌倒；准备阶段就出现明显关节振荡。新资产的 armature 与旧模型不同，
因此默认使用更高频的内环反馈，而没有为使实验通过而复制旧模型的转子惯量。
失败记录保存在 `outputs/tiangong3_idle/`。

本次没有 ROS/DDS 通信时延，也没有验证实际电机内环；完整动作运行不等于已完成
Tiangong3 的策略适配。提高跟踪质量需要进一步校准执行器模型、适配参考动作并评估微调。

机械转换回归检查：

```bash
source /opt/ros/humble/setup.bash
.venv/bin/python -m unittest policy.sonic_track.test_tiangong3_asset -v
```
