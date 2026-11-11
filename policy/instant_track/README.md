# INSTANTTRACK：Tiangong 29 关节参考动作跟踪

已接入 `/home/lzp/projects/instantTrack/logs/tiangong_single_20261109_004112/ckpt_438000.pt`，
使用 checkpoint 的 `pi` 权重（训练保存的 EMA 策略），当前默认参考动作是
`run1_subject2_first_forward_landing_end`，采样率为 50 Hz，共 154 帧、3.06 秒：
2.2 秒原跑步片段、0.86 秒落腿收尾，落地后的参考帧已删除。

完整部署包已随仓库提交，包含 `policy.onnx`、参考动作 NPZ、manifest 和验证数据。
修改 `config/instant_track.yaml` 的 `export_dir` 后重启控制程序即可切换；以下目录均位于
`policy/instant_track/model/`：

| 目录 | 参考动作 | 帧数 | 时长 |
| --- | --- | ---: | ---: |
| `ckpt_438000_squat_001__A362` | 下蹲 | 494 | 9.86 秒 |
| `ckpt_438000` | 第一段直行，保留起步和停步 | 601 | 12 秒 |
| `ckpt_438000_jump_and_land_light_001__A001_M` | 轻跳落地 | 485 | 9.68 秒 |
| `ckpt_438000_salute_002__A067` | 敬礼 | 122 | 2.42 秒 |
| `ckpt_438000_full_backup_before_first_forward` | 完整 `walk1_subject1` 原始导出备份 | 13,066 | 261.3 秒 |
| `ckpt_438000_run1_subject2_first_forward` | `run1_subject2.csv` 首次向前移动，裁至摆动腿落地（当前默认） | 154 | 3.06 秒 |

新增跑步片段的裁剪 CSV 位于 `policy/instant_track/motions/run1_subject2_first_forward.csv`，
保留原始第 336–402 帧（0 起始、含首尾），仅把时间戳归零；部署包通过训练 MotionLib
重采样到 50 Hz，并重新计算速度和未来参考。`source_trim.json` 记录原文件、SHA-256 和裁剪范围。
该包继续使用同一 `ckpt_438000` 权重，ONNX 与原包一致。

当前加载的是 `motions/run1_subject2_first_forward_landing_end.csv`。它保留原
`run1_subject2_first_forward_stand_end.csv` 的第 0–92 帧（30 Hz、含首尾），
原姿态不变；在右侧摆动腿的参考足底接近地面后截断，删除后续收尾帧和 2 秒站姿停留。
截点由训练机器人正运动学确认：右脚各接触几何的下表面均距地面不足 5 mm。
CSV 时长为 3.0667 秒，训练 MotionLib 重采样后实际播放 3.06 秒；速度、未来参考和
验证向量全部重新生成。`landing_trim.json` 记录此次裁剪；`standing_tail.json` 仅记录
之前完整收尾的生成来源，不表示当前动作仍含完整 1 秒收尾和 2 秒停留。
播放结束仍由 InstantTrack 在 FINISHED 持续跟踪静止末帧，不切换 FSM 的 STAND 状态。

2026-11-11 使用 Tiangong3 + DEX armature、显式 PD 的实际 ROS 闭环测试：
`z → 1 → b`，第 10 秒移除吊绳，第 14 秒进入准备后自动播放，154 帧播放完成且未摔倒。
结束后继续控制 10.90 秒，水平位移 0.0942 m，最后 1 秒平均水平速度 0.00057 m/s。
作为对照，原始 2.2 秒片段停在跑步姿态，结束后 9.78 秒漂移约 1.58 m。
原始 2.2 秒导出包备份在本地 `outputs/run1_subject2_stand_end_20261111/original_bundle/`；
此次修改前的完整 5.2 秒包备份在 `outputs/run1_subject2_landing_end_20261111/full_stand_end_bundle/`。
当前测试报告和视频位于 `outputs/run1_subject2_landing_end_20261111/`；这些输出不属于部署必需文件。
该片段首帧与站立姿态差异较大，使用 `b` 准备，不从 STAND 直接按 `p`。

manifest 中的训练 checkpoint、原始动作和裁剪脚本路径用于记录来源；
部署加载使用包内文件，不要求这些原始路径存在。

这是独立的 `INSTANTTRACK` 状态（枚举 17），原 SONICTRACK 与其他策略保持可用。
`config/dex_config.yaml` 的 `instant_track.enabled: true` 注册此状态；腰、臂控制白名单已加入。
部署运行依赖现有 NumPy、SciPy、ONNX Runtime，不导入 PyTorch、mjlab 或训练仓库。

## 启动与控制

先启动现有 xmujoco 仿真，再在本项目启动控制节点，ROS 域与仿真保持一致：

```bash
cd /home/lzp/projects/xmigcs_dex3
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=71
export ROS_LOCALHOST_ONLY=1
.venv/bin/python rl_control_node_sim.py
```

在控制终端按键，不需要回车：

| 按键 | 行为 |
| --- | --- |
| `z` | 进入 ZERO |
| `b` | 从 ZERO 或 STAND 进入 INSTANTTRACK，持续推理，2 秒准备完成且姿态检查通过后自动播放 |
| `p` | STAND 中立即从首帧播放；READY 开始播放；HOLD 继续播放 |
| `i` | 播放中暂停参考；准备中取消自动播放；policy 持续运行 |
| `b` | READY/HOLD/FINISHED 时重新准备并自动播放；PREPARING 中重新启用自动播放但不重置计时；PLAYING 中不重启动作 |
| `[` / `]` | 未播放时选择上一条/下一条参考，并重新准备；默认只导出一条 |
| `c` | 进入现有 STOP 状态 |
| `z` | 退出跟踪，回到 ZERO |

当前跑步包推荐 `z → 1 → b`，其中 `1` 进入 STAND，`b` 准备后自动播放。
对于首帧接近站姿的其他动作，支持 `z → 1 → p`。STAND 中按一次 `p`，
在当前控制周期直接进入 PLAYING，运行第 0 帧策略推理并使用策略 Kp/Kd。
首帧插值阶段及其时长配置已删除，无需先按 `b`，也不会停留在 READY。
观测历史用当前反馈初始化，previous action 由上一状态的关节目标换算，随后以 50 Hz 持续推理。
重复/长按 `p` 不重置播放进度，旧的播放指令不会在重新进入 STAND 后自动触发。
进入时仍检查腿/腰姿态及机身倾斜，超出准入阈值时由主节点进入 STOP。
`i` 暂停参考推进，`z` / `c` 可随时退出。

也支持 `z → b` 或 `z → 1 → b`：只按一次 `b` 即可完成准备并自动播放。
按 `b` 后即以 50 Hz 运行 InstantTrack，包括 PREPARING 阶段；参考关节姿态、机身姿态和
Kp/Kd 在 2 秒内从当前反馈/增益平滑过渡到首帧。准备完成且姿态检查通过后直接进入
PLAYING，从第 0 帧开始推进；保留观测历史、上一拍 action 和推理时钟，无需手动 `p`。
准备期间按 `i` 可取消自动播放，此时准备完成后进入 READY，保持首帧并持续推理。
在准备期间再按 `p` 或 `b` 可重新启用自动播放，计时不会重置。

准备完成及实际起播均检查双腿/腰部相对首帧的
最大关节误差及机身倾斜误差，两者默认阈值均为 0.35 rad。手臂的策略残差不再被旧的
全关节姿态 PD 准入检查误判。未通过时保持 PREPARING 并持续运行 policy，
待检查通过后自动开始播放；手动 READY → `p` 的检查不通过时仍需再次按 `p`。
参考 yaw 在进入/重新准备时对齐当前机身 yaw，之后固定；已在 InstantTrack 内按 `p` 不再重置。

动作结束进入 FINISHED，policy 继续跟踪静止的末帧，不自动循环跳回第一帧；按 `b` 重新准备后自动重播。
READY/HOLD/FINISHED 的未来参考由同一姿态重复 10 次构造，参考位移、线/角速度和关节速度设为零。
PLAYING 才使用导出包的原始未来参考并推进帧号。暂停时仍持续推理，但冻结步行中的单脚支撑姿态
不保证动态稳定；当前 checkpoint 在 READY 中也存在水平漂移，不能替代 STAND 的原地站立控制。
STOP 同样沿用本项目原有保持逻辑，并非电机断电。

实机仍使用 `rl_control_node.py` 和 `sim: false`，经现有 ROS 话题及串并联踝转换下发。
本次保留当前 `sim: true` 配置，未连接实机。参考首帧包含抬臂姿态。
当前准备阶段持续运行 policy，已验证 Tiangong3 + DEX armature、显式 PD 下
从无吊带 STAND 切入、保持再播放；实机并联踝动力学尚未验证。

## 数据与控制约定

配置位于 `config/instant_track.yaml`。`export_dir` 相对本项目根目录解析，也支持绝对路径。
当前导出文件在 `model/ckpt_438000_run1_subject2_first_forward/`，迁移到另一台机器时需要复制整个目录：

- `policy.onnx`：包含观测/参考 RunningNorm、编码器、解码器及参考动作残差连接。
- `motion_000.npz`：原训练预处理生成的位置、速度、根姿态和未来参考特征。
- `manifest.json`：checkpoint、源数据和代码校验值，关节顺序、默认角、增益、缩放及力矩上限。
- `validation.npz`：原训练 MDP 观测及 PyTorch 输出验证样本。

完整部署包在 `model/` 中随项目管理。只复制 Python 源码不能运行此策略，
需要一起复制选定目录内的模型、动作、manifest 和验证向量。

观测输入为 `(1, 10, 99)`：历史从旧到新，每帧依次为：

| 字段 | 维度 | 约定 |
| --- | --- | --- |
| 关节位置偏差 | 29 | `q - q_default` |
| 关节速度 | 29 | 原始 rad/s，无额外 0.1 缩放 |
| 机体角速度 | 3 | 机体坐标系，无额外 0.25 缩放 |
| 投影重力 | 3 | `-R[2, :]` |
| 上一拍 action | 29 | ONNX 原始输出，目标限幅前 |
| 参考相对姿态 | 6 | `R_robot.T @ R_ref` 的前两列，按行展平 |

PLAYING 的未来参考输入为 `(1, 730)`，采用训练代码 `MotionLib.goal()`，参考当前帧之后的
第 1～10 帧，间隔 0.02 秒。参考相对位置、姿态、局部速度及关节状态均由原训练代码生成；
不是 SONIC 的 1751/930 维接口。观测只依赖关节反馈和 IMU，不需要根位置真值或相机。

目标计算为 `q_default + action_scale * action`，29 关节 `action_scale=0.25`。
参考动作残差已包含在 ONNX 中，部署端不再次加参考角，也不添加训练中不存在的 action clip。
所有关节按名称映射到 RobotData 串联顺序；目标速度、前馈力矩均为零。

Kp/Kd/力矩上限使用训练机器人配置导出的 nominal 值，非网络实时输出。
默认开启与 DEXAMP 相同的反馈 PD 力矩约束，每个 100 Hz 控制周期裁剪位置目标：

```text
q_lower = q + (Kd * dq - torque_limit) / Kp
q_upper = q + (Kd * dq + torque_limit) / Kp
q_target = clip(q_target, q_lower, q_upper)
```

这是部署端的目标约束；训练环境在物理子步内限制执行器力矩，两者反馈频率不同。
`control.torque_limit_enabled: false` 可关闭这层约束以复现原始网络目标。
固定关节角度限位仍受全局 `robot_interface.clip_actions` 控制，当前为 false。
实机四个踝电机最终 Kp/Kd 仍由全局 `ankle_kp_p / ankle_kd_p` 覆盖为 15 / 1.25；
串联策略增益用于踝力矩转换。推理或反馈出现非有限值时由主控制节点切换 STOP。

命令发布 100 Hz，推理、观测历史和参考时钟严格 50 Hz。没有 ROS `/clock` 同步；
不要仅暂停外部 MuJoCo 而让控制节点继续运行。

## 重新导出或添加参考动作

导出使用训练项目的 Python 环境；部署仍使用本项目 `.venv`：

```bash
cd /home/lzp/projects/xmigcs_dex3
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/xmigcs-mpl \
../instantTrack/.venv/bin/python scripts/export_instant_track.py \
  --checkpoint ../instantTrack/logs/tiangong_single_20261109_004112/ckpt_438000.pt \
  --motion ../assets/data/lafan1/tiangong2dex_jointVel/walk1_subject1.csv \
  --output policy/instant_track/model/ckpt_438000 \
  --validation-scene .validation/instant_track/training_scene.xml
```

重复 `--motion /path/to/clip.csv` 可导出多个参考。也支持训练机器人 loader 接受的 PKL。
导出时读取 checkpoint 架构和 `pi`，严格加载参数，核对残差默认角/缩放，并校验
ONNX 与 PyTorch 输出。参考预处理保留训练的重采样、中心差分速度及未来帧截断逻辑。
需要切换模型/参考时先停止控制节点，导出完成后重启。

`--validation-scene` 可省略；它只生成离线验证场景，XML 的 meshdir 指向本机训练资产。

## 离线验证

```bash
source /opt/ros/humble/setup.bash
.venv/bin/python -m unittest policy.instant_track.test_instant_track \
  policy.sonic_track.test_sonic_track test_sim_adaptation -v

.venv/bin/python -m policy.instant_track.check_mujoco --start reference --seconds 262 \
  --output .validation/instant_track/reference_full.json
.venv/bin/python -m policy.instant_track.check_mujoco --start prepare --support-orientation --seconds 17 \
  --output .validation/instant_track/supported_prepare_17s.json
```

最初接入时的检查结果（旧版姿态 PD 准备/暂停逻辑）：

- ONNX 对 PyTorch 最大绝对输出误差 `2.861e-6`。
- 新策略测试覆盖原始 MDP 观测、旋转布局、真实权重推理、50 Hz 调度、暂停/续播/结束、
  起播姿态检查、位置目标力矩约束、非有限值、FSM 及 29 个 ROS 电机消息。
- 从参考首帧的位姿及速度初始化，完整播放 261.3 秒，13,065 次推理后 FINISHED；
  未触发训练中的高度差 0.25 m / 倾斜指标差 0.8 终止条件，关节 RMSE 约 0.1095 rad。
- 从 ZERO 开始，前 2.01 秒提供竖直及姿态支撑，随后无支撑跟踪 15 秒，通过同一终止检查。
- 无姿态支撑的准备测试出现倾斜，记录保留在 `.validation/instant_track/prepare_17s.json`；
  后续新增的起播检查会拒绝这种姿态开始播放。

场景使用训练机器人 nominal 参数、CPU MuJoCo 3.12、200 Hz 原生位置 PD、100 Hz 部署
命令及 50 Hz 推理。无域随机化、无 ROS 时延、无实机并联踝动力学。
这些结果未覆盖外部 xmujoco 资产差异和实机运动。

## 2026-11-10：持续推理及 STAND 直接切换

实际 xmujoco Tiangong3 + DEX armature、显式 PD，1 kHz 物理、100 Hz 命令、50 Hz InstantTrack：
第 8 秒进入 STAND，第 10 秒解除吊带，第 14 秒按 `b`，第 30 秒按 `p`。
完整运行 60 秒无摔倒，最低根高度 0.929 m、最大关节速度 9.53 rad/s，
InstantTrack 共推理 2300 次，最终 PLAYING/帧号 1500。
READY 期间帧号保持 0，但机身约 14 秒内水平漂移 2.66 m；这是步行模型冻结参考的局限，
不能将该结果理解为原地位置锁定。报告与轨迹：`.validation/instant_track_hold/stand_hold_play_final.*`。

实际 ROS 发布/订阅闭环运行约 55 秒，同样经过 STAND → `b` → `p`，最低根高度 0.928 m，
最终保持 PLAYING。READY 期间帧号始终为 0，并新增 698 次推理；完整 InstantTrack 段共
2057 次推理。报告：`.validation/instant_track_hold/ros_closed_loop.json`。
InstantTrack、STAND、SONIC 及接口共 37 项回归测试通过。

复现此切换测试：

```bash
source /opt/ros/humble/setup.bash
.venv/bin/python scripts/check_tiangong3_control.py \
  --model /home/lzp/projects/xmujoco/urdf/tiangong3_29dof_dex_armature.xml \
  --pd-mode explicit --stand-at 8 --prepare-at 14 --band-seconds 10 \
  --play-at 30 --seconds 60 --output .validation/instant_track_hold/recheck.json
```

加 `--b-only` 可验证只按 `b` 自动播放；加 `--hold-only` 则模拟 `b` 后按 `i` 取消自动播放，
用于验证首帧保持。这两个选项都不会发送手动 `p`。

## STAND 直接播放验证

当前无插值路径已通过 25 项 InstantTrack/STAND/接口回归测试，以及 35 秒无吊带仿真。
按 `p` 的当前控制周期即进入 PLAYING 并完成首次推理，无准备/等待阶段；
最低根高度 0.936 m、最大关节速度 10.78 rad/s，未摔倒。
报告：`.validation/stand_direct_play/physical_no_interpolation.json`。

以下为已删除插值功能的历史验证记录。

原 1 秒配置通过了 15 项 InstantTrack 回归测试和 35 秒无吊带切换仿真。
实测 STARTING 持续 1.000 秒后进入 PLAYING，最低根高度 0.929 m，未摔倒。
报告：`.validation/stand_direct_play/physical_1s.json`。

更早的 0.2 秒配置验证记录如下。

`z → 1 → p` 已通过 39 项回归测试，覆盖旧播放事件隔离、重复按键、0.2 秒插值、
自动起播、历史衔接、取消起播和异常姿态拒绝，以及已有 STAND/SONIC/接口行为。

Tiangong3 + DEX armature、显式 PD：第 8 秒进入 STAND、第 10 秒解除吊带、
第 14 秒按 `p`。60 秒物理仿真通过，实测 STARTING 持续 0.200 秒后直接 PLAYING，
全程未进入 READY，最低根高度 0.934 m、最大关节速度 10.79 rad/s。
实际 ROS 发布/订阅闭环约 55 秒通过，最终保持 PLAYING，最低根高度 0.928 m。
报告位于 `.validation/stand_direct_play/physical.json` 和 `ros_closed_loop.json`。

```bash
source /opt/ros/humble/setup.bash
.venv/bin/python scripts/check_tiangong3_control.py \
  --model /home/lzp/projects/xmujoco/urdf/tiangong3_29dof_dex_armature.xml \
  --pd-mode explicit --stand-at 8 --direct-play --play-at 14 \
  --band-seconds 10 --seconds 60 --output .validation/stand_direct_play/recheck.json
```

## b 准备后自动播放验证

26 项 InstantTrack/STAND/接口回归测试通过，包含自动起播、姿态检查等待、准备中取消/重新
启用自动播放、重复按键不重置，以及切换时保留历史和上一拍 action。
Tiangong3 + DEX armature、显式 PD 的两项测试均未发送手动 `p`：

- STAND → `b`：40 秒仿真，第 10 秒解除吊带、第 14 秒按 `b`，准备 2.000 秒后直接 PLAYING；
  最低根高度 0.926 m，未摔倒。
- ZERO → `b`：35 秒仿真，第 10 秒按 `b`，约 2.160 秒后通过姿态检查并自动起播，第 14 秒解除吊带；
  最低根高度 0.939 m，未摔倒。

报告：`.validation/instant_b_autoplay/from_stand.json` 和 `from_zero.json`。

```bash
source /opt/ros/humble/setup.bash
.venv/bin/python scripts/check_tiangong3_control.py \
  --model /home/lzp/projects/xmujoco/urdf/tiangong3_29dof_dex_armature.xml \
  --pd-mode explicit --stand-at 8 --prepare-at 14 --band-seconds 10 \
  --b-only --seconds 40 --output .validation/instant_b_autoplay/recheck.json
```
