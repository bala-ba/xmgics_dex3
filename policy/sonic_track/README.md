# SONICTRACK：Tiangong2Dex SONIC 动作跟踪

通过现有 ROS2 `/leg/cmd_ctrl`、`/waist/cmd_ctrl`、`/arm/cmd_ctrl`
控制全部 29 个关节。沿用串行关节 `RobotData` 接口；仿真使用
`rl_control_node_sim.py`，不会执行实机并联踝变换。

## 模型和动作配置

编辑本目录 `config/sonic_track.yaml`。路径相对 **xmigcs_dex3 根目录**解析，也可使用绝对路径。

- `export_dir`：包含 encoder、decoder、manifest、robot_mapping、training_config 的完整导出目录。
  当前使用 `model/export_100000_20261103_111558`，对应 `model_step_100000.pt`。
  完整导出已随仓库提供，不需要训练项目或 PyTorch；启动时核对 SHA-256、关节集合和模型接口。
- `motion_files`：可信的 Tiangong 29 关节 MotionLib `.pkl` 文件列表。
  当前使用仓库内 `motions/walk_forward_loop_001__A022.pkl`（20.58 秒）。
  换动作只需修改此列表并重启节点，不需重新导出网络。
  文件中的 `dof` 必须采用导出 mapping 的 `mjcf` 顺序，`root_rot` 为 `xyzw`。
- `idle_motion_file`：IDLE 自动循环跟踪的动作，当前为 `idle_loop_001__A028.pkl`。
  与 `motion_files` 的选择相互独立，进入 IDLE 后无需按 `p`。
- `control.idle_loop_blend_seconds`：每次 IDLE 动作播完后，接回首帧的过渡时间，默认 1 秒。
  该源动作约 25.56 秒，首尾关节角有差异，因此用保持端点关节位置/速度连续的过渡闭合循环。
- 默认角、PD、动作缩放、软限位来自导出文件；不要编辑 manifest 来调参。
- 主循环 `dt=0.01`，SONIC `decimation=2`：命令发布 100 Hz，推理与历史更新 50 Hz。
  启动时检查两者乘积等于导出的 0.02 秒。

`config/dex_config.yaml` 的 `sonic_track.enabled: true` 注册此状态。
腰、臂状态白名单及字符串到枚举映射已加入 `SONICTRACK`。

## 启动和按键

先启动现有 xmujoco 仿真，另开控制终端：

```bash
cd /home/lzp/projects/xmigcs_dex3
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=71
export ROS_LOCALHOST_ONLY=1
.venv/bin/python rl_control_node_sim.py
```

ROS 域需要与仿真终端一致。已有控制节点需先退出，再启动更新后的节点。
新增依赖已安装在本机 `.venv` 中；其他环境按 `requirements_sim.txt` 安装。

在 **xmigcs_dex3 控制终端**输入（不需要回车）：

| 按键 | 行为 |
| --- | --- |
| `z` | 进入 ZERO |
| `t` | 从 ZERO 进入 SONICTRACK；1 秒姿态/PD 准备，再平滑接入自动播放的 IDLE 循环 |
| `p` | 出现 `READY / IDLE LOOP` 后，播放选中的参考动作 |
| `i` | 平滑接回 IDLE 动作首帧，随后持续循环，继续推理 |
| `[` / `]` | 选择上一条/下一条动作；不打断正在播放的动作 |
| `c` | 进入 STOP，终止跟踪 |
| `z` | 退出 SONICTRACK，回到 ZERO |

选中动作结束会自动过渡回 IDLE 循环。播放过程中再次按 `p` 不会重新开始；先按 `i`，
等待返回 IDLE 后再按 `p`。重复按 `t` 不清空历史或重置动作。
准备阶段的 `p` 会提示尚未就绪，不会排队自动播放。`w/a/s/d/q/e` 不作为 SONIC 输入。

IDLE 的参考会持续随时间更新，不再使用固定 `q_default` 作为站立参考。
初次进入时由默认姿态平滑衔接到循环首帧；RETURNING 衔接到循环首帧后自动继续播放。
未来 10 帧也包含循环首尾过渡和 RETURNING 后的动作，历史观测在这些边界不会重置。
READY 仍表示准备与参考衔接计时完成，并非已经通过物理稳定性检测。

当前 IDLE 播放站立循环，`p` 播放已打包的向前行走动作。
切换动作请修改 `motion_files`，`idle_motion_file` 独立控制待机循环。

虚拟吊绳由 xmujoco 管理，SONICTRACK 不会自动切换吊绳。准备完成后在 MuJoCo 窗口
用空格切换吊绳，确认移除支撑后再评估自主站立和跟踪。

## 推理约定

参考时刻为当前至未来 0.9 秒，共 10 帧、间隔 0.1 秒。Encoder 的位置/速度输入为
所有位置帧展平后拼接所有速度帧，不能逐帧交错。

Decoder 输入为 64 维 token + 930 维本体历史；历史按字段排列，每个字段内部从旧到新：
角速度 30、关节位置偏差 290、关节速度 290、上一拍动作 290、机体重力方向 30。
初始历史重复首帧，上一拍动作初始为零。播放/停止/切换参考不重建网络或清空历史。

`action = clip(raw, -20, 20)`，目标为 `q_default + 0.25 * action`，再按 mapping 软限位裁剪。
历史保留裁剪后的 action。反馈和输出按关节名在 IsaacLab 顺序与 xmigcs 顺序间转换；
所有 29 个目标角及 PD 均下发，目标速度与前馈力矩为零。

参考朝向在进入 SONICTRACK 时对齐当前 yaw，之后不追随机器人漂移。
只使用 IMU 姿态/机体角速度和关节反馈；不需要根节点位置真值或深度图。
非有限反馈或推理失败会抛出 RuntimeError，由现有控制节点切换到 STOP。

计时使用主循环的固定步长，与现有控制器一致。当前 xmujoco 是异步墙钟仿真，
此接入不新增 `/clock` 同步；不要只暂停 MuJoCo 而让控制器与参考时间继续前进。

## 验证

```bash
cd /home/lzp/projects/xmigcs_dex3
source /opt/ros/humble/setup.bash
.venv/bin/python -m unittest policy.sonic_track.test_sonic_track -v
```

测试覆盖已导出的 PyTorch 验证向量、观测排序、关节映射、50 Hz 调度、准备和播放切换、
自动 IDLE 循环、循环接缝、RETURNING 的未来帧与返回后的连续性、
文件损坏检测、非有限输入，以及腿/腰/臂实际 ROS 消息的电机 ID、目标角和 PD。
测试不连接机器人或运行中的 ROS 图。数值和接口验证不等同于实机运动验证。

`sonic_runtime.py` 的参考采样与历史实现移植自
`GR00T-WholeBodyControl/tools/tiangong/sim2sim/contract.py`；部署时不导入训练项目代码。
