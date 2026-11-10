# xmigcs_dex3 的 SONIC / MuJoCo 部署

环境安装、模型清单与传输完整性检查见 [DEPLOYMENT.md](DEPLOYMENT.md)。
以下命令在克隆或解压后的项目根目录执行，不要求固定的 `/home/lzp/projects` 路径。

## 当前配置

- 默认 `sim: true`、键盘控制、关闭深度图；控制命令 100 Hz，SONIC / InstantTrack 推理 50 Hz。
- SONIC 完整导出位于仓库内 `policy/sonic_track/model/export_100000_20261103_111558/`。
  IDLE 循环 `idle_loop_001__A028.pkl`，`p` 播放 `walk_forward_loop_001__A022.pkl`。
- InstantTrack 默认是 `ckpt_438000` 的第一段直行，其他四套动作包也已打包。
- STAND、DEXAMP、DEXDELTA、MULTIAMP 已启用并包含模型。
- WALKAMP 模型已补齐但默认禁用；WALKDEPTH / WALKDEPTH100HZ 缺少其指定模型，保持禁用。
- 内置 `simulation/simulator_view_asyn.py`，无需单独复制 xmujoco。
  默认模型 `assets/tiangong3/tiangong3_29dof_dex_armature.xml`，启动脚本使用显式 PD。

## 启动

同一 ROS 域只运行一个控制节点。先退出旧 xmigcs 控制端；如果仿真已经在运行，
保留仿真，只启动新的控制端即可。以下两终端均使用域 71。

终端 A：在有 DISPLAY 的桌面环境启动 MuJoCo。

```bash
export ROS_DOMAIN_ID=71
./scripts/run_mujoco_sim.sh

# 等价的完整启动方式：
# source /opt/ros/humble/setup.bash
# export ROS_DOMAIN_ID=71 ROS_LOCALHOST_ONLY=1 MUJOCO_GL=glfw
# .venv/bin/python -u simulation/simulator_view_asyn.py \
#   --robot tiangong3 \
#   --model assets/tiangong3/tiangong3_29dof_dex_armature.xml \
#   --pd-mode explicit
```

终端 B：启动控制节点。

```bash
export ROS_DOMAIN_ID=71
./scripts/run_sonic_sim.sh
```

脚本自动定位项目目录、加载 `${ROS_DISTRO:-humble}` 并使用本仓库的 `.venv`。
不设置 `ROS_DOMAIN_ID` 时默认 71，`ROS_LOCALHOST_ONLY` 默认 1。

在终端 B 输入按键，无需回车：

| 按键 | 行为 |
|---|---|
| `z` | 进入 ZERO |
| `1` | 从 ZERO 进入 STAND，清零速度并持续运行站立策略 |
| `b` | 从 ZERO/STAND 进入 InstantTrack，准备完成且姿态检查通过后自动播放 |
| `t` | 从 ZERO 进入 SONIC，准备后自动播放 IDLE LOOP |
| `p` | STAND 中立即从首帧播放 InstantTrack；跟踪状态内播放/续播 |
| `i` | InstantTrack 暂停参考但继续推理；SONIC 过渡回 IDLE LOOP |
| `[` / `]` | 选择上一条／下一条动作 |
| `c` | 退出跟踪，进入 STOP 保持状态 |

SONIC 动作结束后自动 RETURNING → IDLE。InstantTrack 结束后继续推理并跟踪静止末帧。
InstantTrack 冻结参考时可能水平漂移，原地站立使用 STAND；详细行为见
[InstantTrack 说明](policy/instant_track/README.md)。
虚拟吊绳仍由 MuJoCo 窗口的空格键控制，SONIC 不自动释放吊绳；测试无支撑站立时需移除吊绳。
控制器按固定控制周期推进参考时间，不与 `/clock` 锁步同步；不要只暂停仿真而让控制器继续运行。

终端 C：检查话题。

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=71
export ROS_LOCALHOST_ONLY=1
ros2 topic hz /leg/cmd_ctrl
# Ctrl+C 结束频率检查后：
ros2 topic echo /leg/status --once
```

修改模型或动作时编辑 `policy/sonic_track/config/sonic_track.yaml`，随后重启控制端。
路径相对本仓库解析；不要修改导出 manifest 的内容来调参。

## 验证

```bash
source /opt/ros/humble/setup.bash
.venv/bin/python scripts/check_deployment.py --ros
.venv/bin/python -m unittest policy.sonic_track.test_sonic_track \
  policy.instant_track.test_instant_track policy.stand.test_stand \
  policy.sonic_track.test_tiangong3_asset test_sim_adaptation -q
```

完整性检查和单元测试不连接机器人。无窗口检查仿真接口可运行：

```bash
./scripts/run_mujoco_sim.sh --headless --duration 3
```

主循环依次读取反馈、运行策略和发送命令。策略抛出 RuntimeError 或
FloatingPointError 时回退到 STOP 保持；STOP 并非电机断电。
实机仍需机器人侧驱动、标定和对应的网络环境，本次迁移验证不等同于实机动作验证。
