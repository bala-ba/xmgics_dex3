# STAND 原地站立控制

`STAND` 已加入 `config/dex_config.yaml` 的 `enabled_policy_states`。
使用现有 `model/policy_msl.onnx`，100 Hz 持续推理，根据关节状态、IMU 角速度和
投影重力调整双腿与腰部的 15 个关节目标；手臂保持 `config/stand.yaml` 中的姿态。
零速度指令下仍持续推理，不会退化成固定目标 PD。

## 操作

重启控制节点后，在控制终端按键，无需回车：

| 按键 | 行为 |
| --- | --- |
| `z` | 进入 ZERO，准备初始姿态 |
| `1`（数字一） | 从 ZERO 进入 STAND，并清零前进、横向和转向速度 |
| 在 STAND 中再次按 `1` | 清零速度，保持 STAND 推理及历史 |
| `r` | 清零速度，保持当前状态 |
| `p` | 直接进入 InstantTrack，立即从首帧播放 |
| `z` / `c` | 退出 STAND，进入 ZERO / STOP |

先在吊带支撑下 `z`，姿态稳定后按 `1`；确认输出 `[FSMStateSTAND] enter` 后，
等待约 2 秒，再在 MuJoCo 窗口按空格解除吊带。
进入 ZERO/STOP 后不再运行站立策略，它们是关节姿态 PD 状态。

如使用 WebSocket 控制端，`enable_standing_policy` 已映射到 `gotoSTAND`，
对应控制页面的站立按钮；进入时同样清零速度。

STAND 内直接按 `p` 可立即进入 InstantTrack，从首帧开始播放，
无需先按 `b`，也不会停留在首帧 READY。推荐顺序为 `z → 1 → p`。
`b` 会持续运行 policy，经过 2 秒准备且姿态检查通过后自动播放，无需再按 `p`。
准备期间按 `i` 可取消自动播放，改为准备后保持首帧；再按 `p` 可播放。
InstantTrack 冻结参考时存在水平漂移，原地站立仍使用 STAND。
详见 [InstantTrack 控制说明](../instant_track/README.md)。

## 启动示例

终端 1，Tiangong3 + DEX armature + 显式 PD：

```bash
cd /home/lzp/projects/xmujoco
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=71 ROS_LOCALHOST_ONLY=1 MUJOCO_GL=glfw
/home/lzp/projects/xmigcs_dex3/.venv/bin/python -u scripts/simulator_view_asyn.py \
  --robot tiangong3 --model urdf/tiangong3_29dof_dex_armature.xml --pd-mode explicit
```

终端 2：

```bash
cd /home/lzp/projects/xmigcs_dex3
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=71 ROS_LOCALHOST_ONLY=1
.venv/bin/python -u rl_control_node_sim.py
```

## 验证

2026-11-10，零速度命令；第 10 秒进入 STAND，第 12 秒解除吊带。
采用实际键盘映射、FSM、ONNX 和 xmujoco PD，物理 1000 Hz、控制及推理 100 Hz。

| 模型 | 总时长 | 最低根高度 | 最大机身倾斜 | 最大水平位移 |
| --- | ---: | ---: | ---: | ---: |
| Tiangong3 + DEX armature，显式 PD | 60 s | 0.963 m | 0.047 rad | 0.134 m |
| 原 DEX，显式 PD | 60 s | 0.919 m | 0.065 rad | 0.120 m |

两项均无摔倒或非有限状态；报告、轨迹位于 `.validation/stand/`。
另使用实际 ROS 发布/订阅与控制节点完成约 36 秒闭环测试：Tiangong3 + DEX
armature、显式 PD，第 12 秒解除吊带，最终保持 STAND，累计推理 2619 次，
最低根高度 0.962 m；报告为 `.validation/stand/ros_closed_loop.json`。
这覆盖上述模型与零速度站立条件，未测试有速度指令的行走、外部推扰或实机。

复现无 ROS 抖动的物理测试：

```bash
cd /home/lzp/projects/xmigcs_dex3
source /opt/ros/humble/setup.bash
.venv/bin/python scripts/check_stand_control.py
```

状态切换、持续推理、命令路由和异常处理回归：

```bash
.venv/bin/python -m unittest policy.stand.test_stand -v
```
