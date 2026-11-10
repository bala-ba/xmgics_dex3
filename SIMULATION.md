# xmigcs_dex3 的 SONIC / MuJoCo 部署

本机已建立独立 `.venv`：Python 3.10.20、NumPy 1.24.4、ONNX Runtime 1.19.2、
MuJoCo 3.12.0、ROS 2 Humble，以及 `lib/22` 的 `sptlib_python`。
原 `/home/lzp/projects/xmigcs` 的代码和环境不受此适配影响。

## 当前配置

- 默认 `sim: true`、键盘控制、深度图关闭，主循环 100 Hz。
- SONIC 50 Hz 推理，非推理周期保持上次目标；不对网络动作做线性插值。
- 使用原 `model_step_100000.pt` 对应的完整导出 `ckpt_100000_8a3b22bf`，
  encoder、decoder、manifest、mapping 和训练配置仍位于旁边的 GR00T 项目中。
- IDLE 自动循环 `idle_loop_001__A028.pkl`。
- `p` 默认播放原 xmigcs 当前选择的
  `big_light_two_hands_front_medium_to_behind_medium_R_002__A521.pkl`。
- `STAND` 已启用，数字键 `1` 从 ZERO 进入，100 Hz 持续运行站立策略。
  操作和 Tiangong3/Dex 验证结果见 [STAND 说明](policy/stand/README.md)。
- `DEXAMP`、`DEXDELTA`、`MULTIAMP` 保留原 Dex 3.0 模型和 PD 限幅实现。
  `enabled_policy_states` 控制它们是否加载；可设为 `[]` 只运行 STOP/ZERO/SONIC。
- `WALKAMP`、`WALKDEPTH`、`WALKDEPTH100HZ` 默认不加载，因为仓库缺少其配置引用的模型。
  补齐模型和相应输入配置后才能加入 `enabled_policy_states`。

本次使用现有 `xmujoco/urdf/dex_evt_depth.xml`，没有更换机器人动力学资产。
Dex 3.0 的踝库保留用于实机；仿真路径不调用并联踝转换。
因此，本地仿真验证不代表已验证 Dex 3.0 实机动力学或硬件部署。

## 启动

同一 ROS 域只运行一个控制节点。先退出旧 xmigcs 控制端；如果仿真已经在运行，
保留仿真，只启动新的控制端即可。以下两终端均使用域 71。

终端 A：在有 DISPLAY 的桌面环境启动 MuJoCo。

```bash
cd /home/lzp/projects/xmujoco
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=71
export ROS_LOCALHOST_ONLY=1
../xmigcs_dex3/.venv/bin/python scripts/simulator_view_asyn.py
```

终端 B：启动控制节点。

```bash
cd /home/lzp/projects/xmigcs_dex3
export ROS_DOMAIN_ID=71
./scripts/run_sonic_sim.sh
```

脚本自动切换工作目录、加载 Humble 并使用本仓库的 `.venv`。
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

## 重建 Python 3.10 环境

本机已完成安装，无需重复执行。换机器时先安装 ROS Humble 及
`lib/22/ros-humble-bodyctrl-msgs_0.0.0-0jammy_amd64.deb`，再执行：

```bash
cd /home/lzp/projects/xmigcs_dex3
uv venv --python 3.10 .venv
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    uv pip install --python .venv/bin/python \
    --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    -r requirements_sim.txt \
    ./lib/22/sptlib_python-0.1.0-cp310-cp310-linux_x86_64.whl
printf '%s\n' /opt/ros/humble/lib/python3.10/site-packages \
    /opt/ros/humble/local/lib/python3.10/dist-packages \
    > .venv/lib/python3.10/site-packages/ros_paths.pth
```

取消代理只对该安装命令生效，用于绕开本机之前出现的 SSL EOF；TLS 验证保持开启。
Python 3.10 使用 `requirements_sim.txt` 和 `lib/22`，不能安装 `lib/24` 的 Python 3.12 wheel。

## 验证和边界

```bash
source /opt/ros/humble/setup.bash
.venv/bin/python -m unittest policy.sonic_track.test_sonic_track test_sim_adaptation -v
.venv/bin/python -m policy.sonic_track.check_mujoco
```

2026-11-02 验证结果：

- 19 项测试通过：真实 ONNX 验证向量、历史与关节顺序、29 关节 ROS 消息、
  50 Hz 调度、循环和播放边界、仿真模式初始化、策略异常回退、无效踝结果拒绝。
- 离线 MuJoCo：60 秒 IDLE，覆盖两次接缝；27.38 秒动作播放场景，覆盖手动返回、
  完整双手动作和自动返回。均完成，状态有限，无提前退出。
  两场景最低根高度分别约 0.905 m / 0.883 m。
- 离线检查采用 1 ms 物理步长、100 Hz 缓存反馈 PD，吊绳仅在前 2 秒启用；
  不包含 DDS 时延和 GUI 负载。结果写入 `.validation/sonic_mujoco.json`。
- ROS 本地话题检查使用独立域 172 和合成静态反馈，覆盖 `z → t → p → i → c`，
  腿／腰／臂消息均约 100 Hz，仿真过程未调用实机踝解算。
  这是接口检查，不是 ROS + MuJoCo 的完整闭环验证。

主循环恢复为读取反馈 → 策略 → 发命令。策略抛出 RuntimeError 或 FloatingPointError
时切换 STOP 保持；STOP 使用仓库已有 PD 参数，并非电机断电。
踝库的失败、非有限或错误维度返回会在继续写入转换结果／发布命令前抛出异常；
此检查不等同于完整的实机故障恢复机制。
