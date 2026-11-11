# 在另一台电脑部署

默认控制器所需模型和动作已放进本仓库。部署不需要旁边的 `GR00T-WholeBodyControl`、
`instantTrack`、`TienKung-Lab-dev` 或 `xmujoco` 目录，也不需要原始 `.pt` checkpoint、
PyTorch、IsaacLab 或 CUDA。只运行 ONNX CPU 推理即可。

## 文件清单

| 功能 | 仓库内文件 | 状态 |
| --- | --- | --- |
| SONIC | `policy/sonic_track/model/export_100000_20261103_111558/` | encoder、decoder、manifest、mapping、训练配置、验证向量齐全 |
| SONIC 参考动作 | `policy/sonic_track/motions/` | IDLE 的 `idle_loop_001__A028.pkl`、行走 `walk_forward_loop_001__A022.pkl`、下蹲 `squat_001__A362.pkl` |
| InstantTrack | `policy/instant_track/model/` | 6 套包，每套含 `policy.onnx`、`motion_000.npz`、`manifest.json`、`validation.npz`；跑步包裁至摆动腿落地，测试结果见 InstantTrack README |
| 默认 STAND / DEXAMP / DEXDELTA / MULTIAMP | 各 `policy/*/model/` | 齐全 |
| 旧 WALKAMP / WALKLION / WALKLIONDWAQ / BEYONDMIMIC | 各 `policy/*/model/` | 补入当前配置指定的 4 个模型，保持原有禁用/未注册状态 |
| Tiangong3 | `assets/tiangong3/` | 29 DoF XML、DEX armature XML、39 个网格、源 URDF 与构建所需 torque XML |
| ROS 仿真接口 | `simulation/simulator_view_asyn.py`、`simulation/elastic_band.py` | 内置，可直接启动 |
| 本地接口库 | `lib/22/`、`lib/24/` | x86_64 的 Humble/Python 3.10 和 Jazzy/Python 3.12 wheel、ROS 消息 deb |

InstantTrack 默认是 `ckpt_438000_run1_subject2_first_forward`：2.2 秒跑步、0.86 秒落腿收尾，
落地后的参考帧已删除（154 帧、3.06 秒），按 `z → 1 → b` 准备后自动播放。结束后 policy
继续跟踪静止末帧。另有 `ckpt_438000` 的首段直行（601 帧、12 秒）、下蹲、
轻跳、敬礼和完整步行备份；切换 `policy/instant_track/config/instant_track.yaml` 的
`export_dir` 即可。SONIC 配置位于 `policy/sonic_track/config/sonic_track.yaml`。
导出 manifest 里的 `/home/lzp/...` 训练路径仅记录来源，运行时不读取这些位置。

以下两个旧深度策略的指定模型在本机所有项目中均未找到，因此仍未补齐：

- `policy/walk_depth/model/50hz_200p_x3_v2.onnx`
- `policy/walk_depth_100hz/model/100hz_400p_slow_stair_v2.onnx`

它们默认禁用，不影响 SONIC、InstantTrack 和默认四种策略启动。
现有 `walk_depth/model/policy.onnx` 不是这两个指定模型，不能直接替换。

## 安装环境

已验证组合：**Ubuntu 22.04、x86_64、ROS 2 Humble、Python 3.10**。
不要复制旧电脑的 `.venv`，在新电脑重建。ROS 2 和 Python 系统环境需要先安装；
本仓库的二进制库不能直接用于 Windows、macOS 或 ARM。

进入克隆或解压后的项目根目录，先检查文件（这一步只需系统 Python）：

```bash
python3 scripts/check_deployment.py --files-only
```

安装与 ROS 对应的消息包及 Python 依赖：

```bash
# Ubuntu 22.04 / Humble / Python 3.10
sudo apt install python3.10-venv
sudo apt install ./lib/22/ros-humble-bodyctrl-msgs_0.0.0-0jammy_amd64.deb
source /opt/ros/humble/setup.bash
python3.10 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements_sim.txt \
  ./lib/22/sptlib_python-0.1.0-cp310-cp310-linux_x86_64.whl
.venv/bin/python scripts/check_deployment.py --ros
```

`--ros` 校验文件 SHA-256、SONIC 双网络、全部 InstantTrack 动作包的 CPU 验证向量、
MuJoCo XML/网格、ROS/踝库导入，以及所有默认启用的 FSM 状态初始化。
它不会创建 ROS 节点或发送控制命令。输出中的两个 `OPTIONAL` 缺失提示对应上面禁用的深度策略。

Ubuntu 24.04 使用 ROS Jazzy、Python 3.12、`lib/24`；对应安装命令如下，
该组合提供配套包，但本次未在 24.04 主机验证：

```bash
sudo apt install python3.12-venv
sudo apt install ./lib/24/ros-jazzy-bodyctrl-msgs_0.0.0-0noble_amd64.deb
source /opt/ros/jazzy/setup.bash
python3.12 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements_mujoco.txt \
  ./lib/24/sptlib_python-0.1.0-cp312-cp312-linux_x86_64.whl
.venv/bin/python scripts/check_deployment.py --ros
```

## 启动

仿真两个终端都进入项目目录，分别运行：

```bash
# 终端 A：Tiangong3 + DEX armature，显式 PD
./scripts/run_mujoco_sim.sh
```

```bash
# 终端 B：控制器
./scripts/run_sonic_sim.sh
```

脚本默认 Humble、ROS 域 71、仅本机通信；Jazzy 请先 `source /opt/ros/jazzy/setup.bash`。
也可显式设置 `ROS_DISTRO=jazzy`。完整参数和按键见 [SIMULATION.md](SIMULATION.md)。

实机入口仍是 `rl_control_node.py`，需要把 `config/dex_config.yaml` 设置为 `sim: false`，
加载实机的 ROS 环境并与机器人使用相同 ROS 域。跨电脑通信不能使用
`ROS_LOCALHOST_ONLY=1`。机器人侧的 `body_control` 驱动和硬件标定由机器人环境提供，
不是本仓库的模型文件；本次没有验证实机控制。

## 迁移后的修复

- SONIC 模型和动作路径改为相对本仓库解析，文件 SHA-256 与原导出保持一致。
- 四种输入控制器的配置定位不再依赖终端当前目录。
- 删除按用户名 `ubuntu` 禁止仿真的旧逻辑；仿真入口仍在启动线程前强制 `sim=True`。
- Tiangong3 网格引用改为仓库内相对路径，资产构建也能使用内置 URDF，默认运行不再重建 XML。
- `deployment_files.json` 记录当前模型、动作、资产和本地库的大小及 SHA-256，用于检查传输是否完整。

## 本次验证记录

2026-11-10，在 Humble / Python 3.10 环境，把待提交文件复制到 `/tmp` 下名称含空格的
独立目录，并从项目外部工作目录执行：89 个资产文件的完整性检查、SONIC 及全部 5 套
InstantTrack 验证向量、全部默认状态初始化、43 项回归测试均通过。
内置仿真器在独立本机 ROS 域运行 3 秒、约 1 kHz 后正常退出。
验证未连接实机；新电脑仍需按上文安装 ROS 和匹配的 Python/本地库。
