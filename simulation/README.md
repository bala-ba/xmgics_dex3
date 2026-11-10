# 内置 xmujoco 接口

`simulator_view_asyn.py` 与 `elastic_band.py` 来自本机此前使用的
`xmujoco/scripts/`（上游 `https://github.com/Dknt0/xmujoco`，基线 commit
`ac4954429b00299133a64cf6c7626381dfca5cf7`，包含此前 Tiangong3、headless、PD 模式的本地适配）。

保持原来的 ROS 话题、关节映射、反馈和 PD 实现。移入本仓库后只调整默认模型位置：
`assets/tiangong3/tiangong3_29dof_dex_armature.xml`。

```bash
./scripts/run_mujoco_sim.sh
# 无窗口短时检查（只启动仿真，不启动策略）
./scripts/run_mujoco_sim.sh --headless --duration 3
# 零 armature XML，使用隐式 PD
./scripts/run_mujoco_sim.sh --model assets/tiangong3/tiangong3_29dof.xml --pd-mode implicit
```

`--robot dex` 仍支持，但需用 `--model` 指定自己的完整 Dex XML 资产。
本目录不内置原 Dex 的 XML/网格。当前入口默认关闭深度图，窗口需要桌面 DISPLAY。
