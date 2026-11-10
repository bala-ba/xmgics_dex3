"""Offline SONIC regression using the existing xmujoco asset, without DDS/viewer.

Run: .venv/bin/python -m policy.sonic_track.check_mujoco
Uses 1 ms physics, 100 Hz cached PD feedback, and the virtual band until 2 s.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from common.joystick import ControlFlag
from common.robot_data import RobotData
from policy.zero.fsm_zero import FSMStateZero
from .fsm_sonic_track import FSMStateSONICTRACK, ROOT
from .sonic_runtime import SERIAL_JOINT_NAMES


def run_scenario(xml, scenario):
    rd = RobotData()
    rd.control_dt_ = .01
    zero = FSMStateZero(rd)
    state = FSMStateSONICTRACK(rd)
    # Cover two IDLE seams, a manual return, then a complete selected motion.
    seconds = 60. if scenario == "idle" else max(
        25., 12. + 2*state.blend + state.references[0].times[-1] + 5.)
    ticks = int(np.ceil(seconds / state.dt))
    model = mj.MjModel.from_xml_path(str(xml))
    model.opt.timestep = .001
    data = mj.MjData(model)
    qa = np.array([model.joint(n).qposadr[0] for n in SERIAL_JOINT_NAMES])
    va = np.array([model.joint(n).dofadr[0] for n in SERIAL_JOINT_NAMES])
    motors = np.array([model.actuator("motor_"+n).id for n in SERIAL_JOINT_NAMES])
    data.qpos[qa] = zero.zero_positions_
    mj.mj_forward(model, data)
    bottoms = []
    for i in range(model.ngeom):
        if ((model.geom(i).name or "").startswith("foot_")
                and model.geom_type[i] == mj.mjtGeom.mjGEOM_CYLINDER):
            axis_z = data.geom_xmat[i].reshape(3, 3)[2, 2]
            radius, half = model.geom_size[i, :2]
            extent = radius*np.sqrt(max(0., 1-axis_z**2)) + half*abs(axis_z)
            bottoms.append(data.geom_xpos[i, 2] - extent)
    if not bottoms:
        raise ValueError("Expected xmujoco foot cylinders for initial grounding")
    data.qpos[2] += .0005 - min(bottoms)
    mj.mj_forward(model, data)
    rd.joint_kp_s_[:] = zero.kp_pos_
    rd.joint_kd_s_[:] = zero.kd_pos_
    flag = ControlFlag(fsm_state_command="gotoSONICTRACK")
    rd.control_flag = flag
    minimum_height, maximum_speed = float(data.qpos[2]), 0.
    errors, samples, phases, idle_cycles = [], [], set(), set()
    finite = True
    started = time.perf_counter()
    for tick in range(ticks):
        rd.q_a_s_[-29:] = data.qpos[qa]
        rd.q_dot_a_s_[-29:] = data.qvel[va]
        euler = Rotation.from_quat(data.qpos[[4, 5, 6, 3]]).as_euler("xyz")
        rd.imu_data_[:3] = euler[::-1]
        rd.imu_data_[3:6] = data.qvel[3:6]
        if tick == 0:
            state.on_enter()
            history = state.history
        if scenario == "play_return" and tick in (600, 1000, 1200):
            flag.motion_command = "idle" if tick == 1000 else "play"
            flag.motion_event_id += 1
        state.run(flag)
        assert state.history is history, "History reset at a playback boundary"
        phases.add(state.phase)
        if state.phase == "IDLE" and state.elapsed >= state.playlist.idle_epoch:
            idle_cycles.add(int((state.elapsed-state.playlist.idle_epoch) / state.playlist.idle_loop.period))
        cached_q, cached_v = data.qpos[qa].copy(), data.qvel[va].copy()
        for _ in range(10):
            torque = rd.joint_kp_s_*(rd.q_d_s_[-29:]-cached_q) - rd.joint_kd_s_*cached_v
            data.ctrl[motors] = np.clip(torque, *model.actuator_ctrlrange[motors].T)
            data.xfrc_applied[:] = 0.
            if data.time < 2.:
                delta = np.array([0., 0., 3.]) - data.qpos[:3]
                distance = np.linalg.norm(delta)
                direction = delta / distance
                force = (300*distance - 100*np.dot(data.qvel[:3], direction))*direction
                data.xfrc_applied[model.body("waist_pitch_link").id, :3] = force
            mj.mj_step(model, data)
        minimum_height = min(minimum_height, float(data.qpos[2]))
        maximum_speed = max(maximum_speed, float(np.max(np.abs(data.qvel[va]))))
        finite = bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all())
        if tick >= 220:
            reference = state.playlist.sample(state.elapsed)[0][0]
            errors.append(data.qpos[qa][state.contract.serial_to_policy] - reference)
        if tick % 100 == 0:
            samples.append(dict(time=float(data.time), height=float(data.qpos[2]), phase=state.phase))
        if not finite or data.qpos[2] < .45 or maximum_speed > 100:
            break
    event_names = [e["action"] for e in state.playlist.events]
    covered = (max(idle_cycles, default=0) >= 2 if scenario == "idle" else
               "motion_finished" in event_names and state.phase == "IDLE")
    result = dict(
        scenario=scenario, passed=bool(finite and tick == ticks-1 and covered),
        simulated_seconds=float(data.time), planned_seconds=ticks*state.dt,
        minimum_root_height_m=minimum_height, maximum_joint_speed_rad_s=maximum_speed,
        joint_rmse_rad=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        all_finite=finite, no_early_exit=tick == ticks-1, phases=sorted(phases),
        completed_idle_cycles=max(idle_cycles, default=0), inferences=state.inference_count,
        motion=str(state.references[0].path) if hasattr(state.references[0], "path") else str(state.config_path),
        events=state.playlist.events, samples=samples,
        wall_seconds=time.perf_counter()-started,
    )
    print(json.dumps({k: v for k, v in result.items() if k not in ("samples", "events")}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", type=Path, default=ROOT/"assets/tiangong3/tiangong3_29dof_dex_armature.xml")
    parser.add_argument("--output", type=Path, default=ROOT/".validation/sonic_mujoco.json")
    args = parser.parse_args()
    results = [run_scenario(args.xml, name) for name in ("idle", "play_return")]
    report = dict(
        xml=str(args.xml.resolve()), xml_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest(),
        conditions="ZERO pose/gains, grounded once; 1 ms physics, 100 Hz cached PD feedback; band until 2 s; no DDS jitter or viewer.",
        results=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"Report: {args.output}")
    return 0 if all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
