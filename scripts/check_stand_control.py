"""Run the keyboard ZERO -> STAND path against xmujoco, then release support."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bodyctrl_msgs.msg import CmdMotorCtrl, MotorCtrl
from common.body_id_map import BodyServoIdMap
from common.robot_data import RobotData
from common.stdin_keyboard_control import KeyboardController
from FSM.fsm_base import FSMStateName
from FSM.robot_fsm import RobotFSMImpl
from policy.instant_track.runtime import SERIAL_JOINT_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=ROOT.parent/'xmujoco/urdf/tiangong3_29dof_dex_armature.xml')
    parser.add_argument('--pd-mode', choices=('auto', 'explicit', 'implicit'), default='explicit')
    parser.add_argument('--seconds', type=float, default=60.)
    parser.add_argument('--enter-at', type=float, default=10.)
    parser.add_argument('--band-seconds', type=float, default=12.)
    parser.add_argument('--output', type=Path, default=ROOT/'.validation/stand/tiangong3.json')
    args = parser.parse_args()
    if (not all(np.isfinite(v) for v in (args.seconds, args.enter_at, args.band_seconds))
            or not 0 <= args.enter_at < args.band_seconds < args.seconds):
        parser.error('Require 0 <= enter-at < band-seconds < seconds')
    sys.path.insert(0, str(ROOT.parent/'xmujoco/scripts'))
    from simulator_view_asyn import RobotSimulator
    sim = RobotSimulator(str(args.model), depth=False, headless=True, pd_mode=args.pd_mode)
    model, data = sim.model, sim.data
    va = np.array([model.joint(n).dofadr[0] for n in SERIAL_JOINT_NAMES])
    rd = RobotData()
    rd.control_dt_ = .01
    config = yaml.safe_load((ROOT/'config/dex_config.yaml').read_text())
    config['enabled_policy_states'] = ['STAND']
    config['sonic_track']['enabled'] = config['instant_track']['enabled'] = False
    fsm = RobotFSMImpl(rd, config)
    keyboard = KeyboardController()
    keyboard._process_key('z')
    ids = BodyServoIdMap()
    ids.body_can_id_map_init()
    cadence = round(rd.control_dt_/model.opt.timestep)
    assert np.isclose(cadence*model.opt.timestep, rd.control_dt_)
    entered, reason = False, None
    records, states = [], []
    max_speed = max_tilt = max_displacement = 0.
    initial_xy = data.qpos[:2].copy()
    for tick in range(round(args.seconds/model.opt.timestep)):
        sim.read_sensors()
        if tick % cadence == 0:
            rd.q_a_s_[-29:] = [sim.joint_positions[n] for n in SERIAL_JOINT_NAMES]
            rd.q_dot_a_s_[-29:] = [sim.joint_velocities[n] for n in SERIAL_JOINT_NAMES]
            rd.imu_data_[:3] = Rotation.from_quat(sim.imu_orientation[[1, 2, 3, 0]]).as_euler('xyz')[::-1]
            rd.imu_data_[3:6] = sim.imu_angular_velocity
            if not entered and data.time >= args.enter_at:
                keyboard._process_key('1')
                entered = True
            rd.control_flag = keyboard.get_keyboard_flag()
            rd.walk_cmd_[:] = 0
            fsm.run_fsm(rd.control_flag)
            msg = CmdMotorCtrl()
            for i in range(29):
                cmd = MotorCtrl()
                cmd.name = ids.get_id_by_index(i)
                cmd.pos, cmd.spd, cmd.tor = (float(a[6+i]) for a in (rd.q_d_s_, rd.q_dot_d_s_, rd.tau_d_s_))
                cmd.kp, cmd.kd = float(rd.joint_kp_s_[i]), float(rd.joint_kd_s_[i])
                msg.cmds.append(cmd)
            sim.joint_cmd_callback(msg)
            records.append([data.time, data.qpos[2], np.max(np.abs(data.qvel[va])),
                            np.linalg.norm(data.qpos[:2]-initial_xy)])
            states.append(data.qpos.copy())
        sim.elastic_band.enable = data.time < args.band_seconds
        sim.add_virtual_force()
        sim.set_motor_cmd()
        mj.mj_step(model, data)
        speed = float(np.max(np.abs(data.qvel[va])))
        tilt = float(np.arccos(np.clip(Rotation.from_quat(data.qpos[[4, 5, 6, 3]]).as_matrix()[2, 2], -1, 1)))
        displacement = float(np.linalg.norm(data.qpos[:2]-initial_xy))
        max_speed, max_tilt, max_displacement = max(max_speed, speed), max(max_tilt, tilt), max(max_displacement, displacement)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            reason = 'non-finite state'
        elif data.qpos[2] < .45 or tilt > .8:
            reason = 'fallen: root height below 0.45 m or tilt above 0.8 rad'
        elif speed > 30:
            reason = 'joint speed above 30 rad/s'
        elif displacement > 1:
            reason = 'root moved more than 1 m under zero velocity command'
        if reason:
            break
    stand = fsm.state_objects[FSMStateName.STAND]
    report = dict(passed=reason is None and fsm.current_state == FSMStateName.STAND,
                  failure=reason, seconds=float(data.time), enter_at=args.enter_at,
                  band_seconds=args.band_seconds, pd_mode=sim.pd_mode,
                  model_xml=str(args.model.resolve()), model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  policy_model=stand.model_path, policy_sha256=hashlib.sha256(Path(stand.model_path).read_bytes()).hexdigest(),
                  policy_hz=1/(stand.dt_*stand.decimation_), inferences=stand.inference_count,
                  minimum_root_height_m=float(min(r[1] for r in records)),
                  maximum_joint_speed_rad_s=max_speed, maximum_tilt_rad=max_tilt,
                  maximum_xy_displacement_m=max_displacement, final_state=fsm.current_state.name,
                  conditions='Actual keyboard mapping and FSM; 1 kHz physics/PD, 100 Hz commands; no ROS jitter')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'), records=records, qpos=states)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
