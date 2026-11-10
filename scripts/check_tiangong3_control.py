"""Run ZERO/STAND -> InstantTrack through xmujoco's actual PD, without ROS jitter.

Source /opt/ros/humble/setup.bash first (message types are used, no ROS node is created).
"""
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
from common.body_id_map import BodyServoIdMap
from common.robot_data import RobotData
from common.stdin_keyboard_control import KeyboardController
from FSM.fsm_base import FSMStateName
from FSM.robot_fsm import RobotFSMImpl
from policy.instant_track.fsm_instant_track import FSMStateINSTANTTRACK
from policy.instant_track.runtime import Runtime, SERIAL_JOINT_NAMES
from bodyctrl_msgs.msg import CmdMotorCtrl, MotorCtrl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--simulator-root', type=Path, help='Optional external xmujoco checkout')
    parser.add_argument('--model', type=Path, help='XML override for model comparison experiments')
    parser.add_argument('--export-dir', type=Path, help='InstantTrack export override for this test only')
    parser.add_argument('--seconds', type=float, default=35.)
    parser.add_argument('--finish-hold-seconds', type=float, default=0.,
                        help='continue controlling the final pose for this long after playback finishes')
    parser.add_argument('--prepare-at', type=float, default=10.)
    parser.add_argument('--stand-at', type=float, help='enter STAND before b, instead of entering from ZERO')
    parser.add_argument('--play-at', type=float, default=12.5)
    parser.add_argument('--hold-only', action='store_true', help='press b then i to cancel automatic playback and hold frame 0')
    parser.add_argument('--b-only', action='store_true', help='press b only; verify preparation automatically starts playback')
    parser.add_argument('--direct-play', action='store_true', help='press p from STAND to immediately play from frame 0')
    parser.add_argument('--band-seconds', type=float, default=14.)
    parser.add_argument('--zero-only', action='store_true')
    parser.add_argument('--pd-mode', choices=('auto', 'implicit', 'explicit'), default='auto')
    parser.add_argument('--output', type=Path, default=ROOT/'.validation/tiangong3_debug/control.json')
    args = parser.parse_args()
    if (not all(np.isfinite(v) for v in (args.seconds, args.prepare_at, args.play_at, args.band_seconds,
                                       args.finish_hold_seconds))
            or args.seconds <= 0 or args.prepare_at < 0
            or (not args.zero_only and not args.hold_only and not args.b_only and not args.direct_play
                and args.play_at < args.prepare_at + 2.1)
            or args.band_seconds < 0 or args.finish_hold_seconds < 0):
        parser.error('Require positive duration, nonnegative preparation/band times, and play at least 2.1 s after preparation')
    if args.export_dir and args.zero_only:
        parser.error('export-dir cannot be combined with zero-only')
    if args.finish_hold_seconds and (args.zero_only or args.hold_only):
        parser.error('finish-hold-seconds requires playback')
    if args.stand_at is not None and (not np.isfinite(args.stand_at)
            or not 0 <= args.stand_at < (args.play_at if args.direct_play else args.prepare_at)):
        parser.error('Require 0 <= stand-at < prepare-at')
    if args.b_only and (args.hold_only or args.direct_play or args.zero_only):
        parser.error('b-only cannot be combined with hold-only/direct-play/zero-only')
    if args.direct_play and (args.stand_at is None or args.hold_only or args.zero_only):
        parser.error('direct-play requires stand-at and cannot be combined with hold-only/zero-only')
    if args.zero_only and (args.hold_only or args.stand_at is not None):
        parser.error('zero-only cannot be combined with hold-only or stand-at')
    simulator_scripts = args.simulator_root/'scripts' if args.simulator_root else ROOT/'simulation'
    sys.path.insert(0, str(simulator_scripts))
    from simulator_view_asyn import RobotSimulator
    default_model = (args.simulator_root/'urdf/tiangong3_29dof.xml' if args.simulator_root
                     else ROOT/'assets/tiangong3/tiangong3_29dof.xml')
    model_path = (args.model or default_model).resolve()
    sim = RobotSimulator(str(model_path),
                         depth=False, headless=True, pd_mode=args.pd_mode)
    model, data = sim.model, sim.data
    qa = np.array([model.joint(n).qposadr[0] for n in SERIAL_JOINT_NAMES])
    va = np.array([model.joint(n).dofadr[0] for n in SERIAL_JOINT_NAMES])
    rd = RobotData()
    rd.control_dt_ = .01
    cfg = yaml.safe_load((ROOT/'config/dex_config.yaml').read_text())
    cfg['enabled_policy_states'] = ['STAND'] if args.stand_at is not None else []
    cfg['sonic_track']['enabled'] = False
    cfg['instant_track']['enabled'] = not args.zero_only and args.export_dir is None
    fsm = RobotFSMImpl(rd, cfg)
    if args.export_dir is not None:
        fsm.state_objects[FSMStateName.INSTANTTRACK] = FSMStateINSTANTTRACK(
            rd, runtime=Runtime(args.export_dir.expanduser().resolve()))
    keyboard = KeyboardController()
    keyboard._process_key('z')
    ids = BodyServoIdMap()
    ids.body_can_id_map_init()
    cadence = round(rd.control_dt_/model.opt.timestep)
    assert np.isclose(cadence*model.opt.timestep, rd.control_dt_)
    played = prepared = stood = hold_cancelled = False
    maximum_speed = 0.
    reason = None
    finished_at = None
    records, phases, states, policy_steps = [], [], [], []
    for tick in range(round(args.seconds/model.opt.timestep)):
        sim.read_sensors()
        if tick % cadence == 0:
            rd.q_a_s_[-29:] = [sim.joint_positions[n] for n in SERIAL_JOINT_NAMES]
            rd.q_dot_a_s_[-29:] = [sim.joint_velocities[n] for n in SERIAL_JOINT_NAMES]
            rd.imu_data_[:3] = Rotation.from_quat(sim.imu_orientation[[1, 2, 3, 0]]).as_euler('xyz')[::-1]
            rd.imu_data_[3:6] = sim.imu_angular_velocity
            if args.stand_at is not None and not stood and data.time >= args.stand_at:
                keyboard._process_key('1')
                stood = True
            if not args.zero_only and not args.direct_play and not prepared and data.time >= args.prepare_at:
                keyboard._process_key('b')
                prepared = True
            if (args.hold_only and not hold_cancelled and prepared
                    and fsm.current_state == FSMStateName.INSTANTTRACK):
                keyboard._process_key('i')
                hold_cancelled = True
            if not args.zero_only and not args.hold_only and not args.b_only and not played and data.time >= args.play_at:
                keyboard._process_key('p')
                played = True
            rd.control_flag = keyboard.get_keyboard_flag()
            fsm.run_fsm(rd.control_flag)
            state = fsm.state_objects[fsm.current_state]
            msg = CmdMotorCtrl()
            for i in range(29):
                cmd = MotorCtrl()
                cmd.name = ids.get_id_by_index(i)
                cmd.pos, cmd.spd, cmd.tor = (float(a[6+i]) for a in (rd.q_d_s_, rd.q_dot_d_s_, rd.tau_d_s_))
                cmd.kp, cmd.kd = float(rd.joint_kp_s_[i]), float(rd.joint_kd_s_[i])
                msg.cmds.append(cmd)
            sim.joint_cmd_callback(msg)
            phase = getattr(state, 'phase', fsm.current_state.name)
            if phase == 'FINISHED' and finished_at is None:
                finished_at = float(data.time)
            phases.append(phase)
            records.append([data.time, data.qpos[2], np.max(np.abs(data.qvel[va])),
                            np.max(np.abs(data.qpos[qa]-rd.q_d_s_[-29:]))])
            states.append(data.qpos.copy())
            policy_steps.append([data.time, getattr(state, 'frame', -1), getattr(state, 'inference_count', 0)])
            if played and data.time > args.play_at + .1 and phase in ('PREPARING', 'READY'):
                reason = 'playback entry blocked'
                break
        sim.elastic_band.enable = data.time < args.band_seconds
        sim.add_virtual_force()
        sim.set_motor_cmd()
        mj.mj_step(model, data)
        speed = float(np.max(np.abs(data.qvel[va])))
        maximum_speed = max(maximum_speed, speed)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            reason = 'non-finite state'
        elif data.qpos[2] < .45:
            reason = 'root below 0.45 m'
        elif speed > 30:
            reason = f'joint speed above 30 rad/s: {SERIAL_JOINT_NAMES[int(np.argmax(np.abs(data.qvel[va])))]}'
        if reason or (finished_at is not None and data.time-finished_at >= args.finish_hold_seconds):
            break
    expected_phase = 'READY' if args.hold_only else 'PLAYING'
    if reason is None and args.finish_hold_seconds and (
            finished_at is None or data.time-finished_at < args.finish_hold_seconds):
        reason = 'playback and requested final-pose hold did not complete within --seconds'
    record_array, state_array = np.asarray(records), np.asarray(states)
    hold_mask = np.asarray(phases) == 'FINISHED'
    hold_drift = None
    hold_tail_speed = None
    if np.count_nonzero(hold_mask) >= 2:
        hold_states, hold_times = state_array[hold_mask], record_array[hold_mask, 0]
        hold_drift = float(np.linalg.norm(hold_states[-1, :2]-hold_states[0, :2]))
        tail = np.searchsorted(hold_times, hold_times[-1]-1.)
        elapsed = hold_times[-1]-hold_times[tail]
        if elapsed > 0:
            hold_tail_speed = float(np.linalg.norm(hold_states[-1, :2]-hold_states[tail, :2])/elapsed)
    report = dict(passed=reason is None and (args.zero_only or expected_phase in phases),
                  failure=reason, simulated_seconds=float(data.time), pd_mode=sim.pd_mode,
                  model_xml=str(model_path),
                  model_xml_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  simulator_sha256=hashlib.sha256((simulator_scripts/'simulator_view_asyn.py').read_bytes()).hexdigest(),
                  physics_dt=float(model.opt.timestep), command_dt=rd.control_dt_,
                  armature_by_joint={name: float(model.dof_armature[model.joint(name).dofadr[0]]) for name in SERIAL_JOINT_NAMES},
                  checkpoint=None if not hasattr(state, 'runtime') else state.runtime.manifest['checkpoint'],
                  export_dir=None if not hasattr(state, 'runtime') else str(state.runtime.export.resolve()),
                  motion=None if not hasattr(state, 'runtime') else state.runtime.manifest['motions'][state.selected]['name'],
                  stand_at=args.stand_at, prepare_at=None if args.direct_play else args.prepare_at,
                  direct_play=args.direct_play,
                  b_only=args.b_only, manual_play_sent=played,
                  play_at=None if args.hold_only or args.b_only else args.play_at, band_seconds=args.band_seconds,
                  maximum_joint_speed_rad_s=maximum_speed,
                  minimum_root_height_m=float(np.min(record_array[:, 1])),
                  maximum_root_height_m=float(np.max(record_array[:, 1])),
                  final_phase=getattr(state, 'phase', 'ZERO'), phases=sorted(set(phases)),
                  final_frame=getattr(state, 'frame', None),
                  inferences=getattr(state, 'inference_count', 0),
                  finished_at_seconds=finished_at,
                  final_pose_hold_seconds=0. if finished_at is None else float(data.time-finished_at),
                  final_hold_xy_displacement_m=hold_drift,
                  final_hold_last_second_xy_speed_m_s=hold_tail_speed,
                  conditions='Actual xmujoco PD + deployment FSM; 1 kHz physics, 100 Hz commands; no ROS jitter')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'), records=records, phases=phases, qpos=states,
                        policy_steps=policy_steps)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
