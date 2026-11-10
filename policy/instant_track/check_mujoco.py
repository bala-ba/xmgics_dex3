"""Run the actual deployment FSM against an exported InstantTrack training scene.

No ROS graph or robot connection. Generate --xml with export_instant_track.py
--validation-scene, then run with xmigcs/.venv after sourcing ROS.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from common.robot_data import RobotData
from policy.zero.fsm_zero import FSMStateZero
from .fsm_instant_track import FSMStateINSTANTTRACK, ROOT
from .runtime import SERIAL_JOINT_NAMES, rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--xml', type=Path, default=ROOT/'.validation/instant_track/training_scene.xml')
    parser.add_argument('--start', choices=['reference', 'prepare'], default='reference')
    parser.add_argument('--seconds', type=float, default=15.)
    parser.add_argument('--support-orientation', action='store_true',
                        help='also support root orientation during preparation, released before playback')
    parser.add_argument('--output', type=Path, default=ROOT/'.validation/instant_track/mujoco.json')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error('--seconds must be finite and positive')
    rd = RobotData()
    rd.control_dt_ = .01
    state = FSMStateINSTANTTRACK(rd)
    r, ref = state.runtime, state.motion
    model = mj.MjModel.from_xml_path(str(args.xml))
    # Same nominal CPU settings as instanttrack.env.make_sim; no randomization.
    model.opt.timestep = .005
    model.opt.integrator = mj.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.cone = mj.mjtCone.mjCONE_PYRAMIDAL
    model.opt.solver = mj.mjtSolver.mjSOL_NEWTON
    model.opt.iterations, model.opt.ls_iterations = 100, 50
    model.opt.tolerance, model.opt.ccd_iterations = 1e-6, 35
    model.geom_friction[:, 0] = 1.
    data = mj.MjData(model)
    joints = [model.joint('robot/'+n).id for n in SERIAL_JOINT_NAMES]
    qa, va = model.jnt_qposadr[joints], model.jnt_dofadr[joints]
    motors = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == j)[0] for j in joints])
    if model.nq != 36 or model.nv != 35 or model.nu != 29:
        raise ValueError('Expected the exported training scene with 29 position actuators')
    root_id = model.body('robot/pelvis').id
    if args.start == 'reference':
        data.qpos[:3], data.qpos[3:7] = ref['root_pos'][0], ref['root_quat'][0]
        data.qpos[qa] = ref['q'][0][r.policy_to_serial]
        data.qvel[:3] = ref['root_vel'][0]
        data.qvel[3:6] = rotation(ref['root_quat'][0]).T@ref['root_omega'][0]
        data.qvel[va] = ref['dq'][0][r.policy_to_serial]
        rd.joint_kp_s_[:], rd.joint_kd_s_[:] = r.kp[r.policy_to_serial], r.kd[r.policy_to_serial]
    else:
        zero = FSMStateZero(rd)
        data.qpos[:3], data.qpos[3:7] = [0., 0., .95], [1., 0., 0., 0.]
        data.qpos[qa] = zero.zero_positions_
        rd.joint_kp_s_[:], rd.joint_kd_s_[:] = zero.kp_pos_, zero.kd_pos_
    mj.mj_forward(model, data)
    report = dict(checkpoint=r.manifest['checkpoint'], iteration=r.manifest['iteration'],
                  motion=r.manifest['motions'][0]['name'], start=args.start,
                  scene=str(args.xml.resolve()), scene_sha256=hashlib.sha256(args.xml.read_bytes()).hexdigest(),
                  conditions='CPU MuJoCo 3.12; nominal training scene, native position PD at 200 Hz; '
                             'deployment FSM/target limiting at 100 Hz, ONNX at 50 Hz; no ROS jitter or parallel ankle dynamics',
                  suspension_seconds=state.preparation+.01 if args.start == 'prepare' else 0.,
                  orientation_support=bool(args.support_orientation and args.start == 'prepare'),
                  requested_seconds=args.seconds, failed_at_seconds=None)
    errors, times, states, targets, frames, inference_times = [], [], [], [], [], []
    started = time.perf_counter()
    for tick in range(int(np.ceil(args.seconds/rd.control_dt_))):
        rd.q_a_s_[-29:], rd.q_dot_a_s_[-29:] = data.qpos[qa], data.qvel[va]
        rd.imu_data_[:3] = Rotation.from_quat(data.qpos[[4, 5, 6, 3]]).as_euler('xyz')[::-1]
        rd.imu_data_[3:6] = data.qvel[3:6]
        if tick == 0:
            state.on_enter()
            if args.start == 'reference':
                # Reference-state initialization, as in training evaluation.
                q, _, _, quat = state._feedback()
                state._finish_preparation(q, quat)
        if state.phase == 'READY' and rd.control_flag.motion_command != 'play':
            rd.control_flag.motion_command = 'play'
            rd.control_flag.motion_event_id += 1
        before = time.perf_counter()
        count = state.inference_count
        state.run(rd.control_flag)
        if state.inference_count != count:
            inference_times.append(time.perf_counter()-before)
        model.actuator_gainprm[motors, 0] = rd.joint_kp_s_
        model.actuator_biasprm[motors, 1] = -rd.joint_kp_s_
        model.actuator_biasprm[motors, 2] = -rd.joint_kd_s_
        data.ctrl[motors] = rd.q_d_s_[-29:]
        for _ in range(2):
            data.xfrc_applied[:] = 0.
            if args.start == 'prepare' and state.phase in ('PREPARING', 'READY'):
                # Vertical support only during pose preparation; no root state reset.
                data.xfrc_applied[root_id, 2] = (
                    model.body_mass.sum()*9.81 + 800*(ref['root_pos'][0, 2]-data.qpos[2])-100*data.qvel[2])
                if args.support_orientation:
                    R = rotation(data.qpos[3:7])
                    error = Rotation.from_matrix(ref['root_R'][0]@R.T).as_rotvec()
                    data.xfrc_applied[root_id, 3:] = 150*error-20*(R@data.qvel[3:6])
            mj.mj_step(model, data)
        frame = min(state.frame, len(ref['q'])-1)
        finite = np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        error_z = abs(data.qpos[2]-ref['root_pos'][frame, 2])
        tilt_error = abs(rotation(data.qpos[3:7])[2, 2]-ref['root_R'][frame, 2, 2]) if finite else np.inf
        if state.phase == 'PLAYING':
            errors.append(data.qpos[qa]-ref['q'][frame][r.policy_to_serial])
        times.append(data.time)
        states.append(data.qpos.copy())
        targets.append(rd.q_d_s_[-29:].copy())
        frames.append(frame)
        if not finite or (state.phase == 'PLAYING' and (error_z > .25 or tilt_error > .8)):
            report.update(failed_at_seconds=float(data.time), height_error_m=float(error_z),
                          tilt_error=float(tilt_error), finite=bool(finite))
            break
        if state.phase == 'FINISHED':
            break
    report.update(passed=report['failed_at_seconds'] is None and state.inference_count > 0,
                  simulated_seconds=float(data.time),
                  inferences=state.inference_count, phase=state.phase,
                  joint_rmse_rad=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                  inference_p95_ms=float(np.percentile(inference_times, 95)*1000) if inference_times else None,
                  wall_seconds=time.perf_counter()-started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'), time=times, qpos=states, targets=targets, frames=frames)
    print(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
