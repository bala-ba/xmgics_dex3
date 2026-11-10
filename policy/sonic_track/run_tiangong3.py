"""Run the frozen Tiangong2Dex SONIC policy on Tiangong3 in MuJoCo.

Writes actual physics trajectories, policy outputs, metrics and optional MP4.
The reference panel is explicitly kinematic; it is never applied to the robot.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "glfw" if "--viewer" in sys.argv else "egl")
import mujoco as mj
import numpy as np
import joblib
from scipy.spatial.transform import Rotation

from common.joystick import ControlFlag
from common.robot_data import RobotData
from policy.zero.fsm_zero import FSMStateZero
from .fsm_sonic_track import FSMStateSONICTRACK, DEFAULT_CONFIG, ROOT
from .sonic_runtime import SERIAL_JOINT_NAMES, rotation
from .tiangong3_asset import build, OUTPUT


def ground_once(model, data):
    """Place authored foot contact cylinders just above the plane at startup."""
    mj.mj_forward(model, data)
    bottoms = []
    for i in range(model.ngeom):
        if ((model.geom(i).name or "").startswith("foot_")
                and model.geom_type[i] == mj.mjtGeom.mjGEOM_CYLINDER):
            z = data.geom_xmat[i].reshape(3, 3)[2, 2]
            radius, half = model.geom_size[i, :2]
            bottoms.append(data.geom_xpos[i, 2] - radius*np.sqrt(max(0., 1-z*z)) - half*abs(z))
    if not bottoms:
        raise ValueError("Asset has no expected foot contact cylinders")
    data.qpos[2] += .0005 - min(bottoms)
    mj.mj_forward(model, data)


class Video:
    def __init__(self, model, path, fps, width=640, height=640):
        import imageio.v2 as imageio
        self.renderer = mj.Renderer(model, height=height, width=width)
        self.writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8, macro_block_size=1)
        self.option = mj.MjvOption()
        self.option.geomgroup[:] = 0
        self.option.geomgroup[[0, 2]] = 1
        self.frames = 0
        self.fps, self.width, self.height = fps, width, height
        self.last = None

    def panel(self, data):
        camera = mj.MjvCamera()
        # Follow each pelvis so drift cannot crop the actual robot out of view.
        # Absolute drift remains visible numerically in the HUD and report.
        camera.lookat[:] = [data.qpos[0], data.qpos[1], .9]
        camera.distance, camera.azimuth, camera.elevation = 3.2, 135., -12.
        self.renderer.update_scene(data, camera=camera, scene_option=self.option)
        return self.renderer.render().copy()

    def append(self, data, reference, phase, error, band, reason=None):
        from PIL import Image, ImageDraw, ImageFont
        frame = Image.fromarray(np.concatenate([self.panel(data), self.panel(reference)], axis=1))
        draw = ImageDraw.Draw(frame)
        font = ImageFont.load_default(size=18)
        draw.rectangle((0, 0, 2*self.width, 84), fill=(17, 25, 35))
        draw.text((12, 10), "TIANGONG3 | SONIC + PHYSICS", font=font, fill="white")
        draw.text((self.width+12, 10), "REFERENCE | KINEMATIC ONLY", font=font, fill="white")
        draw.text((12, 40), f"t={data.time:.2f}s  {phase}  joint RMSE={error:.3f} rad", font=font, fill="white")
        draw.text((self.width+12, 40), "Tiangong2Dex checkpoint: step 100000", font=font, fill="white")
        draw.text((12, 65), "STARTUP SUPPORT ON" if band else "NO EXTERNAL SUPPORT", font=ImageFont.load_default(size=13), fill="#a6dce5")
        draw.text((self.width+12, 65), f"Actual base: x={data.qpos[0]:+.2f} m, y={data.qpos[1]:+.2f} m | cameras follow pelvis",
                  font=ImageFont.load_default(size=13), fill="#a6dce5")
        if reason:
            draw.rectangle((0, self.height-58, 2*self.width, self.height), fill=(115, 25, 25))
            draw.text((12, self.height-44), f"STOP: {reason}", font=font, fill="white")
        self.last = np.asarray(frame)
        self.writer.append_data(self.last)
        self.frames += 1

    def close(self, failed=False):
        try:
            if failed and self.last is not None:
                for _ in range(2*self.fps):
                    self.writer.append_data(self.last)
                    self.frames += 1
        finally:
            self.writer.close()
            self.renderer.close()


def run(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.source is not None:
        model, asset = build(args.source, args.xml)
    else:
        model = mj.MjModel.from_xml_path(str(args.xml))
        report_path = args.xml.with_suffix('.json')
        asset = json.loads(report_path.read_text()) if report_path.is_file() else {'xml': str(args.xml)}
    rd = RobotData()
    rd.control_dt_ = .01
    zero = FSMStateZero(rd)
    state = FSMStateSONICTRACK(rd, args.config)
    contract = state.contract
    qa = np.array([model.joint(n).qposadr[0] for n in SERIAL_JOINT_NAMES])
    va = np.array([model.joint(n).dofadr[0] for n in SERIAL_JOINT_NAMES])
    motors = np.array([model.actuator("motor_"+n).id for n in SERIAL_JOINT_NAMES])
    joint_ids = np.array([model.joint(n).id for n in SERIAL_JOINT_NAMES])
    limits = model.jnt_range[joint_ids]
    data, reference = mj.MjData(model), mj.MjData(model)
    data.qpos[qa] = np.clip(zero.zero_positions_, *limits.T)
    ground_once(model, data)
    reference.qpos[:] = data.qpos
    reference.qpos[qa] = state.idle_reference.q[0][contract.policy_to_serial]
    ground_once(model, reference)
    reference_height = float(reference.qpos[2])
    rd.joint_kp_s_[:], rd.joint_kd_s_[:] = zero.kp_pos_, zero.kd_pos_
    flag = ControlFlag(fsm_state_command="gotoSONICTRACK")
    rd.control_flag = flag
    seconds = args.duration or (60. if args.idle_only else
                               args.play_at + state.references[0].times[-1] + 2*state.blend + 5.)
    records = {k: [] for k in ("time", "phase", "qpos", "qvel", "target_serial", "reference_serial", "reference_qpos", "raw_action_policy", "torque_serial")}
    video = None if args.no_video else Video(model, output/"simulation.mp4", args.fps)
    if args.viewer:
        import mujoco.viewer
        viewer_context = mj.viewer.launch_passive(model, data)
    else:
        viewer_context = nullcontext(None)
    started, next_frame = time.perf_counter(), 0.
    reason, minimum_height, maximum_speed = None, float(data.qpos[2]), 0.
    errors, phases = [], set()
    clipped_targets, clipped_torques, played = 0, 0, False
    try:
        with viewer_context as viewer:
            for tick in range(int(np.ceil(seconds/state.dt))):
                if viewer is not None and not viewer.is_running():
                    reason = "viewer closed"
                    break
                rd.q_a_s_[-29:], rd.q_dot_a_s_[-29:] = data.qpos[qa], data.qvel[va]
                rd.imu_data_[:3] = Rotation.from_quat(data.qpos[[4, 5, 6, 3]]).as_euler("xyz")[::-1]
                rd.imu_data_[3:6] = data.qvel[3:6]
                if tick == 0:
                    state.on_enter()
                    history = state.history
                if not args.idle_only and not played and state.elapsed+1e-9 >= args.play_at:
                    flag.motion_command, flag.motion_event_id = "play", 1
                    played = True
                state.run(flag)
                if state.history is not history:
                    raise RuntimeError("Policy history unexpectedly reset")
                phases.add(state.phase)
                target = np.clip(rd.q_d_s_[-29:], *limits.T)
                clipped_targets += int(np.count_nonzero(target != rd.q_d_s_[-29:]))
                cached_q, cached_v = data.qpos[qa].copy(), data.qvel[va].copy()
                for _ in range(10):
                    q, v = (cached_q, cached_v) if args.pd_feedback == "cached" else (data.qpos[qa], data.qvel[va])
                    requested = rd.joint_kp_s_*(target-q) - rd.joint_kd_s_*v
                    torque = np.clip(requested, *model.actuator_ctrlrange[motors].T)
                    clipped_torques += int(np.count_nonzero(torque != requested))
                    data.ctrl[motors] = torque
                    data.xfrc_applied[:] = 0.
                    if data.time < args.band_seconds:
                        delta = np.array([0., 0., 3.]) - data.qpos[:3]
                        distance = np.linalg.norm(delta)
                        direction = delta/distance
                        force = (300*distance - 100*np.dot(data.qvel[:3], direction))*direction
                        data.xfrc_applied[model.body("waist_pitch_link").id, :3] = force
                    mj.mj_step(model, data)
                qref, _, pref, quatref = state.playlist.sample(state.elapsed)
                reference.qpos[qa] = qref[0][contract.policy_to_serial]
                reference.qpos[:3] = pref[0]
                reference.qpos[2] += reference_height - float(state.idle_reference.pos[0, 2])
                reference.qpos[3:7] = quatref[0]
                mj.mj_forward(model, reference)
                error = data.qpos[qa] - reference.qpos[qa]
                if data.time >= state.ready_time:
                    errors.append(error.copy())
                speed = float(np.max(np.abs(data.qvel[va])))
                minimum_height, maximum_speed = min(minimum_height, float(data.qpos[2])), max(maximum_speed, speed)
                if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                    reason = "non-finite physics state"
                elif data.qpos[2] < .45:
                    reason = "root height below 0.45 m"
                elif speed > 100:
                    reason = "joint speed above 100 rad/s"
                row = dict(time=float(data.time), phase=state.phase, qpos=data.qpos.copy(), qvel=data.qvel.copy(),
                           target_serial=target.copy(), reference_serial=reference.qpos[qa].copy(), reference_qpos=reference.qpos.copy(),
                           raw_action_policy=np.zeros(29) if state.last_raw_action is None else state.last_raw_action.copy(),
                           torque_serial=torque.copy())
                for key, value in row.items():
                    records[key].append(value)
                if video and (data.time+1e-9 >= next_frame or reason):
                    video.append(data, reference, state.phase, float(np.sqrt(np.mean(error**2))),
                                 data.time < args.band_seconds, reason)
                    next_frame += 1/args.fps
                if viewer is not None:
                    viewer.sync()
                    delay = started + float(data.time) - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                if reason:
                    break
    finally:
        if video:
            video.close(failed=reason is not None)
    np.savez_compressed(output/"trajectory.npz", **{k: np.asarray(v) for k, v in records.items()},
                        serial_joint_names=np.array(SERIAL_JOINT_NAMES), policy_joint_names=np.array(contract.names))
    q = np.asarray(records["qpos"])[:, qa]
    csv = np.column_stack([records["time"], np.asarray(records["qpos"])[:, :7], q])
    np.savetxt(output/"actual_motion.csv", csv, delimiter=",", comments="",
               header=",".join(["time", "root_x", "root_y", "root_z", "root_qw", "root_qx", "root_qy", "root_qz", *SERIAL_JOINT_NAMES]))
    # Actual physics rollout, sampled at 50 Hz in the original MotionLib order.
    order = contract.mapping["joint_order"]["mjcf"]
    indices = [SERIAL_JOINT_NAMES.index(name) for name in order]
    actual_qpos = np.asarray(records["qpos"])
    joblib.dump({output.name: dict(
        fps=50., dof=q[1::2, indices], root_rot=actual_qpos[1::2][:, [4, 5, 6, 3]],
        root_trans_offset=actual_qpos[1::2, :3],
        metadata=dict(robot="tiangong3", kind="actual MuJoCo physics rollout",
                      joint_names=order, source_export=str(contract.export),
                      includes_preparation_and_return=True))}, output/"generated_motion.pkl")
    events = state.playlist.events
    finished = any(e["action"] == "motion_finished" for e in events)
    passed = reason is None and (args.idle_only or (finished and state.phase == "IDLE"))
    report = dict(passed=passed, stop_reason=reason, simulated_seconds=float(data.time), requested_seconds=seconds,
                  mode="idle" if args.idle_only else "motion", motion=str(state.references[0].path),
                  idle_motion=str(state.idle_reference.path), source_checkpoint=contract.manifest["training_step"],
                  export_dir=str(contract.export), export_manifest_sha256=hashlib.sha256((contract.export/"export_manifest.json").read_bytes()).hexdigest(),
                  policy_unchanged=True, asset=asset, phases=sorted(phases), events=events,
                  inferences=state.inference_count, minimum_root_height_m=minimum_height,
                  maximum_joint_speed_rad_s=maximum_speed, joint_rmse_rad=float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
                  maximum_root_xy_displacement_m=float(np.linalg.norm(actual_qpos[:, :2]-actual_qpos[0, :2], axis=1).max()),
                  final_root_xy_m=actual_qpos[-1, :2].tolist(),
                  physics_dt=.001, command_hz=100, inference_hz=50, pd_feedback=args.pd_feedback,
                  startup_band_seconds=args.band_seconds, target_clipped_joint_ticks=clipped_targets,
                  torque_clipped_joint_steps=clipped_torques, final_phase=state.phase,
                  video_frames=video.frames if video else 0, video_fps=args.fps,
                  conditions="No ROS/DDS jitter. Tiangong3 URDF dynamics; old policy gains and references. Reference is kinematic visualization only.",
                  elapsed_wall_seconds=time.perf_counter()-started)
    (output/"report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ("asset", "events")}, indent=2), flush=True)
    print(f"Output: {output}", flush=True)
    return 0 if passed else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        help="Rebuild XML from this URDF asset directory; default loads packaged XML")
    parser.add_argument("--xml", type=Path, default=OUTPUT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/tiangong3_sonic_step100000")
    parser.add_argument("--idle-only", action="store_true")
    parser.add_argument("--duration", type=float)
    parser.add_argument("--play-at", type=float, default=6.)
    parser.add_argument("--band-seconds", type=float, default=2.)
    parser.add_argument("--pd-feedback", choices=("cached", "fresh"), default="fresh",
                        help="fresh: 1 kHz inner PD feedback; cached: reproduce old 100 Hz feedback hold")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    if (args.duration is not None and args.duration <= 0) or args.play_at < 2 or args.band_seconds < 0 or args.fps <= 0:
        parser.error("Positive duration/FPS, play-at >= 2 s and band-seconds >= 0 required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
