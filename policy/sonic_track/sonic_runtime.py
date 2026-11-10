"""SONIC export reader and motion runtime, independent of ROS and IsaacLab.

Arrays crossing the policy boundary are ALWAYS in export-manifest joint order.
The robot encoder branch is historically called ``g1``, including for Tiangong.
Adapted from GR00T-WholeBodyControl/tools/tiangong/sim2sim/contract.py.
The frozen export is the only source of model/PD parameters; no training imports.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

# RobotData serial-buffer order (BodyServoIdMap motor order), NOT IsaacLab order.
SERIAL_JOINT_NAMES = (
    "hip_pitch_l_joint", "hip_roll_l_joint", "hip_yaw_l_joint",
    "knee_pitch_l_joint", "ankle_pitch_l_joint", "ankle_roll_l_joint",
    "hip_pitch_r_joint", "hip_roll_r_joint", "hip_yaw_r_joint",
    "knee_pitch_r_joint", "ankle_pitch_r_joint", "ankle_roll_r_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "shoulder_pitch_l_joint", "shoulder_roll_l_joint", "shoulder_yaw_l_joint",
    "elbow_pitch_l_joint", "elbow_yaw_l_joint", "wrist_pitch_l_joint", "wrist_roll_l_joint",
    "shoulder_pitch_r_joint", "shoulder_roll_r_joint", "shoulder_yaw_r_joint",
    "elbow_pitch_r_joint", "elbow_yaw_r_joint", "wrist_pitch_r_joint", "wrist_roll_r_joint",
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rotation(wxyz):
    return Rotation.from_quat(np.asarray(wxyz)[..., [1, 2, 3, 0]])


def wxyz(rot):
    return rot.as_quat()[..., [3, 0, 1, 2]]


class Contract:
    def __init__(self, export):
        self.export = Path(export).resolve()
        self.manifest = json.loads((self.export / "export_manifest.json").read_text())
        self.mapping = json.loads((self.export / "robot_mapping.json").read_text())
        for name in ("model_encoder.onnx", "model_decoder.onnx", "robot_mapping.json", "training_config.yaml"):
            if sha256(self.export / name) != self.manifest["hashes"][name]:
                raise ValueError(f"Frozen SONIC export changed: {name}")
        for key, expected in {
            "encoder_input_shape": [1, 1751], "encoder_output_shape": [1, 64],
            "decoder_input_shape": [1, 994], "decoder_output_shape": [1, 29],
        }.items():
            if self.manifest[key] != expected:
                raise ValueError(f"Unsupported SONIC {key}: {self.manifest[key]}")
        self.names = self.manifest["policy_joint_order"]
        self.default = np.asarray(self.manifest["default_joint_angles_isaaclab"], np.float32)
        self.canonical = self.mapping["joint_order"]["mjcf"]
        for order in (self.names, self.canonical):
            if len(order) != 29 or set(order) != set(SERIAL_JOINT_NAMES):
                raise ValueError("SONIC must control exactly the 29 Tiangong serial joints")
        self.to_policy = [self.canonical.index(n) for n in self.names]
        self.to_mujoco = [self.names.index(n) for n in self.canonical]
        self.serial_to_policy = [SERIAL_JOINT_NAMES.index(n) for n in self.names]
        self.policy_to_serial = [self.names.index(n) for n in SERIAL_JOINT_NAMES]
        by_name = {j["name"]: j for j in self.mapping["joints"]}
        self.limits = np.asarray([by_name[n]["soft_limit_rad"] for n in self.names])
        self.effort = np.asarray([by_name[n]["effort_nm"] for n in self.names])
        self.velocity = np.asarray([by_name[n]["velocity_rad_s"] for n in self.names])
        gains = self.manifest["actuator_parameters"]
        if gains["joint_order"] != self.names:
            raise ValueError("Exported actuator parameters use a different joint order")
        self.kp = np.asarray(gains["stiffness"], dtype=float)
        self.kd = np.asarray(gains["damping"], dtype=float)
        for values in (self.kp, self.kd):
            if values.shape != (29,) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid exported PD parameters")
        if self.default.shape != (29,) or not np.isfinite(self.default).all():
            raise ValueError("Invalid default joint angles")
        if (self.limits.shape != (29, 2) or not np.isfinite(self.limits).all()
                or np.any(self.limits[:, 0] >= self.limits[:, 1])
                or np.any(self.default < self.limits[:, 0])
                or np.any(self.default > self.limits[:, 1])):
            raise ValueError("Invalid joint limits/default pose")
        self.gains_source = "export_manifest.json/actuator_parameters"
        self.fields = {f["name"]: slice(f["offset"], f["offset"] + f["size"])
                       for f in self.manifest["encoder_input_layout"]}
        sampling = self.manifest["reference_sampling"]
        self.dt = sampling["control_dt_seconds"]
        if (self.dt, sampling["num_future_frames"], sampling["dt_future_ref_frames"],
                sampling["reference_joint_order"], sampling["quaternion_order"]) != (.02, 10, .1, "mjcf", "wxyz"):
            raise ValueError("Unsupported SONIC reference timing/order")
        self.action_scale = float(self.manifest["action_scale"])
        self.action_clip = float(self.manifest["action_clip"])
        if (self.action_scale, self.action_clip) != (.25, 20.):
            raise ValueError("Unsupported SONIC action scaling")
        policy_layout = [(f["name"], f["shape"], f["offset"], f["size"])
                         for f in self.manifest["policy_input_layout"]]
        if policy_layout != [("base_ang_vel", [10, 3], 0, 30),
                             ("joint_pos", [10, 29], 30, 290),
                             ("joint_vel", [10, 29], 320, 290),
                             ("actions", [10, 29], 610, 290),
                             ("gravity_dir", [10, 3], 900, 30)]:
            raise ValueError("Unsupported SONIC history layout")
        expected_fields = {"mode_id": (0, 1), "encoder_index": (1, 4),
                           "command_multi_future_nonflat": (4, 584),
                           "motion_anchor_ori_b": (584, 590),
                           "motion_anchor_ori_b_mf_nonflat": (590, 650)}
        for name, (start, stop) in expected_fields.items():
            if self.fields.get(name) != slice(start, stop):
                raise ValueError(f"Unsupported SONIC encoder field: {name}")
        if self.manifest["encoder_modes"][0] != "g1":
            raise ValueError("SONIC mode 0 must be the robot reference branch")

    def target(self, raw_action):
        # Wrapper clips action before ActionManager; target clipping is separate.
        raw_action = np.asarray(raw_action)
        if raw_action.shape != (29,) or not np.isfinite(raw_action).all():
            raise RuntimeError("Invalid SONIC action")
        applied = np.clip(raw_action, -self.action_clip, self.action_clip)
        target = self.default + self.action_scale * applied
        return np.clip(target, self.limits[:, 0], self.limits[:, 1]), applied

    def encoder(self, q_future, v_future, quat_future, robot_quat):
        out = np.zeros(1751, np.float32)
        # CRITICAL: commands.py concatenates TWO FLATTENED histories, then the
        # observation reshapes to (10,58). Do NOT interleave q and v per frame.
        out[self.fields["command_multi_future_nonflat"]] = np.concatenate(
            [np.asarray(q_future).reshape(-1), np.asarray(v_future).reshape(-1)])
        relative = (rotation(robot_quat).inv() * rotation(quat_future)).as_matrix()
        ori6 = relative[..., :2].reshape(10, 6)  # first two COLUMNS, row-major
        out[self.fields["motion_anchor_ori_b"]] = ori6[0]
        out[self.fields["motion_anchor_ori_b_mf_nonflat"]] = ori6.reshape(-1)
        # All unused encoder branches deliberately zero; mode_id=0 is robot.
        return out


class History:
    """IsaacLab term-major histories: oldest to newest, reset repeats first value."""
    def __init__(self, contract):
        self.c = contract
        self.terms = None
        self.last = np.zeros(29, np.float32)

    def observe(self, q, dq, quat, angular_velocity):
        fields = [angular_velocity, np.asarray(q) - self.c.default, dq, self.last,
                  rotation(quat).inv().apply([0., 0., -1.])]
        fields = [np.asarray(x, np.float32) for x in fields]
        if self.terms is None:
            self.terms = [np.repeat(x[None], 10, axis=0) for x in fields]
        else:
            for buf, x in zip(self.terms, fields):
                buf[:-1] = buf[1:]
                buf[-1] = x
        return np.concatenate([x.reshape(-1) for x in self.terms])


class Reference:
    """50 Hz MotionLib-compatible robot reference and a pre-motion standing phase."""
    def __init__(self, path, contract, ground_height, motion_start=4., blend_duration=1.):
        import joblib

        self.c = contract
        self.path = Path(path).resolve()
        entry = joblib.load(self.path)  # Only use trusted local training datasets.
        if list(entry) != [self.path.stem]:
            raise ValueError("Expected a one-motion MotionLib file")
        entry = entry[self.path.stem]
        fps = float(entry["fps"])
        q = np.asarray(entry["dof"], np.float64)[:, contract.to_policy]
        quat = np.asarray(entry["root_rot"], np.float64)  # MotionLib file: xyzw
        pos = np.asarray(entry["root_trans_offset"], np.float64)
        if (q.shape[1:] != (29,) or quat.shape != (len(q), 4) or pos.shape != (len(q), 3)
                or len(q) < 3 or not np.isfinite(fps) or fps <= 0):
            raise ValueError("Invalid reference dimensions/fps")
        source_times = np.arange(len(q)) / fps
        # Humanoid_Batch.interploate_pose excludes the last source time.
        self.times = (source_times if fps == 50 else np.arange(0, source_times[-1], .02))
        if len(self.times) < 3:
            raise ValueError("Reference must contain at least three 50 Hz samples")
        self.q = np.stack([np.interp(self.times, source_times, x) for x in q.T], axis=1)
        self.pos = np.stack([np.interp(self.times, source_times, x) for x in pos.T], axis=1)
        rots = Slerp(source_times, Rotation.from_quat(quat))(self.times)
        # One fixed yaw alignment at initialization; never track actual drift.
        yaw = np.arctan2(rots[0].as_matrix()[1, 0], rots[0].as_matrix()[0, 0])
        align = Rotation.from_euler("z", -yaw)
        self.pos = align.apply(self.pos - self.pos[0])
        self.pos[:, 2] += ground_height
        self.quat = wxyz(align * rots)
        self.vel = np.diff(self.q, axis=0) / .02
        # Match the training code's penultimate finite difference for final frame.
        self.vel = np.concatenate([self.vel, self.vel[-2:-1]], axis=0)
        self.ground_height = ground_height
        self.start = motion_start
        self.blend = blend_duration
        if not all(np.isfinite(x).all() for x in (self.q, self.vel, self.quat, self.pos)):
            raise ValueError("Reference contains non-finite data")

    def sample(self, sim_times):
        ts = np.atleast_1d(sim_times).astype(float)
        qs, vs, ps, rs = [], [], [], []
        for t in ts:
            stand_pos = np.asarray([0., 0., self.ground_height])
            if t < self.start:
                q, v, p, r = self.c.default, np.zeros(29), stand_pos, np.array([1., 0., 0., 0.])
            elif self.blend > 0 and t < self.start + self.blend:
                u = np.clip((t - self.start) / self.blend, 0, 1)
                a, da = u*u*(3-2*u), 6*u*(1-u)/self.blend
                q = self.c.default + a * (self.q[0] - self.c.default)
                v = da * (self.q[0] - self.c.default)
                p = stand_pos + a * (self.pos[0] - stand_pos)
                r = wxyz(Rotation.from_rotvec(rotation(self.quat[0]).as_rotvec() * a))
            else:
                elapsed = max(0., t - self.start - self.blend)
                idx = min(int(np.floor(elapsed / .02 + 1e-6)), len(self.q) - 1)
                q, p, r = self.q[idx], self.pos[idx], self.quat[idx]
                v = self.vel[idx] if elapsed <= self.times[-1] else np.zeros(29)
            qs.append(q); vs.append(v); ps.append(p); rs.append(r)
        return tuple(np.asarray(x) for x in (qs, vs, ps, rs))

    def future(self, t):
        # Do not preview a moving command while the requested standing phase is active.
        times = np.full(10, t) if t < self.start else t + np.arange(10) * .1
        return self.sample(times)


def blend_pose(first, last, elapsed, duration):
    """Hermite joint transition preserving endpoint positions and velocities."""
    if elapsed <= 0:
        return first
    if elapsed >= duration:
        return last
    u = elapsed / duration
    q0, v0, p0, r0 = first
    q1, v1, p1, r1 = last
    q = ((2*u**3-3*u**2+1)*q0 + (u**3-2*u**2+u)*duration*v0
         + (-2*u**3+3*u**2)*q1 + (u**3-u**2)*duration*v1)
    v = ((6*u**2-6*u)/duration*q0 + (3*u**2-4*u+1)*v0
         + (-6*u**2+6*u)/duration*q1 + (3*u**2-2*u)*v1)
    weight = u*u*(3-2*u)
    relative = rotation(r0).inv() * rotation(r1)
    quat = wxyz(rotation(r0) * Rotation.from_rotvec(relative.as_rotvec()*weight))
    return q, v, p0 + weight*(p1-p0), quat


class LoopingReference:
    """Play the complete clip, then close its seam without a frame-zero jump."""
    def __init__(self, reference, blend_duration=1.):
        if not np.isfinite(blend_duration) or blend_duration <= 0:
            raise ValueError("Idle loop seam duration must be positive and finite")
        if reference.start != 0 or reference.blend != 0:
            raise ValueError("Idle loop requires an unprefixed motion reference")
        self.reference = reference
        self.duration = float(reference.times[-1])
        self.blend = float(blend_duration)
        self.period = self.duration + self.blend
        self.first = tuple(x[0].copy() for x in reference.sample(0.))
        self.last = tuple(x[0].copy() for x in reference.sample(self.duration))

    def sample(self, elapsed):
        phase = max(0., float(elapsed)) % self.period
        if phase < self.duration:
            return tuple(x[0] for x in self.reference.sample(phase))
        return blend_pose(self.last, self.first, phase-self.duration, self.blend)


class ReferencePlaylist:
    """Interactive references only: never write robot state or reset policy history.

    IDLE continuously tracks a dedicated loop, independently of clip selection.
    Stop and natural completion blend back to that loop. Future samples include
    the same loop seam and return transition as the current reference samples.
    """
    def __init__(self, references, ready_time=4., blend_duration=1., heading=0.,
                 *, idle_reference=None, idle_start=0., idle_loop_blend=1.):
        if not references or not np.isfinite(blend_duration) or blend_duration <= 0:
            raise ValueError("Interactive playback requires references and a positive blend")
        if not np.isfinite(ready_time) or ready_time < 0:
            raise ValueError("Interactive ready time must be nonnegative and finite")
        if not np.isfinite(idle_start) or idle_start < 0:
            raise ValueError("Idle start must be nonnegative and finite")
        self.references = references
        self.c = references[0].c
        self.ground_height = references[0].ground_height
        self.ready_time = max(ready_time, idle_start + blend_duration)
        self.blend = blend_duration
        self.selected = 0
        self.active = -1
        self.mode = "IDLE"
        self.events = []
        self.idle_loop = LoopingReference(idle_reference or references[0], idle_loop_blend)
        initial = (self.c.default.copy(), np.zeros(29),
                   np.array([0., 0., self.ground_height]), wxyz(Rotation.from_euler("z", heading)))
        self._set_idle_transition(initial, idle_start)

    def _event(self, t, action):
        event = dict(time=float(t), action=action, selected_motion_index=self.selected,
                     active_motion_index=self.active)
        self.events.append(event)

    def select(self, offset, t):
        self.selected = (self.selected + offset) % len(self.references)
        self._event(t, "select")

    def _set_idle_transition(self, origin, start):
        self.idle_origin = origin
        self.idle_start = start
        self.idle_epoch = start + self.blend
        mat = rotation(origin[3]).as_matrix()
        yaw = np.arctan2(mat[1, 0], mat[0, 0])
        self.idle_alignment = Rotation.from_euler("z", yaw)
        anchor = np.array([origin[2][0], origin[2][1], self.ground_height])
        self.idle_offset = anchor - self.idle_alignment.apply(self.idle_loop.first[2])

    def _loop_pose(self, elapsed):
        q, v, pos, quat = self.idle_loop.sample(elapsed)
        return (q, v, self.idle_alignment.apply(pos) + self.idle_offset,
                wxyz(self.idle_alignment * rotation(quat)))

    def _idle_pose(self, t):
        if t < self.idle_epoch:
            return self._transition(self.idle_origin, self._loop_pose(0.), t-self.idle_start)
        return self._loop_pose(t-self.idle_epoch)

    def _motion_pose(self, elapsed):
        q, v, pos, quat = (x[0] for x in self.references[self.active].sample(elapsed))
        return q, v, self.alignment.apply(pos) + self.offset, wxyz(self.alignment * rotation(quat))

    def _transition(self, first, last, elapsed):
        return blend_pose(first, last, elapsed, self.blend)

    def play(self, t):
        self.advance(t)
        if t + 1e-9 < self.ready_time:
            return f"Standing preparation: play is available at t={self.ready_time:g}s"
        if self.mode != "IDLE":
            return "Motion is active: return to IDLE before playing again"
        self.play_origin = self._idle_pose(t)
        self.active = self.selected
        self.started = t
        self.mode = "PLAYING"
        mat = rotation(self.play_origin[3]).as_matrix()
        self.alignment = Rotation.from_euler("z", np.arctan2(mat[1, 0], mat[0, 0]))
        initial_pos = self.references[self.active].pos[0]
        self.offset = self.play_origin[2] - self.alignment.apply(initial_pos)
        self.first = self._motion_pose(0.)
        self.return_start = t + self.blend + self.references[self.active].times[-1]
        self.last = self._motion_pose(self.references[self.active].times[-1])
        self._set_idle_transition(self.last, self.return_start)
        self._event(t, "play")
        return f"Playing {self.active + 1}/{len(self.references)}: {self.references[self.active].path.stem}"

    def stop(self, t):
        self.advance(t)
        if self.mode == "IDLE":
            return "Already IDLE"
        if self.mode == "RETURNING":
            return "Returning to IDLE"
        self.last = tuple(x[0].copy() for x in self.sample(t))
        self.return_start = t
        self._set_idle_transition(self.last, self.return_start)
        self.mode = "RETURNING"
        self._event(t, "stop")
        return "Returning to IDLE"

    def advance(self, t):
        if self.mode == "PLAYING" and t + 1e-9 >= self.return_start:
            self.mode = "RETURNING"
            self._event(self.return_start, "motion_finished")
        if self.mode == "RETURNING" and t + 1e-9 >= self.return_start + self.blend:
            self.mode = "IDLE"
            self.active = -1
            self._event(self.return_start + self.blend, "idle")

    def sample(self, sim_times):
        poses = []
        for t in np.atleast_1d(sim_times).astype(float):
            if self.mode == "IDLE":
                pose = self._idle_pose(t)
            elif self.mode == "RETURNING" or t >= self.return_start:
                pose = self._idle_pose(t)
            elif t < self.started + self.blend:
                pose = self._transition(self.play_origin, self.first, t-self.started)
            else:
                pose = self._motion_pose(t-self.started-self.blend)
            poses.append(pose)
        return tuple(np.asarray([pose[i] for pose in poses]) for i in range(4))

    def future(self, t):
        # IDLE previews its loop, never a selected but not yet started clip.
        return self.sample(t + np.arange(10) * .1)

    def status(self):
        selected = self.references[self.selected].path.stem
        playing = self.references[self.active].path.stem if self.active >= 0 else "none"
        idle = self.idle_loop.reference.path.stem
        return f"IDLE loop: {idle}\nSelected [{self.selected+1}/{len(self.references)}]: {selected}\nPlaying: {playing}"


class SonicPolicy:
    def __init__(self, contract):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.encoder = ort.InferenceSession(str(contract.export / "model_encoder.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        self.decoder = ort.InferenceSession(str(contract.export / "model_decoder.onnx"), opts,
                                            providers=["CPUExecutionProvider"])
        for session, input_shape, output_shape in ((self.encoder, [1, 1751], [1, 64]),
                                                    (self.decoder, [1, 994], [1, 29])):
            inputs, outputs = session.get_inputs(), session.get_outputs()
            if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape != input_shape
                    or outputs[0].shape != output_shape or inputs[0].type != "tensor(float)"):
                raise ValueError("ONNX input/output signature disagrees with SONIC manifest")

    def infer(self, encoder_input, proprio):
        encoder_input, proprio = np.asarray(encoder_input, np.float32), np.asarray(proprio, np.float32)
        if (encoder_input.shape != (1751,) or proprio.shape != (930,)
                or not np.isfinite(encoder_input).all() or not np.isfinite(proprio).all()):
            raise RuntimeError("Invalid SONIC observation")
        try:
            token = self.encoder.run(None, {self.encoder.get_inputs()[0].name: encoder_input[None]})[0]
            dec = np.concatenate([token.reshape(-1), proprio]).astype(np.float32)
            action = self.decoder.run(None, {self.decoder.get_inputs()[0].name: dec[None]})[0].reshape(29)
        except Exception as exc:
            raise RuntimeError(f"SONIC inference failed: {exc}") from exc
        if not np.isfinite(action).all():
            raise RuntimeError("Non-finite SONIC action")
        return action, token.reshape(-1), dec
