"""29-joint SONIC tracking over the existing RobotData/ROS motor interface."""
from pathlib import Path

import numpy as np
import yaml

from common.control_timing import TickDecimator
from FSM.fsm_base import FSMState, FSMStateName
from .sonic_runtime import Contract, History, Reference, ReferencePlaylist, SonicPolicy, rotation


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config") / "sonic_track.yaml"


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


class FSMStateSONICTRACK(FSMState):
    def __init__(self, robot_data, config_path=DEFAULT_CONFIG):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.SONICTRACK
        self.config_path = resolve_path(config_path)
        config = yaml.safe_load(self.config_path.read_text())
        self.contract = Contract(resolve_path(config["export_dir"]))
        self.dt = float(config["dt"])
        control = config["control"]
        self.clock = TickDecimator(control["decimation"], self.dt, robot_data.control_dt_)
        if not np.isclose(self.dt * self.clock.period_ticks, self.contract.dt, rtol=0, atol=1e-12):
            raise ValueError("SONICTRACK inference interval must equal the exported 0.02 s")
        if robot_data.motor_num != 29 or robot_data.q_a_s_.shape != (35,):
            raise ValueError("SONICTRACK requires RobotData with 29 joints and 6 floating DOFs")
        self.preparation = float(control["preparation_seconds"])
        self.stand = float(control["stand_seconds"])
        self.blend = float(control["blend_seconds"])
        if (not np.isfinite([self.preparation, self.stand, self.blend]).all()
                or self.preparation <= 0 or self.stand < 0 or self.blend <= 0):
            raise ValueError("Invalid SONICTRACK preparation/standing/blend durations")
        self.ready_time = self.preparation + self.stand
        paths = config["motion_files"]
        if not isinstance(paths, list) or not paths:
            raise ValueError("SONICTRACK motion_files must be a nonempty list")
        # Root translation is only carried by the reference, never used as robot
        # feedback or injected into MuJoCo. The actor needs orientation, not xyz.
        self.references = [Reference(resolve_path(p), self.contract, 0., 0., 0.) for p in paths]
        self.idle_reference = Reference(resolve_path(config["idle_motion_file"]), self.contract, 0., 0., 0.)
        self.idle_loop_blend = float(control["idle_loop_blend_seconds"])
        if not np.isfinite(self.idle_loop_blend) or self.idle_loop_blend <= 0:
            raise ValueError("idle_loop_blend_seconds must be positive and finite")
        self.ready_time = max(self.ready_time, self.preparation + self.blend)
        self.policy = SonicPolicy(self.contract)
        self.phase = "INACTIVE"
        self.last_block_reason = ""
        print(f"[SONICTRACK] Loaded step {self.contract.manifest['training_step']}, "
              f"29 joints / 50 Hz, {len(self.references)} motion(s)")

    def can_enter_from(self, state, flag):
        if state != FSMStateName.ZERO:
            self.last_block_reason = "Enter ZERO (z) before SONICTRACK (t)"
            return False
        return True

    def _feedback(self):
        c, rd = self.contract, self.robot_data_
        q = rd.get_joint_pos()[c.serial_to_policy]
        dq = rd.get_joint_vel()[c.serial_to_policy]
        quat = np.asarray(rd.get_robot_quat(), dtype=np.float64)
        omega = rd.get_angular_velocity()
        if not all(np.isfinite(x).all() for x in (q, dq, quat, omega)):
            raise RuntimeError("Non-finite SONICTRACK feedback")
        if quat.shape != (4,) or np.linalg.norm(quat) < 1e-8:
            raise RuntimeError("Invalid SONICTRACK base quaternion")
        return q, dq, quat / np.linalg.norm(quat), omega

    def on_enter(self):
        q, _, quat, _ = self._feedback()
        c, rd = self.contract, self.robot_data_
        mat = rotation(quat).as_matrix()
        heading = np.arctan2(mat[1, 0], mat[0, 0])
        self.playlist = ReferencePlaylist(
            self.references, self.ready_time, self.blend, heading,
            idle_reference=self.idle_reference, idle_start=self.preparation,
            idle_loop_blend=self.idle_loop_blend)
        self.history = History(c)
        self.clock.reset()
        self._ticks = 0
        self.inference_count = 0
        self.last_encoder = self.last_proprio = self.last_raw_action = None
        self._start_q = q.copy()
        self._start_kp = rd.joint_kp_s_[c.serial_to_policy].copy()
        self._start_kd = rd.joint_kd_s_[c.serial_to_policy].copy()
        if not np.isfinite(np.concatenate([self._start_kp, self._start_kd])).all():
            raise RuntimeError("Invalid gains on SONICTRACK entry")
        self._last_motion_event = getattr(rd.control_flag, "motion_event_id", 0)
        self._ready_announced = False
        self.phase = "PREPARING"
        rd.walk_cmd_[:] = 0.
        print("[SONICTRACK] Preparing nominal pose, then automatically tracking the IDLE loop")

    @property
    def elapsed(self):
        # Same nominal control ticks as the existing main-loop clock. Reference
        # playback does not depend on keyboard repeats or ROS publish count.
        return self._ticks * self.dt

    def _write_target(self, target, kp, kd):
        c, rd = self.contract, self.robot_data_
        if not all(np.isfinite(x).all() for x in (target, kp, kd)):
            raise RuntimeError("Invalid SONICTRACK motor target")
        rd.q_d_s_[-29:] = np.clip(target, *c.limits.T)[c.policy_to_serial]
        rd.q_dot_d_s_[-29:] = 0.
        rd.tau_d_s_[-29:] = 0.
        rd.joint_kp_s_[:] = kp[c.policy_to_serial]
        rd.joint_kd_s_[:] = kd[c.policy_to_serial]

    def _consume_motion_event(self, flag):
        event = getattr(flag, "motion_event_id", 0)
        if event == self._last_motion_event:
            return
        self._last_motion_event = event
        command = getattr(flag, "motion_command", "")
        if command == "play":
            print(f"[SONICTRACK] {self.playlist.play(self.elapsed)}")
        elif command == "idle":
            print(f"[SONICTRACK] {self.playlist.stop(self.elapsed)}")
        elif command in ("next", "previous"):
            self.playlist.select(1 if command == "next" else -1, self.elapsed)
            print(f"[SONICTRACK] {self.playlist.status()}")

    def run(self, flag):
        c = self.contract
        q, dq, quat, omega = self._feedback()
        self._consume_motion_event(flag)
        if self.elapsed + 1e-9 < self.preparation:
            u = np.clip(self.elapsed / self.preparation, 0., 1.)
            alpha = u*u*(3.-2.*u)
            self._write_target(self._start_q + alpha*(c.default-self._start_q),
                               self._start_kp + alpha*(c.kp-self._start_kp),
                               self._start_kd + alpha*(c.kd-self._start_kd))
        elif self.clock.step():
            previous_mode = self.playlist.mode
            self.playlist.advance(self.elapsed)
            self.phase = self.playlist.mode
            if previous_mode != self.phase:
                print(f"[SONICTRACK] {self.phase}")
            qref, vref, _, quatref = self.playlist.future(self.elapsed)
            encoder = c.encoder(qref, vref, quatref, quat)
            proprio = self.history.observe(q, dq, quat, omega)
            raw, _, _ = self.policy.infer(encoder, proprio)
            target, applied = c.target(raw)
            self._write_target(target, c.kp, c.kd)
            self.history.last = applied.copy()
            self.last_encoder, self.last_proprio, self.last_raw_action = encoder, proprio, raw
            self.inference_count += 1
        if not self._ready_announced and self.elapsed + 1e-9 >= self.ready_time:
            self._ready_announced = True
            print("[SONICTRACK] READY / IDLE LOOP: p plays selection; i returns to IDLE loop; c stops")
            print(f"[SONICTRACK] {self.playlist.status()}")
        self._ticks += 1

    def on_exit(self):
        self.phase = "INACTIVE"
        print("[SONICTRACK] Exit")

    def check_transition(self, flag):
        if not flag.enable or flag.fsm_state_command == "gotoSTOP":
            return FSMStateName.STOP
        if flag.fsm_state_command == "gotoZERO":
            return FSMStateName.ZERO
        return FSMStateName.SONICTRACK
