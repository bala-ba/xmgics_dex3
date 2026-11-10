"""InstantTrack FSM: prepare, continuously track held/playing references, exit."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from common.control_timing import TickDecimator
from FSM.fsm_base import FSMState, FSMStateName
from .runtime import Runtime, observation, rotation, stationary_goal

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name('config') / 'instant_track.yaml'


class FSMStateINSTANTTRACK(FSMState):
    def __init__(self, robot_data, config_path=DEFAULT_CONFIG, runtime=None):
        super().__init__(robot_data)
        self.current_state_name = FSMStateName.INSTANTTRACK
        self.config_path = Path(config_path)
        cfg = yaml.safe_load(self.config_path.read_text())
        export = Path(cfg['export_dir']).expanduser()
        self.runtime = runtime if runtime is not None else Runtime(export if export.is_absolute() else ROOT/export)
        self.dt = float(cfg['dt'])
        self.clock = TickDecimator(cfg['control']['decimation'], self.dt, robot_data.control_dt_)
        if not np.isclose(self.dt*self.clock.period_ticks, self.runtime.manifest['control_dt'], rtol=0, atol=1e-12):
            raise ValueError('InstantTrack requires 50 Hz inference')
        if robot_data.motor_num != 29 or robot_data.q_a_s_.shape != (35,):
            raise ValueError('InstantTrack requires 29 serial joints')
        self.preparation = float(cfg['control']['preparation_seconds'])
        if not np.isfinite(self.preparation) or self.preparation <= 0:
            raise ValueError('Invalid InstantTrack preparation duration')
        self._stand_play_entry = False
        self.entry_joint_error = float(cfg['control']['entry_max_joint_error'])
        self.entry_tilt_error = float(cfg['control']['entry_max_tilt_error'])
        if any(not np.isfinite(x) or x <= 0 for x in (self.entry_joint_error, self.entry_tilt_error)):
            raise ValueError('Invalid InstantTrack entry tolerances')
        self.torque_limit_enabled = cfg['control']['torque_limit_enabled']
        if not isinstance(self.torque_limit_enabled, bool):
            raise ValueError('torque_limit_enabled must be boolean')
        self.selected = 0
        self.phase = 'INACTIVE'
        self.inference_count = 0
        print(f"[INSTANTTRACK] Loaded checkpoint {self.runtime.manifest['iteration']}, "
              f"29 joints / 50 Hz, {len(self.runtime.motions)} motion(s)")

    @property
    def motion(self):
        return self.runtime.motions[self.selected]

    def can_enter_from(self, state, flag):
        self.last_block_reason = 'Enter ZERO (z) or STAND (1) before INSTANTTRACK (b)'
        return state in (FSMStateName.ZERO, FSMStateName.STAND)

    def prepare_transition(self, state, flag):
        # A fresh play event is routed here by STAND. Explicit b keeps its
        # preparation sequence even if an older p is latched.
        self._stand_play_entry = (state == FSMStateName.STAND
                                  and flag.fsm_state_command != 'gotoINSTANTTRACK'
                                  and flag.motion_command == 'play')

    def _feedback(self):
        rd, r = self.robot_data_, self.runtime
        q = rd.get_joint_pos()[r.serial_to_policy]
        dq = rd.get_joint_vel()[r.serial_to_policy]
        omega = rd.get_angular_velocity()
        quat = np.asarray(rd.get_robot_quat())
        if not all(np.isfinite(x).all() for x in (q, dq, omega, quat)):
            raise RuntimeError('Non-finite InstantTrack feedback')
        rotation(quat)
        return q, dq, omega, quat

    def _prepare(self, direct_play=False, auto_play=False):
        rd, r = self.robot_data_, self.runtime
        q, _, _, quat = self._feedback()
        self._start_q = q.copy()
        R, ref = rotation(quat), self.motion['root_R'][0]
        yaw = np.arctan2(R[1, 0], R[0, 0]) - np.arctan2(ref[1, 0], ref[0, 0])
        self.heading = Rotation.from_euler('z', yaw).as_matrix()
        self._start_R = self.heading.T @ R
        self._prepare_rotvec = Rotation.from_matrix(self._start_R.T @ ref).as_rotvec()
        self._start_kp = rd.joint_kp_s_[r.serial_to_policy].copy()
        self._start_kd = rd.joint_kd_s_[r.serial_to_policy].copy()
        if any(not np.isfinite(x).all() or np.any(x < 0) for x in (self._start_kp, self._start_kd)):
            raise RuntimeError('Invalid InstantTrack entry gains')
        self._ticks = 0
        self.frame = 0
        self.last_frame = None
        self.history = None
        self.last_action = (np.asarray((rd.q_d_s_[-29:][r.serial_to_policy]-r.default)/r.scale,
                                      dtype=np.float32) if direct_play else np.zeros(29, np.float32))
        self.raw_target = q.copy()
        self.inference_count = 0
        self._entry_block_reported = False
        self._auto_play_after_prepare = auto_play
        self.clock.reset()
        if direct_play:
            joint_error, tilt_error = self._entry_errors(q, quat)
            if joint_error > self.entry_joint_error or tilt_error > self.entry_tilt_error:
                raise RuntimeError(f'STAND play entry rejected: leg/waist error {joint_error:.3f} rad, '
                                   f'tilt error {tilt_error:.3f} rad')
            self.phase = 'PLAYING'
            print('[INSTANTTRACK] PLAYING from frame 0 directly from STAND')
        else:
            self.phase = 'PREPARING'
            print(f"[INSTANTTRACK] PREPARING: policy ACTIVE, blending to {r.manifest['motions'][self.selected]['name']}; "
                  f"{'automatic playback armed' if auto_play else 'p to play when ready'}")

    def on_enter(self):
        self._last_event = self.robot_data_.control_flag.motion_event_id
        self._prepare(direct_play=self._stand_play_entry, auto_play=not self._stand_play_entry)
        self._stand_play_entry = False

    def on_reselect(self, flag):
        # Repeated b must not interrupt preparation or reset live policy history.
        if self.phase in ('READY', 'HOLD', 'FINISHED'):
            self._prepare(auto_play=True)
        elif self.phase == 'PREPARING':
            self._auto_play_after_prepare = True

    def _entry_errors(self, q, quat):
        R, ref = rotation(quat), self.motion['root_R'][0]
        # Active policy targets include torque-producing residuals, and arm
        # residuals can exceed the old all-joint PD preparation tolerance.
        # Gate reference playback on the supporting legs/waist and body tilt.
        supporting = self.runtime.serial_to_policy < 15
        joint_error = float(np.max(np.abs(q[supporting]-self.motion['q'][0, supporting])))
        tilt_error = float(np.arccos(np.clip(np.dot(R[2], ref[2]), -1., 1.)))
        return joint_error, tilt_error

    def _finish_preparation(self, q, quat):
        joint_error, tilt_error = self._entry_errors(q, quat)
        if joint_error > self.entry_joint_error or tilt_error > self.entry_tilt_error:
            if not self._entry_block_reported:
                print(f'[INSTANTTRACK] Preparation settling: leg/waist error {joint_error:.3f} rad, '
                      f'tilt error {tilt_error:.3f} rad; policy remains ACTIVE')
                self._entry_block_reported = True
            return False
        if self._auto_play_after_prepare:
            self._auto_play_after_prepare = False
            self.phase = 'PLAYING'
            print('[INSTANTTRACK] PLAYING from frame 0 automatically after preparation')
        else:
            self.phase = 'READY'
            print('[INSTANTTRACK] READY: policy ACTIVE at 50 Hz, holding frame 0; p plays, c STOP / z ZERO')
        return True

    def _event(self, flag, q, quat):
        if flag.motion_event_id == self._last_event:
            return
        self._last_event = flag.motion_event_id
        command = flag.motion_command
        if command == 'play' and self.phase in ('READY', 'HOLD'):
            if self.phase == 'READY':
                joint_error, tilt_error = self._entry_errors(q, quat)
                if joint_error > self.entry_joint_error or tilt_error > self.entry_tilt_error:
                    print(f'[INSTANTTRACK] Play blocked: leg/waist error {joint_error:.3f} rad, '
                          f'tilt error {tilt_error:.3f} rad; wait for policy tracking to settle')
                    return
            # READY/HOLD already infer continuously. Preserve heading, history,
            # previous action and the 50 Hz clock when only starting the reference.
            self.phase = 'PLAYING'
            print(f'[INSTANTTRACK] PLAYING from frame {self.frame}')
        elif command == 'idle' and self.phase == 'PREPARING':
            self._auto_play_after_prepare = False
            print('[INSTANTTRACK] Automatic playback cancelled; preparation will end in READY')
        elif command == 'play' and self.phase == 'PREPARING':
            self._auto_play_after_prepare = True
            print('[INSTANTTRACK] Automatic playback armed; waiting for preparation and entry checks')
        elif command == 'idle' and self.phase == 'PLAYING':
            self.phase = 'HOLD'
            print('[INSTANTTRACK] HOLD: reference paused, policy ACTIVE; p resumes, c stops')
        elif command in ('next', 'previous') and self.phase != 'PLAYING':
            self.selected = (self.selected + (1 if command == 'next' else -1)) % len(self.runtime.motions)
            self._prepare()
        elif command == 'play' and self.phase == 'FINISHED':
            print('[INSTANTTRACK] FINISHED: b prepares and plays again')

    def _write(self, target, kp, kd, q, dq):
        r, rd = self.runtime, self.robot_data_
        if not all(np.isfinite(x).all() for x in (target, kp, kd)):
            raise RuntimeError('Non-finite InstantTrack target/gains')
        if self.torque_limit_enabled:
            # Same serial PD bound as DEXAMP; preserves raw actions in history.
            lower, upper = np.full(29, -np.inf), np.full(29, np.inf)
            mask = kp > 0
            lower[mask] = q[mask] + (kd[mask]*dq[mask]-r.effort[mask])/kp[mask]
            upper[mask] = q[mask] + (kd[mask]*dq[mask]+r.effort[mask])/kp[mask]
            target = np.clip(target, lower, upper)
        rd.q_d_s_[-29:] = target[r.policy_to_serial]
        rd.q_dot_d_s_[-29:] = 0.
        rd.tau_d_s_[-29:] = 0.
        rd.joint_kp_s_[:] = kp[r.policy_to_serial]
        rd.joint_kd_s_[:] = kd[r.policy_to_serial]

    def run(self, flag):
        q, dq, omega, quat = self._feedback()
        r = self.runtime
        self._event(flag, q, quat)
        kp, kd = r.kp, r.kd
        prepared_q = prepared_R = None
        if self.phase == 'PREPARING':
            u = min(self._ticks*self.dt/self.preparation, 1.)
            blend = u*u*(3.-2.*u)
            prepared_q = self._start_q + blend*(self.motion['q'][0]-self._start_q)
            prepared_R = self._start_R @ Rotation.from_rotvec(blend*self._prepare_rotvec).as_matrix()
            kp = self._start_kp + blend*(r.kp-self._start_kp)
            kd = self._start_kd + blend*(r.kd-self._start_kd)
            self._ticks += 1
            if u >= 1.:
                self._finish_preparation(q, quat)
        if self.phase in ('PREPARING', 'READY', 'PLAYING', 'HOLD', 'FINISHED') and self.clock.step():
            if self.phase == 'PLAYING' and self.frame >= len(self.motion['q'])-1:
                self.phase = 'FINISHED'
                print('[INSTANTTRACK] FINISHED: policy ACTIVE at final pose; b prepares again')
            frame = (self.last_frame if self.phase == 'HOLD' and self.last_frame is not None
                     else self.frame)
            ref_R = prepared_R if prepared_R is not None else self.motion['root_R'][frame]
            ref = self.heading @ ref_R
            obs = observation(q, dq, omega, quat, self.last_action, r.default, ref)
            history = (np.repeat(obs[None], 10, axis=0) if self.history is None else
                       np.concatenate((self.history[1:], obs[None])))
            goal = (self.motion['goal'][frame] if self.phase == 'PLAYING' else
                    stationary_goal(prepared_q if prepared_q is not None else self.motion['q'][frame], ref_R))
            action, target = r.infer(history, goal)
            self.history = history
            self.last_action = action.copy()
            self.raw_target = target
            self.last_frame = frame
            if self.phase == 'PLAYING':
                self.frame += 1
            self.inference_count += 1
        self._write(self.raw_target, kp, kd, q, dq)

    def on_exit(self):
        self._auto_play_after_prepare = False
        self.phase = 'INACTIVE'

    def check_transition(self, flag):
        return {'gotoSTOP': FSMStateName.STOP, 'gotoZERO': FSMStateName.ZERO,
                'gotoINSTANTTRACK': FSMStateName.INSTANTTRACK}.get(flag.fsm_state_command)
