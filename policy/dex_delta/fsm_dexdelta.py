"""Dex15 Delta/MSL: zero-padded history, raw gait phase, deterministic actor."""
from pathlib import Path
import numpy as np
import onnxruntime as ort
import yaml
from common.control_timing import TickDecimator
from FSM.fsm_base import FSMState, FSMStateName


class FSMStateDEXDELTA(FSMState):
    def __init__(self, robot_data):
        super().__init__(robot_data)
        root = Path(__file__).resolve().parent
        with (root / 'config/dex_delta.yaml').open() as stream:
            cfg = yaml.safe_load(stream)
        self.motor_num = cfg['motor_num']
        self.action_num = cfg['actions_size']
        self.frames = cfg['size']['num_hist']
        self.width = cfg['size']['observations_size']
        if (self.motor_num, self.action_num, self.frames, self.width) != (29, 15, 15, 56):
            raise ValueError('DEXDELTA requires 29 motors, 15 actions and 15x56 observations')
        if robot_data.motor_num != self.motor_num:
            raise ValueError('DEXDELTA robot motor count mismatch')
        self.clock = TickDecimator(cfg['control']['decimation'], cfg['dt'], robot_data.control_dt_)
        self.policy_dt = cfg['control']['decimation'] * cfg['dt']
        if not any(np.isclose(self.policy_dt, dt) for dt in (.01, .02)):
            raise ValueError('DEXDELTA requires 100Hz or 50Hz inference; match the selected model training')

        def vector(value, size):
            arr = np.asarray(value, dtype=np.float32)
            if arr.shape != (size,) or not np.isfinite(arr).all():
                raise ValueError('Invalid DEXDELTA configuration vector')
            return arr

        self.joint_names = cfg['joint_names']
        motor_names = cfg['motor_joint_names']
        if len(self.joint_names) != 15 or len(set(self.joint_names)) != 15 or len(motor_names) != 29 or len(set(motor_names)) != 29:
            raise ValueError('Invalid DEXDELTA joint mapping')
        self.indices = np.array([motor_names.index(name) for name in self.joint_names])
        self.offset = vector(cfg['init_state']['default_joint_angles'], 15)
        self.scale = float(cfg['control']['action_scale'])
        self.hold = vector(cfg['hold_joint_angles'], 29)
        self.kp = vector(cfg['gains']['kp'], 29)
        self.kd = vector(cfg['gains']['kd'], 29)
        self.torque_limits = vector(cfg['gains']['torque_limits'], 29).astype(np.float64)
        self.torque_limit_enabled = cfg['control']['torque_limit_enabled']
        if not isinstance(self.torque_limit_enabled, bool):
            raise ValueError('DEXDELTA torque_limit_enabled must be boolean')
        if self.torque_limit_enabled and (np.any(self.kp <= 0) or np.any(self.torque_limits <= 0)):
            raise ValueError('DEXDELTA torque limiting requires positive kp and torque_limits')
        self.hold[self.indices] = self.offset
        norm = cfg['normalization']
        self.command_scale = vector(norm['obs_scales']['command'], 3)
        self.velocity_scale = float(norm['obs_scales']['dof_vel'])
        self.angular_scale = float(norm['obs_scales']['ang_vel'])
        self.obs_clip = float(norm['clip_scales']['clip_observations'])
        self.action_clip = float(norm['clip_scales']['clip_actions'])
        self.phase_offset = vector(cfg['clock']['initial_phase'], 2)
        self.period = float(cfg['clock']['period'])
        scalars = [self.scale, self.velocity_scale, self.angular_scale, self.obs_clip, self.action_clip, self.period]
        if not np.isfinite(scalars).all() or min(scalars) <= 0 or np.any(self.kp < 0) or np.any(self.kd < 0):
            raise ValueError('Invalid DEXDELTA scales/PD/clock')
        # These clips are also embedded in this exported graph.
        if self.obs_clip != 18 or self.action_clip != 18:
            raise ValueError('DEXDELTA ONNX has fixed observation/action clips of 18')
        model = root / 'model' / cfg['model_path']
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1 or inputs[0].shape != [1, 840] or outputs[0].shape != [1, 15] or inputs[0].type != 'tensor(float)' or outputs[0].type != 'tensor(float)':
            raise ValueError('Invalid DEXDELTA ONNX interface')
        self.input_name = inputs[0].name
        self.history = np.zeros((15, 56), dtype=np.float32)
        self.previous_action = np.zeros(15, dtype=np.float32)
        self.target = self.hold.copy()
        self.inference_count = 0
        self._infer(self.history)  # Warm up before the control thread starts.

    def on_enter(self):
        self.clock.reset()
        self.history.fill(0)
        self.previous_action.fill(0)
        self.target[:] = self.hold
        self.inference_count = 0
        print('[FSMStateDEXDELTA] enter')

    def _frame(self):
        rd = self.robot_data_
        # Integer inference count, not wall time: zero velocity never stops gait.
        phase = (self.inference_count * self.policy_dt / self.period + self.phase_offset) % 1.0
        frame = np.concatenate((
            phase, rd.get_walk_cmd() * self.command_scale,
            rd.get_joint_pos()[self.indices] - self.offset,
            rd.get_joint_vel()[self.indices] * self.velocity_scale,
            self.previous_action, rd.get_angular_velocity() * self.angular_scale,
            rd.get_project_gravity(),  # pelvis IMU, inverse-rotated unit gravity
        )).astype(np.float32)
        if frame.shape != (56,) or not np.isfinite(frame).all():
            raise FloatingPointError('Invalid DEXDELTA observation')
        return np.clip(frame, -self.obs_clip, self.obs_clip)

    def _infer(self, history):
        try:
            result = self.session.run(None, {self.input_name: history.reshape(1, 840)})[0]
        except Exception as exc:
            raise RuntimeError('DEXDELTA inference failed; refusing stale action') from exc
        if result.shape != (1, 15) or not np.isfinite(result).all():
            raise FloatingPointError('Invalid DEXDELTA action')
        return np.clip(result[0], -self.action_clip, self.action_clip)

    def _limit_position_target(self, q_target):
        # Serial joint coordinates, including joints held outside the policy.
        # Desired velocity and feedforward are zero. Recompute on every control tick.
        rd = self.robot_data_
        q = np.asarray(rd.get_joint_pos(), dtype=np.float64)
        dq = np.asarray(rd.get_joint_vel(), dtype=np.float64)
        if q.shape != (29,) or dq.shape != (29,) or not np.isfinite(q).all() or not np.isfinite(dq).all() or not np.isfinite(q_target).all():
            raise FloatingPointError("Invalid DEXDELTA torque-limit feedback/target")
        kp = self.kp.astype(np.float64)
        damping = self.kd.astype(np.float64) * dq
        lower = q + (damping - self.torque_limits) / kp
        upper = q + (damping + self.torque_limits) / kp
        return np.clip(q_target, lower, upper)

    def run(self, flag):
        if self.clock.step():
            # Commit history/action only after a valid new inference.
            history = np.concatenate((self.history[1:], self._frame()[None]), axis=0)
            action = self._infer(history)
            target = self.hold.copy()
            target[self.indices] = self.offset + self.scale * action
            if not np.isfinite(target).all():
                raise FloatingPointError('Invalid DEXDELTA target')
            self.history[:] = history
            self.previous_action[:] = action  # clipped raw action, before scale/offset
            self.target[:] = target
            self.inference_count += 1
        rd = self.robot_data_
        target = self._limit_position_target(self.target) if self.torque_limit_enabled else self.target
        rd.q_d_s_[-self.motor_num:] = target
        rd.q_dot_d_s_[-self.motor_num:] = 0
        rd.tau_d_s_[-self.motor_num:] = 0
        rd.joint_kp_s_[:] = self.kp
        rd.joint_kd_s_[:] = self.kd

    def on_exit(self):
        print('[FSMStateDEXDELTA] exit')

    def check_transition(self, flag):
        return {
            'gotoDEXDELTA': FSMStateName.DEXDELTA,
            'gotoSTOP': FSMStateName.STOP,
            'gotoZERO': FSMStateName.ZERO,
            'gotoWALKAMP': FSMStateName.WALKAMP,
            'gotoDEXAMP': FSMStateName.DEXAMP,
            'gotoMULTIAMP': FSMStateName.MULTIAMP,
            'gotoWALKDEPTH': FSMStateName.WALKDEPTH,
            'gotoWALKDEPTH100HZ': FSMStateName.WALKDEPTH100HZ,
        }.get(flag.fsm_state_command)
