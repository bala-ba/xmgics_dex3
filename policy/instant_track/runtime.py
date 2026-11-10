"""Standalone InstantTrack deployment: no Torch, training checkout or ROS imports."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation

SERIAL_JOINT_NAMES = tuple(
    [f'{j}_{side}_joint' for side in ('l', 'r')
     for j in ('hip_pitch', 'hip_roll', 'hip_yaw', 'knee_pitch', 'ankle_pitch', 'ankle_roll')]
    + ['waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint']
    + [f'{j}_{side}_joint' for side in ('l', 'r')
       for j in ('shoulder_pitch', 'shoulder_roll', 'shoulder_yaw', 'elbow_pitch',
                 'elbow_yaw', 'wrist_pitch', 'wrist_roll')])


def rotation(wxyz):
    q = np.asarray(wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise RuntimeError('Invalid InstantTrack IMU quaternion')
    return Rotation.from_quat(q[[1, 2, 3, 0]]).as_matrix()


def observation(q, dq, omega_body, quat, previous, default, reference_rotation):
    R = rotation(quat)
    relative = R.T @ reference_rotation
    obs = np.concatenate((q-default, dq, omega_body, -R[2], previous, relative[:, :2].reshape(6)))
    if obs.shape != (99,) or not np.isfinite(obs).all():
        raise RuntimeError('Invalid InstantTrack observation')
    return obs.astype(np.float32)


def stationary_goal(q, root_R):
    """MotionLib.goal layout for ten identical, stationary future poses.

    A paused walking goal must not retain future translation or velocities.
    Rotation is expressed in the reference yaw frame, as in training.
    """
    yaw = np.arctan2(root_R[1, 0], root_R[0, 0])
    local_R = Rotation.from_euler('z', -yaw).as_matrix() @ root_R
    frame = np.concatenate((np.zeros(3), local_R[:, :2].reshape(6),
                            np.zeros(6), q, np.zeros(29)))
    goal = np.tile(frame, 10).astype(np.float32)
    if goal.shape != (730,) or not np.isfinite(goal).all():
        raise RuntimeError('Invalid stationary InstantTrack goal')
    return goal


class Runtime:
    def __init__(self, export):
        self.export = Path(export)
        self.manifest = m = json.loads((self.export/'manifest.json').read_text())
        if (m['format_version'], m['robot'], m['control_dt'], m['history'],
                m['observation_size'], m['goal_size']) != (1, 'tiangong', .02, 10, 99, 730):
            raise ValueError('Unsupported InstantTrack export contract')
        names = m['joint_names']
        if len(names) != 29 or set(names) != set(SERIAL_JOINT_NAMES):
            raise ValueError('InstantTrack export must contain exactly the 29 serial joints')
        self.serial_to_policy = np.array([SERIAL_JOINT_NAMES.index(n) for n in names])
        self.policy_to_serial = np.array([names.index(n) for n in SERIAL_JOINT_NAMES])
        files = ['policy.onnx', 'validation.npz'] + [ref['file'] for ref in m['motions']]
        if not m['motions'] or len(set(files)) != len(files):
            raise ValueError('Invalid InstantTrack motion list')
        for name in files:
            if Path(name).name != name:
                raise ValueError('Export payload must be a local filename')
            digest = hashlib.sha256((self.export/name).read_bytes()).hexdigest()
            if digest != m['hashes'].get(name):
                raise ValueError(f'InstantTrack export checksum mismatch: {name}')
        for attr, key in (('default', 'default_joint_pos'), ('scale', 'action_scale'),
                          ('kp', 'kp'), ('kd', 'kd'), ('effort', 'torque_limits')):
            value = np.asarray(m[key], dtype=np.float64)
            if value.shape != (29,) or not np.isfinite(value).all():
                raise ValueError(f'Invalid InstantTrack {key}')
            setattr(self, attr, value)
        if np.any(self.kp <= 0) or np.any(self.kd < 0) or np.any(self.scale <= 0) or np.any(self.effort <= 0):
            raise ValueError('Invalid InstantTrack gains/scale/effort')
        self.motions = []
        for ref in m['motions']:
            with np.load(self.export/ref['file'], allow_pickle=False) as data:
                motion = {k: data[k].copy() for k in data.files}
            n = ref['frames']
            for key, shape in (('q', (n, 29)), ('dq', (n, 29)), ('root_pos', (n, 3)),
                               ('root_quat', (n, 4)), ('goal', (n, 730)),
                               ('root_vel', (n, 3)), ('root_omega', (n, 3))):
                if n < 2 or motion[key].shape != shape or not np.isfinite(motion[key]).all():
                    raise ValueError(f'Invalid InstantTrack motion {key}')
            motion['root_R'] = np.stack([rotation(q) for q in motion['root_quat']])
            self.motions.append(motion)
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.export/'policy.onnx'), options,
                                           providers=['CPUExecutionProvider'])
        if [(x.name, x.shape, x.type) for x in self.session.get_inputs()] != [
                ('obs_history', [1, 10, 99], 'tensor(float)'), ('goal', [1, 730], 'tensor(float)')]:
            raise ValueError('Unexpected InstantTrack ONNX input signature')
        if [(x.name, x.shape) for x in self.session.get_outputs()] != [('actions', [1, 29])]:
            raise ValueError('Unexpected InstantTrack ONNX output signature')

    def infer(self, history, goal):
        if history.shape != (10, 99) or goal.shape != (730,) or not all(
                np.isfinite(x).all() for x in (history, goal)):
            raise RuntimeError('Invalid InstantTrack network input')
        try:
            action = self.session.run(['actions'], {
                'obs_history': history[None].astype(np.float32),
                'goal': goal[None].astype(np.float32)})[0][0]
        except Exception as exc:
            raise RuntimeError(f'InstantTrack inference failed: {exc}') from exc
        if action.shape != (29,) or not np.isfinite(action).all():
            raise RuntimeError('Invalid InstantTrack action')
        # Training has no action clip; the reference skip is already in ONNX.
        return action, self.default + self.scale*action
