"""Real-export numerical and FSM/ROS payload checks; no ROS graph or robot connection."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from builtin_interfaces.msg import Time
import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from common.joystick import ControlFlag
from common.robot_data import RobotData
from common.robot_interface import RobotInterfaceImpl
from common.stdin_keyboard_control import KeyboardController
from FSM.fsm_base import FSMStateName
from FSM.robot_fsm import RobotFSMImpl
from policy.zero.fsm_zero import FSMStateZero
from .fsm_instant_track import FSMStateINSTANTTRACK, ROOT
from .runtime import Runtime, observation, rotation, stationary_goal


class InstantTrackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = Runtime(ROOT/'policy/instant_track/model/ckpt_438000')

    def setUp(self):
        self.stdout = contextlib.redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.addCleanup(self.stdout.__exit__, None, None, None)
        self.rd = RobotData()
        self.rd.control_dt_ = .01
        self.rd.q_a_s_[-29:] = self.runtime.default[self.runtime.policy_to_serial]
        zero = FSMStateZero(self.rd)
        self.rd.joint_kp_s_[:] = zero.kp_pos_
        self.rd.joint_kd_s_[:] = zero.kd_pos_
        self.state = FSMStateINSTANTTRACK(self.rd, runtime=self.runtime)
        self.state.on_enter()

    def ready(self):
        # i cancels b's automatic playback, preserving a stationary READY path.
        self.rd.control_flag.motion_command = 'idle'
        self.rd.control_flag.motion_event_id += 1
        self.reach_reference()
        for _ in range(201):
            self.state.run(self.rd.control_flag)
        self.assertEqual(self.state.phase, 'READY')

    def reach_reference(self):
        r = self.runtime
        self.rd.q_a_s_[-29:] = self.state.motion['q'][0][r.policy_to_serial]
        self.rd.imu_data_[:3] = Rotation.from_matrix(self.state.motion['root_R'][0]).as_euler('xyz')[::-1]

    def event(self, command):
        flag = self.rd.control_flag
        flag.motion_command = command
        flag.motion_event_id += 1
        self.state.run(flag)

    def test_original_training_observation_and_torch_policy_vectors(self):
        r = self.runtime
        with np.load(r.export/'validation.npz', allow_pickle=False) as vectors:
            for i, frame in enumerate(vectors['frame']):
                ref = r.motions[int(vectors['motion'][i])]
                actual = observation(vectors['q'][i], vectors['dq'][i], vectors['omega'][i],
                                     vectors['quat'][i], vectors['previous'][i], r.default, ref['root_R'][frame])
                np.testing.assert_allclose(actual, vectors['obs'][i], rtol=1e-5, atol=1e-6)
                action, target = r.infer(vectors['history'][i], ref['goal'][frame])
                np.testing.assert_allclose(action, vectors['action'][i], rtol=2e-5, atol=2e-5)
                np.testing.assert_allclose(target, r.default+r.scale*vectors['action'][i], atol=1e-5)
        self.assertEqual(r.manifest['iteration'], 438000)
        self.assertTrue(r.manifest['reference_residual'])

    def test_heading_alignment_and_six_dimensional_rotation_layout(self):
        R = Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix()
        ref = Rotation.from_euler('xyz', [-.2, .1, .8]).as_matrix()
        yaw = Rotation.from_euler('z', 1.3).as_matrix()
        quat = lambda mat: Rotation.from_matrix(mat).as_quat()[[3, 0, 1, 2]]
        q = self.runtime.default
        a = observation(q, q*0, np.ones(3), quat(R), q*0, q, ref)
        b = observation(q, q*0, np.ones(3), quat(yaw@R), q*0, q, yaw@ref)
        np.testing.assert_allclose(a, b, atol=1e-6)
        np.testing.assert_allclose(a[-6:], (R.T@ref)[:, :2].reshape(6), atol=1e-6)

    def test_prepare_play_50hz_history_pause_resume_and_end(self):
        self.event('play')
        self.assertEqual(self.state.phase, 'PREPARING')
        self.assertEqual(self.state.inference_count, 1)
        np.testing.assert_array_equal(self.state.history, np.repeat(self.state.history[:1], 10, axis=0))
        self.assertFalse(self.state.history[-1, 64:93].any())
        self.reach_reference()
        for _ in range(200):
            self.state.run(self.rd.control_flag)
        self.assertEqual(self.state.phase, 'PLAYING')
        s = self.state
        self.assertEqual((s.frame, s.inference_count), (1, 101))
        count = s.inference_count
        previous = s.last_action.copy()
        self.event('play')
        self.assertEqual(s.inference_count, count)
        s.run(self.rd.control_flag)
        self.assertEqual((s.frame, s.inference_count), (2, count+1))
        np.testing.assert_array_equal(s.history[-1, 64:93], previous)
        self.event('idle')
        frame, count, held = s.frame, s.inference_count, s.last_frame
        for _ in range(20):
            s.run(self.rd.control_flag)
        self.assertEqual((s.phase, s.frame), ('HOLD', frame))
        self.assertEqual(s.last_frame, held)
        self.assertEqual(s.inference_count, count+10)
        self.event('play')
        s.run(self.rd.control_flag)
        self.assertEqual((s.phase, s.frame), ('PLAYING', frame+1))
        s.on_reselect(self.rd.control_flag)
        self.assertEqual(s.frame, frame+1)
        s.frame = len(s.motion['q'])-2
        s.clock.reset()
        s.run(self.rd.control_flag)
        self.assertEqual(s.last_frame, len(s.motion['q'])-2)
        s.run(self.rd.control_flag)
        s.run(self.rd.control_flag)
        self.assertEqual(s.phase, 'FINISHED')
        self.assertEqual(s.last_frame, len(s.motion['q'])-1)
        count = s.inference_count
        for _ in range(20):
            s.run(self.rd.control_flag)
        self.assertEqual(s.inference_count, count+10)
        s.on_reselect(self.rd.control_flag)
        self.assertEqual((s.phase, s.frame), ('PREPARING', 0))

    def test_ready_continuously_infers_fixed_pose_with_live_feedback(self):
        self.ready()
        s = self.state
        count = s.inference_count
        heading, target = s.heading.copy(), s.raw_target.copy()
        for _ in range(20):
            s.run(self.rd.control_flag)
        self.assertEqual((s.phase, s.frame, s.inference_count), ('READY', 0, count+10))
        self.rd.imu_data_[2] += .1
        for _ in range(20):
            s.run(self.rd.control_flag)
        self.assertFalse(np.allclose(s.raw_target, target))
        self.assertEqual((s.frame, s.inference_count), (0, count+20))
        np.testing.assert_array_equal(s.heading, heading)
        s.on_reselect(self.rd.control_flag)
        self.assertEqual((s.phase, s.inference_count), ('PREPARING', 0))
        self.assertTrue(s._auto_play_after_prepare)

    def test_stationary_goal_has_zero_motion_and_repeated_pose(self):
        R = Rotation.from_euler('xyz', [.1, -.2, .8]).as_matrix()
        q = self.runtime.motions[0]['q'][0]
        goal = stationary_goal(q, R).reshape(10, 73)
        expected_R = Rotation.from_euler('z', -.8).as_matrix() @ R
        np.testing.assert_allclose(goal[:, 3:9], np.tile(expected_R[:, :2].reshape(6), (10, 1)), atol=1e-7)
        np.testing.assert_allclose(goal[:, 15:44], np.tile(q, (10, 1)), atol=1e-7)
        np.testing.assert_array_equal(goal[:, :3], 0)
        np.testing.assert_array_equal(goal[:, 9:15], 0)
        np.testing.assert_array_equal(goal[:, 44:], 0)

    def test_automatic_play_waits_for_entry_pose_while_inference_continues(self):
        self.rd.q_a_s_[6] += 1.
        for _ in range(220):
            self.state.run(self.rd.control_flag)
        self.assertEqual((self.state.phase, self.state.inference_count), ('PREPARING', 110))
        self.reach_reference()
        self.rd.imu_data_[2] += 1.
        self.state.run(self.rd.control_flag)
        self.assertEqual(self.state.phase, 'PREPARING')
        self.reach_reference()
        self.state.run(self.rd.control_flag)
        self.assertEqual((self.state.phase, self.state.inference_count), ('PLAYING', 111))

    def test_pd_limit_recomputed_on_hold_ticks_without_changing_raw_action(self):
        self.ready()
        self.event('play')
        s, r = self.state, self.runtime
        s.run(self.rd.control_flag)
        raw = s.last_action.copy()
        s.raw_target = np.full(29, 100.)
        self.rd.q_dot_a_s_[-29:] = np.linspace(-1, 1, 29)
        s.run(self.rd.control_flag)  # held inference tick
        q, dq = self.rd.get_joint_pos(), self.rd.get_joint_vel()
        tau = self.rd.joint_kp_s_*(self.rd.q_d_s_[-29:]-q)-self.rd.joint_kd_s_*dq
        np.testing.assert_allclose(tau, r.effort[r.policy_to_serial], atol=1e-5)
        np.testing.assert_array_equal(raw, s.last_action)
        self.assertTrue(np.all(s.raw_target == 100.))

    def test_play_requires_support_pose_and_upright_tilt(self):
        self.ready()
        self.rd.q_a_s_[6] += 1.
        self.event('play')
        self.assertEqual(self.state.phase, 'READY')
        self.reach_reference()
        self.rd.imu_data_[2] += 1.
        self.event('play')
        self.assertEqual(self.state.phase, 'READY')
        self.reach_reference()
        # Arm residuals are expected under a live policy and must not prevent p.
        self.rd.q_a_s_[-1] += .6
        self.event('play')
        self.assertEqual(self.state.phase, 'PLAYING')

    def test_nonfinite_feedback_and_inference_do_not_commit_targets(self):
        self.ready()
        count = self.state.inference_count
        before = self.rd.q_d_s_.copy()
        self.rd.q_a_s_[6] = np.nan
        with self.assertRaises(RuntimeError):
            self.state.run(self.rd.control_flag)
        np.testing.assert_array_equal(self.rd.q_d_s_, before)
        self.reach_reference()
        self.state.clock.reset()
        with patch.object(self.runtime, 'session') as session:
            session.run.return_value = [np.full((1, 29), np.nan)]
            with self.assertRaises(RuntimeError):
                self.state.run(self.rd.control_flag)
        np.testing.assert_array_equal(self.rd.q_d_s_, before)
        self.assertEqual(self.state.inference_count, count)

    def test_keyboard_fsm_entry_and_stop_zero_exit(self):
        cfg = yaml.safe_load((ROOT/'config/dex_config.yaml').read_text())
        cfg['enabled_policy_states'] = []
        cfg['sonic_track']['enabled'] = False
        with patch('FSM.robot_fsm.FSMStateINSTANTTRACK', return_value=self.state):
            fsm = RobotFSMImpl(self.rd, cfg)
        keyboard = KeyboardController()
        keyboard._process_key('b')
        fsm.run_fsm(keyboard.keyboard_flag)
        self.assertEqual(fsm.current_state, FSMStateName.STOP)
        keyboard._process_key('z')
        fsm.run_fsm(keyboard.keyboard_flag)
        keyboard._process_key('b')
        fsm.run_fsm(keyboard.keyboard_flag)
        self.assertEqual(fsm.current_state, FSMStateName.INSTANTTRACK)
        keyboard._process_key('z')
        fsm.run_fsm(keyboard.keyboard_flag)
        self.assertEqual(fsm.current_state, FSMStateName.ZERO)
        keyboard._process_key('b')
        fsm.run_fsm(keyboard.keyboard_flag)
        keyboard._process_key('c')
        fsm.run_fsm(keyboard.keyboard_flag)
        self.assertEqual(fsm.current_state, FSMStateName.STOP)
        self.assertEqual(self.state.phase, 'INACTIVE')

    def test_stand_can_enter_instanttrack_without_returning_to_zero(self):
        cfg = yaml.safe_load((ROOT/'config/dex_config.yaml').read_text())
        cfg['enabled_policy_states'] = ['STAND']
        cfg['sonic_track']['enabled'] = False
        with patch('FSM.robot_fsm.FSMStateINSTANTTRACK', return_value=self.state):
            fsm = RobotFSMImpl(self.rd, cfg)
        keyboard = KeyboardController()

        def press(key):
            keyboard._process_key(key)
            self.rd.control_flag = keyboard.get_keyboard_flag()
            fsm.run_fsm(self.rd.control_flag)

        press('z')
        press('1')
        self.assertEqual(fsm.current_state, FSMStateName.STAND)
        before = self.rd.get_joint_pos().copy()
        gains = self.rd.joint_kp_s_.copy()
        with patch.object(self.runtime, 'infer', wraps=self.runtime.infer) as infer:
            press('b')
            goal = infer.call_args.args[1].reshape(10, 73)
            np.testing.assert_allclose(goal[:, 15:44],
                                       np.tile(before[self.runtime.serial_to_policy], (10, 1)), atol=1e-7)
        self.assertEqual(fsm.current_state, FSMStateName.INSTANTTRACK)
        self.assertEqual(self.state.phase, 'PREPARING')
        self.assertEqual(self.state.inference_count, 1)
        self.assertTrue(np.isfinite(self.rd.q_d_s_).all())
        np.testing.assert_allclose(self.rd.joint_kp_s_, gains)
        self.assertTrue(self.state.can_enter_from(FSMStateName.STAND, self.rd.control_flag))
        self.assertFalse(self.state.can_enter_from(FSMStateName.STOP, self.rd.control_flag))

        # b alone must start playback at the end of preparation. A repeated b
        # midway through preparation must not reset history or the timer.
        self.reach_reference()
        for _ in range(99):
            fsm.run_fsm(self.rd.control_flag)
        press('b')
        self.assertEqual((self.state._ticks, self.state.inference_count), (101, 51))
        for _ in range(99):
            fsm.run_fsm(self.rd.control_flag)
        history = self.state.history.copy()
        previous = self.state.last_action.copy()
        with patch.object(self.runtime, 'infer', wraps=self.runtime.infer) as infer:
            fsm.run_fsm(self.rd.control_flag)
            np.testing.assert_array_equal(infer.call_args.args[1], self.state.motion['goal'][0])
        self.assertEqual((self.state.phase, self.state.frame, self.state.inference_count),
                         ('PLAYING', 1, 101))
        self.assertNotEqual(self.rd.control_flag.motion_command, 'play')
        np.testing.assert_array_equal(self.state.history[:-1], history[1:])
        np.testing.assert_array_equal(self.state.history[-1, 64:93], previous)

    def test_preparation_play_event_rearms_cancelled_autoplay(self):
        self.event('idle')
        self.assertFalse(self.state._auto_play_after_prepare)
        self.event('play')
        self.assertTrue(self.state._auto_play_after_prepare)
        self.assertEqual(self.state._ticks, 2)
        self.reach_reference()
        for _ in range(199):
            self.state.run(self.rd.control_flag)
        self.assertEqual((self.state.phase, self.state.frame), ('PLAYING', 1))

    def test_stand_play_starts_immediately_once(self):
        cfg = yaml.safe_load((ROOT/'config/dex_config.yaml').read_text())
        cfg['enabled_policy_states'] = ['STAND']
        cfg['sonic_track']['enabled'] = False
        with patch('FSM.robot_fsm.FSMStateINSTANTTRACK', return_value=self.state):
            fsm = RobotFSMImpl(self.rd, cfg)
        keyboard = KeyboardController()

        def tick(key=None):
            if key is not None:
                keyboard._process_key(key)
            self.rd.control_flag = keyboard.get_keyboard_flag()
            fsm.run_fsm(self.rd.control_flag)

        tick('p')  # A play event outside STAND must not become a deferred start.
        tick('z')
        tick('1')
        tick()
        self.assertEqual(fsm.current_state, FSMStateName.STAND)
        s = self.state
        previous = ((self.rd.q_d_s_[-29:][self.runtime.serial_to_policy]-self.runtime.default)
                    /self.runtime.scale).astype(np.float32)
        with patch.object(self.runtime, 'infer', wraps=self.runtime.infer) as infer:
            tick('p')
            np.testing.assert_array_equal(infer.call_args.args[1], s.motion['goal'][0])
        self.assertEqual((fsm.current_state, s.phase, s.frame, s.inference_count),
                         (FSMStateName.INSTANTTRACK, 'PLAYING', 1, 1))
        np.testing.assert_allclose(self.rd.joint_kp_s_[self.runtime.serial_to_policy],
                                   self.runtime.kp)
        np.testing.assert_array_equal(s.history, np.repeat(s.history[:1], 10, axis=0))
        np.testing.assert_array_equal(s.history[-1, 64:93], previous)
        tick('p')
        self.assertEqual((s.frame, s.inference_count), (1, 1))
        tick()
        self.assertEqual((s.frame, s.inference_count), (2, 2))
        tick('z')
        tick('1')
        tick('b')  # A latched earlier p must not change b's 2 s hold entry.
        self.assertEqual(s.phase, 'PREPARING')

    def test_stand_immediate_play_rejects_bad_entry_pose(self):
        flag = self.rd.control_flag
        flag.fsm_state_command = 'gotoSTAND'
        flag.motion_command = 'play'
        self.state.prepare_transition(FSMStateName.STAND, flag)
        self.rd.q_a_s_[6] += 1.
        before = self.rd.q_d_s_.copy()
        with self.assertRaisesRegex(RuntimeError, 'STAND play entry rejected'):
            self.state.on_enter()
        np.testing.assert_array_equal(self.rd.q_d_s_, before)
        self.assertEqual(self.state.inference_count, 0)

    def test_all_29_ros_commands_and_real_ankle_gain_override(self):
        self.ready()
        self.event('play')
        interface = RobotInterfaceImpl(self.rd, str(ROOT/'config/dex_config.yaml'))
        interface.node = SimpleNamespace(get_logger=lambda: Mock(),
                                        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time)))
        interface.load_config()
        interface.initialized = True
        interface.sim = True
        interface.update_param(FSMStateName.INSTANTTRACK)
        captured = []
        for name in ('pub_leg_motor_cmd', 'pub_waist_motor_cmd', 'pub_arm_motor_cmd'):
            setattr(interface, name, SimpleNamespace(publish=captured.append))
        interface.send_motor_commands(self.rd.control_flag)
        self.assertEqual([len(m.cmds) for m in captured], [12, 3, 14])
        commands = [c for m in captured for c in m.cmds]
        self.assertEqual([c.name for c in commands],
                         [51, 52, 53, 54, 55, 56, 61, 62, 63, 64, 65, 66, 33, 32, 31,
                          11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27])
        np.testing.assert_allclose([c.pos for c in commands], self.rd.q_d_s_[-29:], atol=1e-6)
        np.testing.assert_allclose([c.kp for c in commands], self.rd.joint_kp_s_, atol=1e-5)
        np.testing.assert_allclose([c.kd for c in commands], self.rd.joint_kd_s_, atol=1e-5)
        self.assertTrue(all(c.tor == 0 and c.spd == 0 for c in commands))
        # Verify the hardware branch without touching a robot or the actual solver.
        interface.sim = False
        interface.fun_s2p = Mock()
        interface.fun_s2p.get_p_des.return_value = (True, np.zeros(4), np.zeros(4), np.zeros(4))
        before = self.rd.q_d_s_.copy()
        captured.clear()
        interface.send_motor_commands(self.rd.control_flag)
        commands = [c for m in captured for c in m.cmds]
        for i in (4, 5, 10, 11):
            self.assertEqual((commands[i].kp, commands[i].kd, commands[i].tor), (15., 1.25, 0.))
        np.testing.assert_array_equal(before, self.rd.q_d_s_)

    def test_corrupt_export_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            m = dict(self.runtime.manifest)
            (path/'manifest.json').write_text(json.dumps(m))
            (path/'policy.onnx').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                Runtime(path)


if __name__ == '__main__':
    unittest.main()
