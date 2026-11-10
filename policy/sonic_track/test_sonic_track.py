"""Integration checks using the configured real export, without a robot/ROS graph.

Run from xmigcs after sourcing ROS: .venv/bin/python -m unittest
policy.sonic_track.test_sonic_track -v
"""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import joblib
import numpy as np
from scipy.spatial.transform import Rotation
import yaml
from builtin_interfaces.msg import Time

from common.joystick import ControlFlag
from common.robot_data import RobotData
from common.robot_interface import RobotInterfaceImpl
from common.stdin_keyboard_control import KeyboardController
from FSM.fsm_base import FSMStateName
from FSM.robot_fsm import RobotFSMImpl
from policy.sonic_track.fsm_sonic_track import FSMStateSONICTRACK, ROOT
from policy.sonic_track.sonic_runtime import (
    Contract, History, Reference, ReferencePlaylist, LoopingReference, SERIAL_JOINT_NAMES, rotation, wxyz,
)


class SonicIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rd = RobotData()
        cls.rd.control_dt_ = .01
        with contextlib.redirect_stdout(io.StringIO()):
            cls.state = FSMStateSONICTRACK(cls.rd)
        cls.c = cls.state.contract

    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)
        rd, c = self.rd, self.c
        rd.imu_data_[:] = 0.
        rd.q_a_s_[-29:] = c.default[c.policy_to_serial]
        rd.q_dot_a_s_[:] = 0.
        rd.joint_kp_s_[:] = c.kp[c.policy_to_serial]
        rd.joint_kd_s_[:] = c.kd[c.policy_to_serial]
        rd.control_flag = ControlFlag(fsm_state_command="gotoSONICTRACK")
        self.state.on_enter()

    def tick(self, count, flag=None):
        for _ in range(count):
            self.state.run(flag or self.rd.control_flag)

    def ready(self):
        self.tick(round(self.state.ready_time / self.state.dt) + 1)

    def test_export_golden_vectors_real_onnx(self):
        with np.load(self.c.export / "validation_vectors.npz") as vectors:
            for enc, dec, token_ref, action_ref in zip(
                    vectors["encoder_input"], vectors["decoder_input"],
                    vectors["token_reference"], vectors["action_reference"]):
                action, token, _ = self.state.policy.infer(enc, dec[64:])
                np.testing.assert_array_equal(token, token_ref)
                np.testing.assert_allclose(action, action_ref, atol=1e-4, rtol=1e-5)

    def test_encoder_uses_q_then_dq_and_full_relative_orientation(self):
        q = np.arange(290).reshape(10, 29)
        dq = q + 1000
        robot = Rotation.from_euler("xyz", [.2, -.3, .7])
        ref = Rotation.from_euler("xyz", np.tile([-.1, .5, .9], (10, 1)))
        x = self.c.encoder(q, dq, wxyz(ref), wxyz(robot))
        np.testing.assert_array_equal(x[4:294], q.ravel())
        np.testing.assert_array_equal(x[294:584], dq.ravel())
        expected = (robot.inv()*ref).as_matrix()[..., :2].reshape(-1)
        np.testing.assert_allclose(x[590:650], expected, atol=1e-7)
        self.assertFalse(x[650:].any())

    def test_history_term_order_and_clipped_previous_action(self):
        h = History(self.c)
        h.observe(self.c.default, np.zeros(29), [1, 0, 0, 0], [1, 2, 3])
        _, h.last = self.c.target(np.arange(29.)*3 - 40)
        x = h.observe(self.c.default + .1, np.ones(29), [1, 0, 0, 0], [4, 5, 6])
        np.testing.assert_array_equal(x[:30].reshape(10, 3)[-2:], [[1, 2, 3], [4, 5, 6]])
        np.testing.assert_allclose(x[30:320].reshape(10, 29)[-1], .1, atol=1e-7)
        np.testing.assert_array_equal(x[610:900].reshape(10, 29)[-1], np.clip(np.arange(29.)*3-40, -20, 20))
        np.testing.assert_array_equal(x[900:].reshape(10, 3), np.tile([0, 0, -1], (10, 1)))

    def test_fifty_hz_hold_and_all_joint_outputs(self):
        self.tick(100)
        self.assertEqual(self.state.inference_count, 0)
        self.tick(1)
        self.assertEqual(self.state.inference_count, 1)
        target = self.rd.q_d_s_.copy()
        history = self.state.last_proprio.copy()
        self.tick(1)
        np.testing.assert_array_equal(self.rd.q_d_s_, target)
        np.testing.assert_array_equal(self.state.last_proprio, history)
        self.tick(98)
        self.assertEqual(self.state.inference_count, 50)
        expected, applied = self.c.target(self.state.last_raw_action)
        np.testing.assert_allclose(self.rd.q_d_s_[-29:], expected[self.c.policy_to_serial])
        np.testing.assert_array_equal(self.rd.joint_kp_s_, self.c.kp[self.c.policy_to_serial])
        np.testing.assert_array_equal(self.rd.joint_kd_s_, self.c.kd[self.c.policy_to_serial])
        np.testing.assert_array_equal(self.state.history.last, applied)
        self.assertFalse(self.rd.q_dot_d_s_[-29:].any())
        self.assertFalse(self.rd.tau_d_s_[-29:].any())

    def test_event_consumed_once_and_reselect_preserves_history(self):
        self.ready()
        h = self.state.history
        f = self.rd.control_flag
        f.motion_command, f.motion_event_id = "play", 1
        self.tick(1)
        started = self.state.playlist.started
        self.tick(30)
        self.assertEqual(self.state.playlist.started, started)
        self.assertEqual(len([e for e in self.state.playlist.events if e['action']=='play']), 1)
        f.motion_command, f.motion_event_id = "idle", 2
        self.tick(1)
        self.assertEqual(self.state.playlist.mode, "RETURNING")
        self.tick(102)
        self.assertEqual(self.state.playlist.mode, "IDLE")
        self.assertIs(self.state.history, h)
        f.motion_command, f.motion_event_id = "play", 3
        self.tick(1)
        self.assertEqual(self.state.playlist.mode, "PLAYING")
        self.assertIs(self.state.history, h)

    def test_yaw_aligned_once_and_not_reanchored(self):
        self.rd.imu_data_[0] = .8
        self.state.on_enter()
        self.ready()
        alignment = self.state.playlist.idle_alignment.as_matrix().copy()
        np.testing.assert_allclose(alignment, Rotation.from_euler('z', .8).as_matrix(), atol=1e-7)
        self.rd.imu_data_[0] = 1.0
        self.tick(2)
        ref_quat = self.state.playlist.future(self.state.elapsed-self.state.dt)[3][0]
        expected = (Rotation.from_euler('z', -1.) * rotation(ref_quat)).as_matrix()[:, :2].ravel()
        np.testing.assert_allclose(self.state.last_encoder[590:596], expected, atol=1e-7)
        np.testing.assert_array_equal(self.state.playlist.idle_alignment.as_matrix(), alignment)

    def test_idle_tracks_the_requested_motion_without_play(self):
        self.ready()
        playlist = self.state.playlist
        ref = self.state.idle_reference
        self.assertEqual(ref.path.name, 'idle_loop_001__A028.pkl')
        self.assertEqual(playlist.mode, 'IDLE')
        self.assertFalse(playlist.events)
        t = playlist.idle_epoch + .4
        q, v, _, _ = playlist.sample(t)
        expected = ref.sample(.4)
        np.testing.assert_allclose(q, expected[0], atol=1e-7)
        np.testing.assert_allclose(v, expected[1], atol=1e-7)
        self.assertGreater(np.max(abs(q[0]-self.c.default)), .05)
        history = self.state.history
        before = playlist.future(t)
        playlist.select(1, t)
        for x, y in zip(before, playlist.future(t)):
            np.testing.assert_array_equal(x, y)
        for cycle in (1, 2, 8):
            for x, y in zip(before, playlist.future(t + cycle*playlist.idle_loop.period)):
                np.testing.assert_allclose(x, y, atol=1e-7)
        self.tick(10)
        self.assertIs(self.state.history, history)
        self.assertEqual(playlist.mode, 'IDLE')

    def test_idle_loop_closes_joint_velocity_and_orientation_seam(self):
        loop = LoopingReference(self.state.idle_reference, 1.)
        # The source has a real endpoint mismatch; a plain modulo is unsuitable.
        self.assertGreater(np.max(abs(loop.last[0]-loop.first[0])), .01)
        epsilon = 1e-6
        for boundary, expected in ((loop.duration, loop.last), (loop.period, loop.first)):
            exact = loop.sample(boundary)
            near = loop.sample(boundary + epsilon if boundary == loop.duration else boundary-epsilon)
            for actual, target in zip(exact[:3], expected[:3]):
                np.testing.assert_allclose(actual, target, atol=1e-7)
            for actual, target in zip(near[:3], expected[:3]):
                np.testing.assert_allclose(actual, target, atol=1e-5)
            np.testing.assert_allclose(rotation(near[3]).as_matrix(), rotation(expected[3]).as_matrix(), atol=1e-5)
        playlist = self.state.playlist
        t = playlist.idle_epoch + loop.period - .4
        future = playlist.future(t)
        for j in range(10):
            q, v, _, _ = loop.sample(t-playlist.idle_epoch+j*.1)
            np.testing.assert_allclose(future[0][j], q, atol=1e-7)
            np.testing.assert_allclose(future[1][j], v, atol=1e-7)

    def test_return_previews_and_resumes_moving_idle_loop(self):
        playlist = self.state.playlist
        playlist.play(playlist.ready_time + .2)
        stop = playlist.ready_time + 2.3
        before = playlist.sample(stop)
        playlist.stop(stop)
        for a, b in zip(before, playlist.sample(stop)):
            np.testing.assert_allclose(a, b, atol=1e-7)
        end = stop + playlist.blend
        anticipated = playlist.sample([end, end+.2, end+.4])
        alignment = playlist.idle_alignment.as_matrix().copy()
        playlist.advance(end)
        self.assertEqual(playlist.mode, 'IDLE')
        for a, b in zip(anticipated, playlist.sample([end, end+.2, end+.4])):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(anticipated[0], self.state.idle_reference.sample([0, .2, .4])[0], atol=1e-7)
        np.testing.assert_array_equal(playlist.idle_alignment.as_matrix(), alignment)

    def test_invalid_feedback_and_inference_fails_before_target_write(self):
        self.ready()
        before = self.rd.q_d_s_.copy()
        self.rd.q_a_s_[7] = np.nan
        with self.assertRaises(RuntimeError):
            self.tick(1)
        np.testing.assert_array_equal(before, self.rd.q_d_s_)
        with self.assertRaises(RuntimeError):
            self.state.policy.infer(np.full(1751, np.nan), np.zeros(930))
        with self.assertRaises(RuntimeError):
            self.c.target(np.full(29, np.inf))

    def test_manifest_rejects_corrupt_model_and_wrong_history(self):
        with tempfile.TemporaryDirectory() as directory:
            d = Path(directory)
            m = copy.deepcopy(self.c.manifest)
            for name in ('model_encoder.onnx', 'model_decoder.onnx', 'robot_mapping.json', 'training_config.yaml'):
                (d/name).symlink_to(self.c.export/name)
            m['hashes']['model_encoder.onnx'] = '0'*64
            (d/'export_manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'Frozen SONIC export changed'):
                Contract(d)
            m = copy.deepcopy(self.c.manifest)
            m['policy_input_layout'][0]['offset'] = 1
            (d/'export_manifest.json').write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError, 'history layout'):
                Contract(d)

    def test_keyboard_and_fsm_entry_play_stop(self):
        keyboard = KeyboardController()
        config = yaml.safe_load((ROOT/'config/dex_config.yaml').read_text())
        fsm = RobotFSMImpl(self.rd, config)
        keyboard._process_key('t')
        fsm.run_fsm(keyboard.get_keyboard_flag())
        self.assertEqual(fsm.current_state, FSMStateName.STOP)
        keyboard._process_key('z')
        fsm.run_fsm(keyboard.get_keyboard_flag())
        keyboard._process_key('t')
        self.rd.control_flag = keyboard.get_keyboard_flag()
        fsm.run_fsm(self.rd.control_flag)
        self.assertEqual(fsm.current_state, FSMStateName.SONICTRACK)
        state = fsm.state_objects[FSMStateName.SONICTRACK]
        for _ in range(210): fsm.run_fsm(keyboard.get_keyboard_flag())
        keyboard._process_key('p')
        self.assertEqual(keyboard.get_keyboard_flag().fsm_state_command, 'gotoSONICTRACK')
        fsm.run_fsm(keyboard.get_keyboard_flag())
        self.assertEqual(state.playlist.mode, 'PLAYING')
        history = state.history
        keyboard._process_key('t')
        fsm.run_fsm(keyboard.get_keyboard_flag())
        self.assertIs(state.history, history)
        keyboard._process_key('c')
        fsm.run_fsm(keyboard.get_keyboard_flag())
        self.assertEqual(fsm.current_state, FSMStateName.STOP)
        self.assertEqual(state.phase, 'INACTIVE')

    def test_publishes_29_correct_motor_ids_positions_and_gains(self):
        self.ready()
        interface = RobotInterfaceImpl(self.rd, str(ROOT/'config/dex_config.yaml'))
        interface.node = SimpleNamespace(
            get_logger=lambda: SimpleNamespace(debug=lambda *args: None),
            get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=Time)))
        interface.load_config()
        interface.sim = True
        interface.initialized = True
        captured = []
        for name in ('pub_leg_motor_cmd', 'pub_waist_motor_cmd', 'pub_arm_motor_cmd'):
            setattr(interface, name, SimpleNamespace(publish=captured.append))
        interface.update_param(FSMStateName.SONICTRACK)
        before = self.rd.q_d_s_.copy()
        interface.send_motor_commands(self.rd.control_flag)
        self.assertEqual([len(msg.cmds) for msg in captured], [12, 3, 14])
        commands = [cmd for msg in captured for cmd in msg.cmds]
        expected_ids = [51,52,53,54,55,56,61,62,63,64,65,66,33,32,31,
                        11,12,13,14,15,16,17,21,22,23,24,25,26,27]
        self.assertEqual([x.name for x in commands], expected_ids)
        np.testing.assert_allclose([x.pos for x in commands], before[-29:], atol=1e-7)
        np.testing.assert_allclose([x.kp for x in commands], self.c.kp[self.c.policy_to_serial], atol=1e-5)
        np.testing.assert_allclose([x.kd for x in commands], self.c.kd[self.c.policy_to_serial], atol=1e-5)
        interface.send_motor_commands(self.rd.control_flag)
        np.testing.assert_array_equal(self.rd.q_d_s_, before)

    def test_reference_resampling_future_and_natural_return(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'linear.pkl'
            t = np.arange(121)/60
            q = self.c.default[self.c.to_mujoco] + t[:,None]*.01
            joblib.dump({path.stem: dict(fps=60, dof=q,
                root_rot=Rotation.from_euler('z', .3+.1*t).as_quat(),
                root_trans_offset=np.column_stack([t, t*0, t*0+.95]))}, path)
            ref = Reference(path, self.c, 0., 0., 0.)
            np.testing.assert_allclose(ref.q[:,0], self.c.default[0]+ref.times*.01, atol=1e-7)
            np.testing.assert_allclose(ref.vel, .01, atol=1e-7)
            playlist = ReferencePlaylist([ref], 1., 1., heading=.7)
            playlist.play(1.)
            np.testing.assert_allclose(rotation(playlist.sample(2)[3]).as_euler('xyz')[0,2], .7, atol=1e-7)
            end = playlist.return_start
            before = playlist.sample(end)
            playlist.advance(end)
            self.assertEqual(playlist.mode, 'RETURNING')
            for a,b in zip(before,playlist.sample(end)):np.testing.assert_allclose(a,b)
            playlist.advance(end+1.)
            self.assertEqual(playlist.mode, 'IDLE')
            np.testing.assert_allclose(playlist.sample(end+1.)[0][0], ref.q[0])
            np.testing.assert_allclose(playlist.sample(end+1.2)[0], ref.sample(.2)[0])


if __name__ == '__main__':
    unittest.main()
