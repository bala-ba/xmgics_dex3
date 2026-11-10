"""STAND entry, live zero-command inference, routing and failure regression tests."""
import contextlib
import io
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import yaml

from common.joystick import ControlFlag
from common.robot_data import RobotData
from common.robot_interface import RobotInterfaceImpl
from common.stdin_keyboard_control import KeyboardController
from common.websocket_control import WebsocketController
from FSM.fsm_base import FSMStateName
from FSM.robot_fsm import RobotFSMImpl
from .fsm_stand import FSMStateSTAND

ROOT = Path(__file__).resolve().parents[2]


class StandTests(unittest.TestCase):
    def setUp(self):
        output = contextlib.redirect_stdout(io.StringIO())
        output.__enter__()
        self.addCleanup(output.__exit__, None, None, None)
        self.rd = RobotData()
        self.rd.control_dt_ = .01

    def test_keyboard_entry_reselection_and_exit(self):
        cfg = yaml.safe_load((ROOT/'config/dex_config.yaml').read_text())
        self.assertIn('STAND', cfg['enabled_policy_states'])
        cfg['enabled_policy_states'] = ['STAND']
        cfg['sonic_track']['enabled'] = cfg['instant_track']['enabled'] = False
        fsm = RobotFSMImpl(self.rd, cfg)
        keyboard = KeyboardController()

        def press(key):
            keyboard._process_key(key)
            self.rd.control_flag = keyboard.get_keyboard_flag()
            fsm.run_fsm(self.rd.control_flag)

        press('z')
        self.assertEqual(fsm.current_state, FSMStateName.ZERO)
        keyboard.keyboard_flag.x_speed_command = .4
        keyboard.keyboard_flag.y_speed_command = .3
        keyboard.keyboard_flag.yaw_speed_command = .2
        press('1')
        self.assertEqual(fsm.current_state, FSMStateName.STAND)
        np.testing.assert_array_equal([self.rd.control_flag.x_speed_command,
                                       self.rd.control_flag.y_speed_command,
                                       self.rd.control_flag.yaw_speed_command], [0, 0, 0])
        state = fsm.state_objects[FSMStateName.STAND]
        self.assertEqual(state.inference_count, 1)
        press('1')
        self.assertEqual(state.inference_count, 2)  # Reselection must not reset policy history.
        press('z')
        self.assertEqual(fsm.current_state, FSMStateName.ZERO)
        press('1')
        press('c')
        self.assertEqual(fsm.current_state, FSMStateName.STOP)

    def test_zero_command_keeps_inference_and_feedback_active(self):
        state = FSMStateSTAND(self.rd)
        self.rd.q_a_s_[-29:] = state.hold_pos_
        state.on_enter()
        for _ in range(100):
            state.run(self.rd.control_flag)
        before = self.rd.q_d_s_[-29:].copy()
        self.assertEqual(state.inference_count, 100)
        self.assertFalse(state.gait_started_)
        self.rd.imu_data_[2] = .15
        for _ in range(15):
            state.run(self.rd.control_flag)
        self.assertFalse(np.allclose(before[:15], self.rd.q_d_s_[-29:][:15]))
        np.testing.assert_allclose(self.rd.q_d_s_[-14:], state.hold_pos_[-14:])
        self.assertEqual(state.inference_count, 115)

    def test_commands_reach_stand_and_stop_with_disabled_input(self):
        interface = RobotInterfaceImpl(self.rd)
        interface.update_param(FSMStateName.STAND)
        flag = ControlFlag(x_speed_command=.1, y_speed_command=.02, yaw_speed_command=.03)
        interface.update_robot_cmd(flag)
        np.testing.assert_allclose(self.rd.walk_cmd_, [.1, .02, .03])
        flag.enable = False
        interface.update_robot_cmd(flag)
        np.testing.assert_array_equal(self.rd.walk_cmd_, [0, 0, 0])
        flag.enable = True
        interface.update_param(FSMStateName.ZERO)
        interface.update_robot_cmd(flag)
        np.testing.assert_array_equal(self.rd.walk_cmd_, [0, 0, 0])

    def test_bad_policy_output_is_reported_to_control_node(self):
        state = FSMStateSTAND(self.rd)
        state.on_enter()
        bad = Mock()
        bad.run.return_value = [np.full((1, 15), np.nan, np.float32)]
        with patch.object(state, 'ort_session_', bad), self.assertRaises(RuntimeError):
            state.run(self.rd.control_flag)
        self.assertEqual(state.inference_count, 0)
        self.rd.q_a_s_[-1] = np.nan
        # The arm is outside the learned observation; inject an observed leg joint.
        self.rd.q_a_s_[6] = np.nan
        with self.assertRaises(FloatingPointError):
            state.run(self.rd.control_flag)

    def test_websocket_standing_button_selects_stand(self):
        control = WebsocketController(self.rd, None)
        control.flag.x_speed_command = .4
        control._apply_event('enable_standing_policy', 1)
        flag = control.get_ws_flag()
        self.assertEqual(flag.fsm_state_command, 'gotoSTAND')
        self.assertEqual(flag.x_speed_command, 0)
        event = flag.fsm_event_id
        control._apply_event('enable_standing_policy', 2)
        self.assertEqual(control.get_ws_flag().fsm_event_id, event)
        self.assertIn({'id': 5, 'name': 'fsm_stand'}, json.loads(control._states_json)['states'])


if __name__ == '__main__':
    unittest.main()
