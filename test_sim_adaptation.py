"""Regression checks for Dex 3.0 simulation startup and failure paths.

No ROS graph or hardware connection is created by these tests.
"""
import contextlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import yaml

from common.joystick import ControlFlag
from common.robot_data import RobotData
from common.robot_interface import RobotInterfaceImpl
from common.stdin_keyboard_control import KeyboardController
from FSM.fsm_base import FSMStateName
from FSM.robot_fsm import RobotFSMImpl
from rl_control_node import XMIGCSControlNode
from rl_control_node_sim import XMIGCSControlNode_sim

ROOT = Path(__file__).resolve().parent


class SimulationAdaptationTests(unittest.TestCase):
    def setUp(self):
        self.output = contextlib.redirect_stdout(io.StringIO())
        self.output.__enter__()
        self.addCleanup(self.output.__exit__, None, None, None)

    def test_config_is_independent_of_cwd_and_username(self):
        config = yaml.safe_load((ROOT/'config/dex_config.yaml').read_text())
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            # A different config at the caller's cwd must not override the package.
            path = Path(directory)/'config'
            path.mkdir()
            (path/'dex_config.yaml').write_text('keyboard:\n  max_forward_speed: 999\n')
            try:
                os.chdir(directory)
                with patch('getpass.getuser', return_value='ubuntu'):
                    node = SimpleNamespace()
                    XMIGCSControlNode._load_config(node)
                keyboard = KeyboardController()
                self.assertEqual(node.sim, config['sim'])
                self.assertEqual(keyboard.max_forward_speed, config['keyboard']['max_forward_speed'])
            finally:
                os.chdir(cwd)

    def test_sim_entry_overrides_hardware_config_before_thread_start(self):
        interface = SimpleNamespace(sim=False, camera_debug_vis=False, init=Mock())
        observed = []

        def load(node):
            node.config = {"sim": False}
            node.sim = False
            node.motor_num, node.dt = 29, .01
            node.config_file = str(ROOT / "config/dex_config.yaml")

        def start(node):
            observed.append((node.sim, node.config["sim"], node.robot_interface.sim))

        with patch("rl_control_node.Node.__init__", return_value=None), \
             patch.object(XMIGCSControlNode, "_load_config", load), \
             patch("rl_control_node.get_robot_interface", return_value=interface), \
             patch("rl_control_node.get_robot_fsm"), \
             patch.object(XMIGCSControlNode, "_init_ros_interfaces"), \
             patch.object(XMIGCSControlNode, "_init_control_system"), \
             patch.object(XMIGCSControlNode, "_start_control_thread", start):
            XMIGCSControlNode_sim()
        self.assertEqual(observed, [(True, True, True)])

    def test_policy_failure_reads_current_feedback_then_publishes_stop(self):
        rd = RobotData()
        rd.control_dt_ = .01
        rd.control_flag = ControlFlag(fsm_state_command="gotoSONICTRACK")
        config = yaml.safe_load((ROOT / "config/dex_config.yaml").read_text())
        config["enabled_policy_states"] = []
        fsm = RobotFSMImpl(rd, config)
        sonic = fsm.state_objects[FSMStateName.SONICTRACK]
        rd.q_a_s_[-29:] = sonic.contract.default[sonic.contract.policy_to_serial]
        fsm.current_state = FSMStateName.SONICTRACK
        sonic.on_enter()
        events, sent = [], []

        def feedback(flag, stamp):
            events.append("feedback")
            rd.q_a_s_[-29:] = .123

        def fail(flag):
            events.append("policy")
            np.testing.assert_array_equal(rd.q_a_s_[-29:], np.full(29, .123))
            raise RuntimeError("injected inference failure")

        def send(flag):
            events.append("send")
            sent.append((fsm.current_state, rd.q_d_s_.copy(), rd.joint_kp_s_.copy()))

        node = SimpleNamespace(
            control_running=True, dt=.01, control_flag=rd.control_flag,
            robot_data=rd, robot_fsm=fsm, get_logger=lambda: Mock(),
            _process_controller_data=lambda: events.append("input"),
            _update_robot_data=feedback, _send_control_commands=send,
            robot_interface=SimpleNamespace(refresh_depth=lambda: None,
                                            update_param=Mock(), finish_control=Mock()),
        )
        node._precise_sleep_until = lambda deadline: setattr(node, "control_running", False)
        with patch("rl_control_node.rclpy.ok", return_value=True), \
             patch.object(sonic, "run", side_effect=fail):
            XMIGCSControlNode._rl_control_loop(node)
        self.assertEqual(events, ["input", "feedback", "policy", "send"])
        self.assertEqual(sent[0][0], FSMStateName.STOP)
        np.testing.assert_array_equal(sent[0][1][-29:], np.full(29, .123))
        np.testing.assert_array_equal(sent[0][2], fsm.state_objects[FSMStateName.STOP].kp_pos_)
        self.assertEqual(rd.control_flag.fsm_state_command, "gotoSTOP")
        self.assertFalse(rd.walk_cmd_.any())

    def interface(self):
        rd = RobotData()
        interface = RobotInterfaceImpl(rd, str(ROOT / "config/dex_config.yaml"))
        interface.node = SimpleNamespace(get_logger=lambda: Mock())
        interface.load_config()
        interface.initialized = True
        return interface, rd

    def test_sim_send_never_calls_hardware_solver(self):
        interface, rd = self.interface()
        interface.sim = True
        interface.fun_s2p = Mock()
        rd.q_d_s_[-29:] = np.linspace(-.1, .1, 29)
        with patch.object(interface, "publish_motor_commands"):
            interface.send_motor_commands(ControlFlag())
        self.assertEqual(interface.fun_s2p.mock_calls, [])
        np.testing.assert_array_equal(rd.q_d_p_, rd.q_d_s_)

    def test_bad_ankle_results_do_not_publish_or_write_converted_targets(self):
        interface, rd = self.interface()
        interface.sim = False
        interface.fun_s2p = Mock()
        rd.q_d_s_[-29:] = np.linspace(-.2, .2, 29)
        cases = [
            (False, np.zeros(4), np.zeros(4), np.zeros(4)),
            (True, np.zeros(3), np.zeros(4), np.zeros(4)),
            (True, np.zeros(4), np.zeros(4), np.full(4, np.nan)),
        ]
        with patch.object(interface, "publish_motor_commands") as publish:
            for result in cases:
                with self.subTest(result=result[0:1]):
                    interface.fun_s2p.get_p_des.return_value = result
                    with self.assertRaisesRegex(RuntimeError, "ankle serial-to-parallel failed"):
                        interface.send_motor_commands(ControlFlag())
                    publish.assert_not_called()
                    np.testing.assert_array_equal(rd.q_d_p_, rd.q_d_s_)

    def test_failed_ankle_feedback_does_not_overwrite_serial_feedback(self):
        interface, rd = self.interface()
        rd.q_a_s_[:] = .12
        before = rd.q_a_s_.copy()
        interface.fun_s2p = Mock()
        interface.fun_s2p.get_s_state.return_value = (
            False, np.zeros(4), np.zeros(4), np.zeros(4))
        with self.assertRaisesRegex(RuntimeError, "ankle parallel-to-serial failed"):
            interface.ankle_parallel_to_serial()
        np.testing.assert_array_equal(rd.q_a_s_, before)


if __name__ == "__main__":
    unittest.main()
