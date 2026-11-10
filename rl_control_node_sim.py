import os
import sys

import rclpy
from rclpy.executors import ExternalShutdownException
import yaml

import signal
from rl_control_node import XMIGCSControlNode
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


class XMIGCSControlNode_sim(XMIGCSControlNode):
    def __init__(self, debug=False):
        # MuJoCo exposes serial ankles; set sim before any control tick runs.
        super().__init__(debug, sim=True)

def main(args=None):
    """主函数"""
    rclpy.init(args=args)
    node = None
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        node = XMIGCSControlNode_sim(debug=False)
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except RuntimeError:
        # SIGTERM 关闭 context 后, spin 中未完成的订阅 take 会抛
        # "Unable to convert call argument"; 上下文仍在则属真实错误, 放行
        if rclpy.ok():
            raise
    finally:
        try:
            node.destroy_node()
        except Exception as e:
            pass
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == "__main__":
    main()
