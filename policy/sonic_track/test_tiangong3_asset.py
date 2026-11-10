"""Compare the generated asset to MuJoCo's independent native URDF importer."""
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np

from .sonic_runtime import SERIAL_JOINT_NAMES
from .tiangong3_asset import SOURCE, build


class Tiangong3AssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.model, cls.report = build(SOURCE, Path(cls.temp.name)/"robot.xml")
        cls.urdf = ET.parse(SOURCE/"urdf/tiangong3.urdf").getroot()

    def test_native_urdf_forward_kinematics(self):
        source = SOURCE/"urdf/tiangong3.urdf"
        root = ET.parse(source).getroot()
        for joint in root.findall("joint"):
            if joint.get("name") in SERIAL_JOINT_NAMES:
                joint.set("type", "revolute")
        for mesh in root.findall(".//mesh"):
            mesh.set("filename", str((source.parent/mesh.get("filename")).resolve()))
        options = ET.SubElement(root, "mujoco")
        ET.SubElement(options, "compiler", discardvisual="false", fusestatic="false", strippath="false")
        native = mj.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data, actual = mj.MjData(native), mj.MjData(self.model)
        qa = [native.joint(n).qposadr[0] for n in SERIAL_JOINT_NAMES]
        qb = [self.model.joint(n).qposadr[0] for n in SERIAL_JOINT_NAMES]
        names = [native.body(i).name for i in range(1, native.nbody)]
        ia = [native.body(n).id for n in names]
        ib = [self.model.body(n).id for n in names]
        rng = np.random.default_rng(31)
        for i in range(4):
            q = np.zeros(29) if i == 0 else rng.uniform(-.1, .1, 29)
            data.qpos[qa], actual.qpos[qb] = q, q
            actual.qpos[:3], actual.qpos[3:7] = 0., [1, 0, 0, 0]
            mj.mj_forward(native, data)
            mj.mj_forward(self.model, actual)
            np.testing.assert_allclose(data.xpos[ia], actual.xpos[ib], atol=1e-9)
            np.testing.assert_allclose(data.xmat[ia], actual.xmat[ib], atol=1e-9)

    def test_body_mass_joint_and_actuator_contract(self):
        m = self.model
        self.assertEqual((m.nq, m.nv, m.nu), (36, 35, 29))
        mass = sum(float(x.get("value")) for x in self.urdf.findall("link/inertial/mass"))
        self.assertAlmostEqual(float(m.body_mass.sum()), mass, places=10)
        self.assertTrue(self.report["full_inertias_verified"])
        self.assertEqual(len(self.report["unlocked_joints"]), 6)
        self.assertEqual(mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "head_yaw_joint"), -1)
        self.assertEqual(mj.mj_name2id(m, mj.mjtObj.mjOBJ_JOINT, "head_pitch_joint"), -1)
        joints = {x.get("name"): x for x in self.urdf.findall("joint")}
        for name in SERIAL_JOINT_NAMES:
            j, a = m.joint(name).id, m.actuator("motor_"+name).id
            limit = joints[name].find("limit")
            self.assertEqual(m.actuator_trnid[a, 0], j)
            np.testing.assert_allclose(m.jnt_range[j], [float(limit.get(k)) for k in ("lower", "upper")])
            effort = float(limit.get("effort"))
            np.testing.assert_allclose(m.actuator_ctrlrange[a], [-effort, effort])


if __name__ == "__main__":
    unittest.main()
