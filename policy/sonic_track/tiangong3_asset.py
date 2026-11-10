"""Build a 29-DoF Tiangong3 MJCF from the official URDF and mesh files.

Keep authored mass/full inertia/geometry/limits; use joint damping and friction
from the accompanying torque MJCF. This is a cross-robot transfer asset, not a
replacement for the frozen Tiangong2Dex training mapping.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import mujoco as mj
import numpy as np
from scipy.spatial.transform import Rotation

from .sonic_runtime import SERIAL_JOINT_NAMES

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "assets/tiangong3"
OUTPUT = ROOT / "assets/tiangong3/tiangong3_29dof.xml"


def fmt(values):
    return " ".join(f"{float(x):.16g}" for x in values)


def origin(element):
    if element is None:
        return dict(pos="0 0 0", quat="1 0 0 0")
    angles = np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
    quat = Rotation.from_euler("xyz", angles).as_quat()[[3, 0, 1, 2]]
    return dict(pos=element.get("xyz", "0 0 0"), quat=fmt(quat))


def inertial_tensor(element):
    v = element.find("inertia").attrib
    matrix = np.array([[float(v["ixx"]), float(v["ixy"]), float(v["ixz"])],
                       [float(v["ixy"]), float(v["iyy"]), float(v["iyz"])],
                       [float(v["ixz"]), float(v["iyz"]), float(v["izz"])]])
    pose = origin(element.find("origin"))
    rot = Rotation.from_quat(np.fromstring(pose["quat"], sep=" ")[[1, 2, 3, 0]]).as_matrix()
    return rot @ matrix @ rot.T


def build(source=SOURCE, output=OUTPUT):
    source, output = Path(source).resolve(), Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    urdf_path = source / "urdf/tiangong3.urdf"
    urdf = ET.parse(urdf_path).getroot()
    torque = ET.parse(source / "tiangong3_torq.xml").getroot()
    links = {e.get("name"): e for e in urdf.findall("link")}
    joints = {e.get("name"): e for e in urdf.findall("joint")}
    if not set(SERIAL_JOINT_NAMES).issubset(joints):
        raise ValueError("Tiangong3 is missing required SONIC body joints")
    children = {}
    for joint in joints.values():
        children.setdefault(joint.find("parent").get("link"), []).append(joint)
    dynamics = {}
    for joint in torque.findall("worldbody/.//joint"):
        default = torque.find(f".//default[@class='{joint.get('class')}']/joint")
        dynamics[joint.get("name")] = {key: joint.get(key, default.get(key, "0") if default is not None else "0")
                                      for key in ("damping", "frictionloss", "armature")}

    root = ET.Element("mujoco", model="tiangong3_sonic_29dof")
    ET.SubElement(root, "compiler", angle="radian", fusestatic="false", inertiafromgeom="false")
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast", solver="CG", iterations="100")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="900")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.7 0.7 0.7")
    assets = ET.SubElement(root, "asset")
    ET.SubElement(assets, "texture", name="floor_tex", type="2d", builtin="checker",
                  rgb1="0.15 0.19 0.23", rgb2="0.23 0.28 0.32", width="512", height="512")
    ET.SubElement(assets, "material", name="floor_mat", texture="floor_tex", texrepeat="8 8", reflectance="0.12")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="2 -2 4", dir="-0.4 0.4 -1", diffuse="0.8 0.8 0.8")
    ET.SubElement(world, "geom", name="floor", type="plane", size="20 20 .1", material="floor_mat",
                  friction="1 .01 .01", condim="3", group="0")
    mesh_assets = {}

    def geometry(body, entry, kind, index, link_name):
        geometry = entry.find("geometry")
        shape = list(geometry)[0]
        name = entry.get("name", f"{link_name}_{kind}_{index}")
        attrs = dict(name=name, **origin(entry.find("origin")))
        if shape.tag == "mesh":
            path = (urdf_path.parent / shape.get("filename")).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            key = (path, shape.get("scale", "1 1 1"))
            if key not in mesh_assets:
                asset_name = f"mesh_{len(mesh_assets)}"
                mesh_assets[key] = asset_name
                ET.SubElement(assets, "mesh", name=asset_name, scale=key[1],
                              file=os.path.relpath(path, output.parent))
            attrs.update(type="mesh", mesh=mesh_assets[key])
        elif shape.tag == "cylinder":
            attrs.update(type="cylinder", size=fmt([float(shape.get("radius")), float(shape.get("length"))/2]))
        elif shape.tag == "box":
            attrs.update(type="box", size=fmt(np.fromstring(shape.get("size"), sep=" ")/2))
        elif shape.tag == "sphere":
            attrs.update(type="sphere", size=shape.get("radius"))
        else:
            raise ValueError(f"Unsupported URDF geometry {shape.tag}")
        if kind == "visual":
            color = entry.find("material/color")
            attrs.update(contype="0", conaffinity="0", group="2", density="0",
                         rgba=color.get("rgba") if color is not None else ".75 .75 .75 1")
        else:
            # Same no-self-collision convention as official tiangong3.py/MJCF.
            attrs.update(contype="0", conaffinity="1", group="3", condim="3",
                         friction="1 .01 .01", solref=".005 1", solimp=".99 .999 .00001")
        ET.SubElement(body, "geom", **attrs)

    def body(parent, name, joint=None):
        node = ET.SubElement(parent, "body", name=name,
                             **(origin(joint.find("origin")) if joint is not None else dict(pos="0 0 1")))
        link = links[name]
        if joint is None:
            ET.SubElement(node, "freejoint", name="floating_base")
        elif joint.get("name") in SERIAL_JOINT_NAMES:
            limit = joint.find("limit")
            ET.SubElement(node, "joint", name=joint.get("name"), type="hinge",
                          axis=joint.find("axis").get("xyz"), limited="true",
                          range=f"{limit.get('lower')} {limit.get('upper')}",
                          **dynamics[joint.get("name")])
        elif joint.get("type") != "fixed":
            raise ValueError(f"Unexpected non-policy joint {joint.get('name')}")
        inertia = link.find("inertial")
        if inertia is not None and float(inertia.find("mass").get("value")) > 0:
            matrix = inertial_tensor(inertia)
            ET.SubElement(node, "inertial", mass=inertia.find("mass").get("value"),
                          pos=origin(inertia.find("origin"))["pos"],
                          fullinertia=fmt(matrix[(0, 1, 2, 0, 0, 1), (0, 1, 2, 1, 2, 2)]))
        for kind in ("visual", "collision"):
            for i, entry in enumerate(link.findall(kind)):
                geometry(node, entry, kind, i, name)
        for child in children.get(name, []):
            body(node, child.find("child").get("link"), child)

    body(world, "pelvis")
    actuators = ET.SubElement(root, "actuator")
    for name in SERIAL_JOINT_NAMES:
        effort = float(joints[name].find("limit").get("effort"))
        ET.SubElement(actuators, "motor", name="motor_"+name, joint=name, gear="1",
                      ctrllimited="true", ctrlrange=fmt([-effort, effort]))
    ET.indent(root, space="  ")
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    model = mj.MjModel.from_xml_path(str(output))
    if (model.nq, model.nv, model.nu) != (36, 35, 29):
        raise ValueError("Expected one floating base and 29 torque-controlled joints")
    inertia_error = 0.
    for name, link in links.items():
        element = link.find("inertial")
        if element is None:
            continue
        i = model.body(name).id
        np.testing.assert_allclose(model.body_mass[i], float(element.find("mass").get("value")), atol=1e-12)
        quat = model.body_iquat[i][[1, 2, 3, 0]]
        rot = Rotation.from_quat(quat).as_matrix()
        actual = rot @ np.diag(model.body_inertia[i]) @ rot.T
        inertia_error = max(inertia_error, float(np.max(np.abs(actual-inertial_tensor(element)))))
        # MuJoCo's principal-axis decomposition introduces small roundoff.
        np.testing.assert_allclose(actual, inertial_tensor(element), rtol=0, atol=1e-7)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    provenance = source / "source_provenance.json"
    if provenance.is_file():
        metadata = json.loads(provenance.read_text())
        if metadata["source_urdf_sha256"] != sha(urdf_path):
            raise ValueError("Packaged Tiangong3 URDF changed from its recorded source")
        revision = metadata["source_revision"]
    else:
        revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    report = dict(source_url="https://github.com/Open-X-Humanoid/TienKung-Lab/tree/dev/legged_lab/assets/tiangong3",
                  source_revision=revision, source_urdf=str(urdf_path), source_urdf_sha256=sha(urdf_path),
                  xml=str(output), xml_sha256=sha(output), mass_kg=float(model.body_mass.sum()),
                  dimensions=dict(nq=model.nq, nv=model.nv, nu=model.nu), joint_order=list(SERIAL_JOINT_NAMES),
                  unlocked_joints=[n for n in SERIAL_JOINT_NAMES if joints[n].get("type") == "fixed"],
                  head="yaw and pitch fixed at zero; source head mass retained", full_inertias_verified=True,
                  maximum_inertia_reconstruction_error=inertia_error,
                  joint_dynamics_source="official tiangong3_torq.xml damping/frictionloss; zero armature where unspecified",
                  contact="source URDF collision shapes; self collision disabled as in official tiangong3.py",
                  motor_effort_source="Tiangong3 URDF", meshes=[dict(path=str(p), sha256=sha(p)) for p, _ in mesh_assets])
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    return model, report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    _, report = build(args.source, args.output)
    print(json.dumps({k: v for k, v in report.items() if k != "meshes"}, indent=2))
