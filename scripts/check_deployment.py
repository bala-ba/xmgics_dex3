#!/usr/bin/env python3
"""Check packaged files and run CPU inference without connecting to a robot.

--files-only uses only Python's standard library (useful before installation).
--ros also loads all enabled FSM states and imports ROS/native interfaces;
it never creates a ROS node, starts control threads, or publishes commands.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def packaged_path(value):
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'External deployment dependency: {value}')
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def check_files():
    inventory = json.loads((ROOT / 'deployment_files.json').read_text())
    errors = []
    for name, expected in inventory['files'].items():
        try:
            path = packaged_path(name)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if path.stat().st_size != expected['bytes'] or digest != expected['sha256']:
                raise ValueError(f'File changed or incomplete: {name}')
        except (OSError, ValueError) as exc:
            errors.append(str(exc))
    if errors:
        raise ValueError('\n'.join(errors))
    print(f"PASS: {len(inventory['files'])} packaged files, sizes and SHA-256")


def check_models():
    import numpy as np
    import onnxruntime as ort
    import yaml
    from policy.sonic_track.sonic_runtime import Contract, SonicPolicy, Reference
    from policy.instant_track.runtime import Runtime, observation

    config = yaml.safe_load((ROOT/'policy/sonic_track/config/sonic_track.yaml').read_text())
    contract = Contract(packaged_path(config['export_dir']+'/export_manifest.json').parent)
    sonic = SonicPolicy(contract)
    with np.load(contract.export/'validation_vectors.npz', allow_pickle=False) as vectors:
        for enc, dec, token_ref, action_ref in zip(
                vectors['encoder_input'], vectors['decoder_input'],
                vectors['token_reference'], vectors['action_reference']):
            action, token, _ = sonic.infer(enc, dec[64:])
            np.testing.assert_allclose(token, token_ref, atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(action, action_ref, atol=1e-4, rtol=1e-5)
    for name in [config['idle_motion_file'], *config['motion_files']]:
        reference = Reference(packaged_path(name), contract, ground_height=0.)
        print(f'PASS: SONIC motion {reference.path.name}, {reference.times[-1]:.2f} s')
    print('PASS: SONIC encoder + decoder, CPU golden vectors')
    del sonic

    instant_config = yaml.safe_load((ROOT/'policy/instant_track/config/instant_track.yaml').read_text())
    selected = packaged_path(instant_config['export_dir']+'/manifest.json')
    manifests = sorted((ROOT/'policy/instant_track/model').glob('*/manifest.json'))
    if selected not in manifests:
        manifests.append(selected)
    for manifest in manifests:
        runtime = Runtime(manifest.parent)
        with np.load(runtime.export/'validation.npz', allow_pickle=False) as vectors:
            for i, frame in enumerate(vectors['frame']):
                motion = runtime.motions[int(vectors['motion'][i])]
                obs = observation(vectors['q'][i], vectors['dq'][i], vectors['omega'][i],
                                  vectors['quat'][i], vectors['previous'][i], runtime.default,
                                  motion['root_R'][frame])
                np.testing.assert_allclose(obs, vectors['obs'][i], rtol=1e-5, atol=1e-6)
                action, _ = runtime.infer(vectors['history'][i], motion['goal'][frame])
                np.testing.assert_allclose(action, vectors['action'][i], rtol=2e-5, atol=2e-5)
        print(f'PASS: InstantTrack {manifest.parent.name}, hashes + motion + CPU golden vectors')
        del runtime

    main = yaml.safe_load((ROOT/'config/dex_config.yaml').read_text())
    states = {'walk_amp': 'WALKAMP', 'walk_depth': 'WALKDEPTH', 'walk_depth_100hz': 'WALKDEPTH100HZ',
              'stand': 'STAND', 'dex_amp': 'DEXAMP', 'dex_delta': 'DEXDELTA', 'multi_amp': 'MULTIAMP'}
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    for path in sorted((ROOT/'policy').glob('*/config/*.yaml')):
        config = yaml.safe_load(path.read_text())
        name = config.get('model_path', config.get('onnx_path'))
        if name is None:
            continue
        folder = path.parent.parent
        model = folder/'model'/name
        if not model.is_file():
            if states.get(folder.name) in main['enabled_policy_states']:
                raise FileNotFoundError(f'Enabled policy model missing: {model}')
            print(f'OPTIONAL, unavailable/disabled: {model.relative_to(ROOT)}')
            continue
        model = packaged_path(str(model))
        session = ort.InferenceSession(str(model), options, providers=['CPUExecutionProvider'])
        print(f'PASS: ONNX load {folder.name}/{name}')
        del session


def check_assets():
    import mujoco
    for name in ('tiangong3_29dof.xml', 'tiangong3_29dof_dex_armature.xml'):
        path = ROOT/'assets/tiangong3'/name
        model = mujoco.MjModel.from_xml_path(str(path))
        if (model.nq, model.nv, model.nu) != (36, 35, 29):
            raise ValueError(f'Wrong Tiangong3 dimensions: {name}')
        print(f'PASS: MuJoCo {name}, 29 actuators and meshes')


def check_ros():
    import yaml
    from common.robot_data import RobotData
    from common.robot_interface import RobotInterfaceImpl  # ROS + native ankle ABI
    from FSM.robot_fsm import RobotFSMImpl
    from rl_control_node_sim import XMIGCSControlNode_sim
    config = yaml.safe_load((ROOT/'config/dex_config.yaml').read_text())
    data = RobotData()
    data.control_dt_ = config['dt']
    with contextlib.redirect_stdout(io.StringIO()):
        fsm = RobotFSMImpl(data, config)
    print('PASS: ROS/native imports and enabled FSM: ' + ', '.join(s.name for s in fsm.state_objects))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--files-only', action='store_true')
    modes.add_argument('--ros', action='store_true')
    args = parser.parse_args()
    try:
        check_files()
        if not args.files_only:
            check_models()
            check_assets()
            if args.ros:
                check_ros()
    except Exception as exc:
        print(f'FAIL: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    print('Deployment check passed. No robot connection was made.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
