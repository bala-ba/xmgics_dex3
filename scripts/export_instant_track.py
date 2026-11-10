"""Export InstantTrack's EMA policy and training-preprocessed references for xmigcs.

Run with instantTrack/.venv/bin/python. Runtime only needs NumPy/SciPy/ONNX Runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=ROOT.parent / 'instantTrack')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--motion', type=Path, action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--validation-scene', type=Path, help='optional training-scene XML for CPU validation')
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve() / 'src'))
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import instanttrack.networks as networks
    from instanttrack.kinematics import Kinematics
    from instanttrack.mdp import MDP, obs_dim
    from instanttrack.motion import CONTROL_DT, MotionLib
    from instanttrack.networks import Policy, policy_kwargs
    from instanttrack.robot import get_robot
    from instanttrack.rotations import mat_to_quat, quat_to_mat

    torch.set_num_threads(2)
    torch.manual_seed(438000)
    checkpoint = args.checkpoint.resolve()
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['args']
    if cfg.get('robot') != 'tiangong' or (cfg['hist'], cfg['goal_frames']) != (10, 10):
        raise ValueError('Expected a Tiangong checkpoint with 10 history/goal frames')
    robot = get_robot('tiangong')
    if args.validation_scene:
        # mjlab's exported XML retains file-backed meshes outside spec.assets.
        # Resolve them against the source robot asset; no source files are changed.
        import xml.etree.ElementTree as ET
        from mjlab.scene import Scene
        from mjlab.utils.xml import fix_spec_xml
        from instanttrack.env import make_scene_cfg
        scene = Scene(make_scene_cfg(robot, 1), device='cpu')
        xml = ET.fromstring(fix_spec_xml(scene.spec.to_xml(), meshdir=str(Path(robot.mjcf_path).parent)))
        xml.find('compiler').set('meshdir', str(Path(robot.mjcf_path).parent))
        args.validation_scene.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(xml).write(args.validation_scene, encoding='unicode')
    kin = Kinematics(robot, 'cpu')
    networks.AMP_DTYPE = None
    widths = lambda key: tuple(map(int, cfg[key].split(',')))
    pi = Policy(obs_dim(29), 730, 29, hist=10, enc_hidden=widths('enc_hidden'),
                dec_hidden=widths('dec_hidden'), **policy_kwargs(cfg, robot)).eval()
    pi.load_state_dict(ck['pi'], strict=True)
    if not all(torch.isfinite(v).all() for v in pi.state_dict().values()):
        raise ValueError('Checkpoint contains non-finite weights')
    if pi.ref_off is not None:
        np.testing.assert_allclose(pi.ref_q0.detach().numpy(), robot.default_joint_pos_list(), atol=1e-7)
        np.testing.assert_allclose(pi.ref_alpha.detach().numpy(), robot.action_scale_list(), atol=1e-7)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_file = output / 'policy.onnx'
    torch.onnx.export(pi, (torch.zeros(1, 10, 99), torch.zeros(1, 730)), str(model_file),
                      input_names=['obs_history', 'goal'], output_names=['actions'],
                      opset_version=18, dynamo=False)
    onnx.checker.check_model(str(model_file))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(model_file), options, providers=['CPUExecutionProvider'])
    refs, payloads, validation = [], ['policy.onnx'], []
    max_error = 0.
    with torch.no_grad():
        for index, path in enumerate(args.motion):
            path = path.resolve()
            motion = MotionLib([str(path)], kin, 'cpu', robot.load_motion)
            frames = torch.arange(motion.num_frames)
            goals = motion.goal(frames, 10)
            filename = f'motion_{index:03d}.npz'
            np.savez_compressed(output / filename, q=motion.q.numpy(), dq=motion.qd.numpy(),
                                root_pos=motion.root_p.numpy(), root_quat=mat_to_quat(motion.root_R).numpy(),
                                root_vel=motion.root_v.numpy(), root_omega=motion.root_w.numpy(),
                                goal=goals.numpy())
            refs.append(dict(name=path.stem, file=filename, source=str(path), sha256=sha256(path),
                             frames=motion.num_frames, duration=(motion.num_frames - 1)*CONTROL_DT))
            payloads.append(filename)
            mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list()), 33)
            for f in torch.linspace(0, motion.num_frames - 1, 16).long().tolist():
                q = motion.q[f:f+1] + .03*torch.randn(1, 29)
                dq = motion.qd[f:f+1] + .1*torch.randn(1, 29)
                quat = torch.randn(1, 4)
                quat /= quat.norm(dim=-1, keepdim=True)
                R = quat_to_mat(quat)
                omega = torch.randn(1, 3)
                previous = torch.randn(1, 29)
                S = (motion.root_p[f:f+1], R, motion.root_v[f:f+1],
                     (R @ omega.unsqueeze(-1)).squeeze(-1), q, dq)
                obs = mdp.obs(S, previous, torch.zeros(1, 33), torch.tensor([f]))
                hist = pi.obs_norm.mean + pi.obs_norm.std()*torch.randn(1, 10, 99)
                hist[:, -1] = obs
                goal = goals[f:f+1]
                expected = pi(hist, goal).numpy()
                actual = session.run(None, {'obs_history': hist.numpy(), 'goal': goal.numpy()})[0]
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)
                max_error = max(max_error, float(np.max(np.abs(actual-expected))))
                validation.append(dict(motion=index, frame=f, q=q[0].numpy(), dq=dq[0].numpy(),
                                       quat=quat[0].numpy(), omega=omega[0].numpy(), previous=previous[0].numpy(),
                                       obs=obs[0].numpy(), history=hist[0].numpy(), action=expected[0]))
    np.savez_compressed(output / 'validation.npz', **{
        name: np.asarray([row[name] for row in validation]) for name in validation[0]})
    payloads.append('validation.npz')
    acts = robot.robot_cfg().articulation.actuators
    drives = [next(a for a in acts if any(re.fullmatch(pattern, name)
              for pattern in a.target_names_expr)) for name in robot.joint_names]
    sources = ['networks.py', 'mdp.py', 'motion.py', 'rotations.py', 'kinematics.py',
               'robot.py', 'tiangong/tiangong_constants.py']
    manifest = dict(format_version=1, robot='tiangong', iteration=int(ck['iteration']),
                    checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint), weights_key='pi',
                    precision='float32', control_dt=CONTROL_DT, history=10, observation_size=99, goal_size=730,
                    joint_names=list(robot.joint_names), default_joint_pos=robot.default_joint_pos_list(),
                    action_scale=robot.action_scale_list(), kp=[a.stiffness for a in drives],
                    kd=[a.damping for a in drives], torque_limits=[a.effort_limit for a in drives],
                    joint_limits=np.column_stack((kin.q_lo.numpy(), kin.q_hi.numpy())).tolist(),
                    reference_residual=bool(cfg.get('pi_ref_skip')), motions=refs,
                    observation_fields=['joint_pos_minus_default', 'joint_vel', 'body_angular_velocity',
                                        'projected_gravity', 'previous_raw_action', 'reference_relative_rotation_6d'],
                    hashes={name: sha256(output/name) for name in payloads},
                    source_hashes={name: sha256(args.source_root/'src/instanttrack'/name) for name in sources},
                    onnx_torch_max_abs_error=max_error, training_args=cfg)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(dict(export=str(output), checkpoint_iteration=manifest['iteration'],
                          motions=refs, onnx_torch_max_abs_error=max_error), indent=2))


if __name__ == '__main__':
    main()
