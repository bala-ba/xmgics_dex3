"""Append a braking/default-standing reference to an exported InstantTrack CSV.

The original CSV rows are preserved. Re-export the resulting CSV with
export_instant_track.py to rebuild velocities, future goals and golden vectors.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def quintic(start, velocity, end, seconds, duration):
    """Zero endpoint acceleration; match initial velocity and stop at the end."""
    u = np.clip(np.asarray(seconds)/duration, 0., 1.)[:, None]
    delta = end-start
    v = velocity*duration
    return start + v*u + (10*delta-6*v)*u**3 + (-15*delta+8*v)*u**4 + (6*delta-3*v)*u**5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-csv', type=Path, required=True)
    parser.add_argument('--source-bundle', type=Path, required=True)
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--transition-seconds', type=float, default=1.)
    parser.add_argument('--hold-seconds', type=float, default=2.)
    parser.add_argument('--orientation-seconds', type=float, default=.6)
    parser.add_argument('--standing-height', type=float, required=True,
                        help='Pelvis height for the default pose, computed from robot FK')
    args = parser.parse_args()
    durations = (args.transition_seconds, args.hold_seconds, args.orientation_seconds, args.standing_height)
    if not all(np.isfinite(x) and x > 0 for x in durations) or args.orientation_seconds > args.transition_seconds:
        parser.error('Require finite positive durations/height and orientation time <= transition time')
    if args.output_csv.resolve() == args.source_csv.resolve():
        parser.error('Output must not overwrite the source CSV')
    manifest = json.loads((args.source_bundle/'manifest.json').read_text())
    lines = args.source_csv.read_text().splitlines()
    metadata = dict(line[2:].split(': ', 1) for line in lines if line.startswith('# ') and ': ' in line)
    fps = float(metadata['sample_rate'])
    transition_frames = round(args.transition_seconds*fps)
    hold_frames = round(args.hold_seconds*fps)
    if not all(np.isclose(n/fps, t) for n, t in ((transition_frames, args.transition_seconds),
                                               (hold_frames, args.hold_seconds))):
        parser.error('Durations must be whole source frames')
    table = list(csv.reader(line for line in lines if not line.startswith('#')))
    header, rows = table[0], table[1:]
    source = np.asarray(rows, dtype=float)
    if len(rows) < 3 or not np.isfinite(source).all():
        raise ValueError('Invalid source CSV')
    joint_columns = [header.index('dof_'+n) for n in manifest['joint_names']]
    position_columns = [header.index('root_'+c) for c in ('x', 'y', 'z')]
    quaternion_columns = [header.index('root_q'+c) for c in ('x', 'y', 'z', 'w')]
    with np.load(args.source_bundle/manifest['motions'][0]['file'], allow_pickle=False) as motion:
        q0, dq0 = motion['q'][-1].astype(float), motion['dq'][-1].astype(float)
        p0, v0 = motion['root_pos'][-1].astype(float), motion['root_vel'][-1].astype(float)
        R0 = Rotation.from_quat(motion['root_quat'][-1][[1, 2, 3, 0]])
        omega0 = motion['root_omega'][-1].astype(float)
        np.testing.assert_allclose(source[-1, joint_columns], q0, atol=1e-6)
        np.testing.assert_allclose(source[-1, position_columns], p0, atol=1e-6)
    seconds = np.arange(1, transition_frames+hold_frames+1)/fps
    default = np.asarray(manifest['default_joint_pos'])
    q = quintic(q0, dq0, default, seconds, args.transition_seconds)
    end_position = p0 + v0*args.transition_seconds*.5
    end_position[2] = args.standing_height
    p = quintic(p0, v0, end_position, seconds, args.transition_seconds)
    yaw = np.arctan2(R0.as_matrix()[1, 0], R0.as_matrix()[0, 0])
    end_rotation = Rotation.from_euler('z', yaw)
    displacement = (R0.inv()*end_rotation).as_rotvec()
    rotvec = quintic(np.zeros(3), R0.inv().apply(omega0), displacement,
                    seconds, args.orientation_seconds)
    quaternions = (R0*Rotation.from_rotvec(rotvec)).as_quat()
    # Store a bit-for-bit constant endpoint, avoiding polynomial roundoff in the hold.
    q[transition_frames-1:] = default
    p[transition_frames-1:] = end_position
    quaternions[seconds >= args.orientation_seconds] = end_rotation.as_quat()
    limits = np.asarray(manifest['joint_limits'])
    if np.any(q < limits[:, 0]-1e-6) or np.any(q > limits[:, 1]+1e-6):
        raise ValueError('Generated tail exceeds joint limits; adjust the transition duration')
    tail = np.repeat(source[-1][None], len(seconds), axis=0)
    tail[:, header.index('time')] = source[-1, header.index('time')]+seconds
    tail[:, joint_columns], tail[:, position_columns], tail[:, quaternion_columns] = q, p, quaternions
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open('w', newline='') as stream:
        stream.write('\n'.join(line for line in lines if line.startswith('#'))+'\n')
        stream.write(f'# stand_tail_transition_seconds: {args.transition_seconds}\n'
                     f'# stand_tail_hold_seconds: {args.hold_seconds}\n'
                     '# stand_tail_pose: manifest.default_joint_pos; upright, final source yaw\n')
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)
        writer.writerows([[f'{x:.9f}' for x in row] for row in tail])
    report = dict(source_csv=str(args.source_csv.resolve()),
                  source_csv_sha256=hashlib.sha256(args.source_csv.read_bytes()).hexdigest(),
                  source_bundle=str(args.source_bundle.resolve()),
                  source_motion_sha256=hashlib.sha256(
                      (args.source_bundle/manifest['motions'][0]['file']).read_bytes()).hexdigest(),
                  source_manifest_sha256=hashlib.sha256(
                      (args.source_bundle/'manifest.json').read_bytes()).hexdigest(),
                  source_frames=len(rows), fps=fps,
                  preserved_source_poses=True, transition_seconds=args.transition_seconds,
                  hold_seconds=args.hold_seconds, orientation_seconds=args.orientation_seconds,
                  default_joint_pos=default.tolist(), standing_root_height=args.standing_height,
                  final_yaw_radians=float(yaw), root_braking_displacement=(end_position-p0).tolist(),
                  output_csv=str(args.output_csv.resolve()), frames=len(rows)+len(tail),
                  duration_seconds=(len(rows)+len(tail)-1)/fps,
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output_csv.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
