import os
import sys

project_base_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.append(project_base_dir)

import pickle
from tqdm import tqdm

from planning.robot.util_arm import get_arm_chain


def adjust_motion_by_eef_offset(input_path, output_path, eef_offset_move, eef_offset_hold):
    '''
    Load motion (every arm q path)
    Get target pos and orientation from q (FK)
    Adjust target pos by eef offset
    IK to get q
    '''
    
    with open(input_path, 'rb') as f:
        motion = pickle.load(f)

    chain_move = get_arm_chain('panda', 'move')
    chain_hold = get_arm_chain('panda', 'hold')
    chains = {'move': chain_move, 'hold': chain_hold}
    eef_offsets = {'move': eef_offset_move, 'hold': eef_offset_hold}

    new_motion = []

    for motion_step in tqdm(motion):
        motion_type, body_type, path, active_part, description = motion_step
        if body_type == 'arm':
            chain = chains[motion_type]
            eef_offset = eef_offsets[motion_type]
            new_path = []
            for q_active in path:
                fk = chain.forward_kinematics_active(q_active)
                target_pos = fk[:3, 3] + eef_offset
                target_ori = fk[:3, :3]
                q_new, ik_success = chain.inverse_kinematics(target_position=target_pos, target_orientation=target_ori, orientation_mode='all', initial_position=chain.active_to_full(q_active), optimizer='least_squares')
                assert ik_success
                new_path.append(chain.active_from_full(q_new))
            motion_step[2] = new_path
        
        new_motion.append(motion_step)
    
    with open(output_path, 'wb') as f:
        pickle.dump(new_motion, f)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--motion-input', type=str, required=True)
    parser.add_argument('--motion-output', type=str, required=True)
    parser.add_argument('--eef-offset-move', type=float, nargs=3, required=True)
    parser.add_argument('--eef-offset-hold', type=float, nargs=3, required=True)
    args = parser.parse_args()

    adjust_motion_by_eef_offset(args.motion_input, args.motion_output, args.eef_offset_move, args.eef_offset_hold)
