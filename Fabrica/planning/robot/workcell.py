import numpy as np


def get_board_dx():
    return 2.5


def get_move_arm_pos(arm_type):
    dx = get_board_dx()
    if arm_type == 'xarm7':
        return np.array([14.5 * dx, 10 * dx, 0])
    elif arm_type == 'panda':
        return np.array([18 * dx, 8 * dx, 0])
    elif arm_type == 'ur5e':
        return np.array([18 * dx, 10 * dx, 0])
    elif arm_type == 'yumi':
        return np.array([0, 8 * dx, 10])

    raise NotImplementedError


def get_hold_arm_pos(arm_type):
    dx = get_board_dx()
    if arm_type == 'xarm7':
        return np.array([-14.5 * dx, 10 * dx, 0])
    elif arm_type == 'panda':
        return np.array([-18 * dx, 8 * dx, 0])
    elif arm_type == 'ur5e':
        return np.array([-18 * dx, 10 * dx, 0])
    elif arm_type == 'yumi':
        return np.array([0, 8 * dx, 10])
    
    raise NotImplementedError


def get_dual_arm_pos(arm_type):
    return get_move_arm_pos(arm_type), get_hold_arm_pos(arm_type)


def get_single_arm_pos(arm_type):
    return get_move_arm_pos(arm_type)


def get_move_arm_euler():
    return np.array([0, 0, -np.pi / 2])


def get_hold_arm_euler():
    return np.array([0, 0, -np.pi / 2])


def get_dual_arm_euler():
    return get_move_arm_euler(), get_hold_arm_euler()


def get_single_arm_euler():
    return get_move_arm_euler()


def get_move_arm_box(arm_type):
    arm_pos = get_move_arm_pos(arm_type)
    if arm_type == 'yumi':
        # Expanded crossover to 40cm: Right arm (Move) can now reach the left side of the assembly
        return arm_pos - np.array([100.0, 100.0, 10.0]), arm_pos + np.array([100.0, 100.0, 100.0])
    return arm_pos - np.array([100.0, 100.0, 0.0]), arm_pos + np.array([30.0, 50.0, 80.0])

def get_hold_arm_box(arm_type):
    arm_pos = get_hold_arm_pos(arm_type)
    if arm_type == 'yumi':
        # Expanded crossover to 40cm: Left arm (Hold) can now reach the right side of the assembly
        return arm_pos - np.array([100.0, 100.0, 10.0]), arm_pos + np.array([100.0, 100.0, 100.0])
    return arm_pos - np.array([30.0, 100.0, 0.0]), arm_pos + np.array([100.0, 50.0, 80.0])


def get_dual_arm_box(arm_type):
    return get_move_arm_box(arm_type), get_hold_arm_box(arm_type)


def get_single_arm_box(arm_type):
    return get_move_arm_box(arm_type)


def get_assembly_center(arm_type):
    if arm_type == 'yumi':
        return np.array([0, -6 * get_board_dx(), 0])
    return np.array([0, -6 * get_board_dx(), 0])

def get_fixture_min_y(arm_type):
    dx = get_board_dx()
    if arm_type == 'xarm7':
        return 6 * dx
    elif arm_type == 'panda':
        return 4 * dx
    elif arm_type == 'ur5e':
        return 6 * dx
    elif arm_type == 'yumi':
        return -22 * dx 
    raise NotImplementedError

def get_max_bin_size_double(arm_type):
    dx = get_board_dx()
    if arm_type == 'yumi':
        # Halves depth to 25 cm
        return [8 * dx, 10 * dx]
    return [8 * dx, 20 * dx]

def get_max_bin_size_blocking(arm_type):
    dx = get_board_dx()
    if arm_type == 'yumi':
        # Halves depth to 25 cm
        return [12 * dx, 10 * dx]
    return [12 * dx, 20 * dx]