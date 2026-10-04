# coding= utf8
import scipy.optimize
import numpy as np
from ikpy import logs
import warnings


ORIENTATION_COEFF = 1.
DEFAULT_OPTIMIZER = 'least_squares'


def inverse_kinematic_optimization(chain, target_frame, starting_nodes_angles, regularization_parameter=None, max_iter=None, orientation_mode=None, no_position=False, optimizer=None):
    """
    Computes the inverse kinematic on the specified target

    Parameters
    ----------
    chain: ikpy.chain.Chain
        The chain used for the Inverse kinematics.
    target_frame: numpy.array
        The desired target.
    starting_nodes_angles: numpy.array
        The initial pose of your chain.
    regularization_parameter: float
        The coefficient of the regularization.
    max_iter: int
        Maximum number of iterations for the optimisation algorithm.
    orientation_mode: str
        Orientation to target. Choices:
        * None: No orientation
        * "X": Target the X axis
        * "Y": Target the Y axis
        * "Z": Target the Z axis
        * "all": Target the three axes
    no_position: bool
        Do not optimize against position
    optimizer: str
        The optimizer to use. Choices:
        * "least_squares": Use scipy.optimize.least_squares
        * "scalar": Use scipy.optimize.minimize
    """
    if optimizer is None: optimizer = DEFAULT_OPTIMIZER

    # Begin with the position
    target = target_frame[:3, -1]

    # Initial function call when optimizing
    def optimize_basis(x):
        # y = np.append(starting_nodes_angles[:chain.first_active_joint], x)
        y = chain.active_to_full(x, starting_nodes_angles)
        fk = chain.forward_kinematics(y)

        return fk

    # Compute error to target
    def optimize_target_function(fk):
        target_error = fk[:3, -1] - target

        # We need to return the fk, it will be used in a later function
        # This way, we don't have to recompute it
        return target_error

    if orientation_mode is None:
        if no_position:
            raise ValueError("Unable to optimize against neither position or orientation")

        else:
            def optimize_function(x):
                fk = optimize_basis(x)
                target_error = optimize_target_function(fk)
                return target_error
    else:
        # Only get the first orientation vector
        if orientation_mode == "X":
            target_orientation = target_frame[:3, 0]

            def get_orientation(fk):
                return fk[:3, 0]

        elif orientation_mode == "Y":
            target_orientation = target_frame[:3, 1]

            def get_orientation(fk):
                return fk[:3, 1]

        elif orientation_mode == "Z":
            target_orientation = target_frame[:3, 2]

            def get_orientation(fk):
                return fk[:3, 2]

        elif orientation_mode == "all":
            target_orientation = target_frame[:3, :3]

            def get_orientation(fk):
                return fk[:3, :3]
        else:
            raise ValueError("Unknown orientation mode: {}".format(orientation_mode))

        if not no_position:
            def optimize_function(x):
                # Note: This function casts x into a np.float64 array, to have good precision in the computation of the gradients
                fk = optimize_basis(x)

                target_error = optimize_target_function(fk)
                orientation_error = (get_orientation(fk) - target_orientation).ravel()

                # Put more pressure on optimizing the distance to target, to avoid being stuck in a local minimum where the orientation is perfectly reached, but the target is nowhere to be reached
                total_error = np.concatenate([target_error, ORIENTATION_COEFF * orientation_error])

                return total_error
        else:
            def optimize_function(x):
                fk = optimize_basis(x)

                orientation_error = (get_orientation(fk) - target_orientation).ravel()
                total_error = orientation_error

                return total_error

    if starting_nodes_angles is None:
        raise ValueError("starting_nodes_angles must be specified")
    
    starting_nodes_angles_active = chain.active_from_full(starting_nodes_angles)

    # Nullspace regularization
    if regularization_parameter is not None:

        def regularization_function(x):
            return np.linalg.norm(x - starting_nodes_angles_active)
        
        def optimize_total(x):
            target_error = optimize_function(x)

            # Compute the regularization term
            reg_term = regularization_function(x)
            
            # Dynamically scale regularization parameter
            target_error_magnitude = np.linalg.norm(target_error)
            reg_term_magnitude = np.linalg.norm(reg_term)
            adaptive_parameter = regularization_parameter * (target_error_magnitude / max(reg_term_magnitude, 1e-6))

            # Combine errors
            total_error = np.concatenate([target_error, [adaptive_parameter * reg_term]])
            return total_error
    
    else:
        optimize_total = optimize_function

    # Compute bounds
    real_bounds = [link.bounds for link in chain.links]
    real_bounds = chain.active_from_full(real_bounds)

    logs.logger.info("Bounds: {}".format(real_bounds))

    if max_iter is not None:
        logs.logger.info("max_iter is not used anymore in the IK, using it as a parameter will raise an exception in the future")

    with np.errstate(divide='ignore', invalid='ignore', over='ignore'): # ignore numerical warnings

        def optimize_scalar(x):
            return np.linalg.norm(optimize_total(x))

        # least squares optimization
        if optimizer == 'COBYLA':
            cons = []
            for factor in range(len(real_bounds)):
                lower, upper = real_bounds[factor]
                l = {'type': 'ineq',
                    'fun': lambda x, lb=lower, i=factor: x[i] - lb}
                u = {'type': 'ineq',
                    'fun': lambda x, ub=upper, i=factor: ub - x[i]}
                cons.append(l)
                cons.append(u)
            res = scipy.optimize.minimize(optimize_scalar, starting_nodes_angles_active, constraints=cons, method='COBYLA')
        elif optimizer == 'basinhopping': # NOTE: powerful but slow
            res = scipy.optimize.basinhopping(optimize_scalar, starting_nodes_angles_active, niter=10, niter_success=3, minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': real_bounds})
            res = res.lowest_optimization_result
        elif optimizer == "least_squares":
            # We need to unzip the bounds
            res = scipy.optimize.least_squares(optimize_total, starting_nodes_angles_active, bounds=np.moveaxis(real_bounds, -1, 0))
        else:
            res = scipy.optimize.minimize(optimize_scalar, starting_nodes_angles_active, bounds=real_bounds, method=optimizer)

    if res.status != -1:
        logs.logger.info("Inverse kinematic optimisation OK, termination status: {}".format(res.status))
    else:
        logs.logger.warning("Inverse kinematic optimisation returned an error: termination status: {}".format(res.status))

    # print('IK result:', res.success, res.status, res.message, optimize_function(res.x), regularization_function(res.x))

    # real_bounds_reshaped = np.moveaxis(real_bounds, -1, 0)
    # res.x = np.clip(res.x, real_bounds_reshaped[0], real_bounds_reshaped[1])

    return chain.active_to_full(res.x, starting_nodes_angles), res.success
