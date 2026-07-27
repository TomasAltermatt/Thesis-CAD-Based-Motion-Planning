#include "Actuator/ActuatorMotor.h"
#include "Joint/Joint.h"

namespace redmax {

ActuatorMotor::ActuatorMotor(std::string name, Joint* joint, ControlMode control_mode, VectorX ctrl_min, VectorX ctrl_max, VectorX ctrl_P, VectorX ctrl_D, VectorX ctrl_I, dtype h)
    : Actuator(name, joint->_ndof, ctrl_min, ctrl_max) {
    
    _joint = joint;
    _control_mode = control_mode;
    _ctrl_P = ctrl_P;
    _ctrl_D = ctrl_D;
    _ctrl_I = ctrl_I;
    _h = h;
    _pos_error = VectorX::Zero(_ndof);
    _vel_error = VectorX::Zero(_ndof);
    _integral_error = VectorX::Zero(_ndof);
}

ActuatorMotor::ActuatorMotor(std::string name, Joint* joint, ControlMode control_mode, dtype ctrl_min, dtype ctrl_max, dtype ctrl_P, dtype ctrl_D, dtype ctrl_I, dtype h)
    : Actuator(name, joint->_ndof, ctrl_min, ctrl_max) {
    
    _joint = joint;
    _control_mode = control_mode;
    _ctrl_P = VectorX::Constant(_ndof, ctrl_P);
    _ctrl_D = VectorX::Constant(_ndof, ctrl_D);
    _ctrl_I = VectorX::Constant(_ndof, ctrl_I);
    _h = h;
    _pos_error = VectorX::Zero(_ndof);
    _vel_error = VectorX::Zero(_ndof);
    _integral_error = VectorX::Zero(_ndof);
}

void ActuatorMotor::update_dofs(const VectorX& dofs, const VectorX& dofs_vel) {
    for (int i = 0;i < _ndof;i++) {
        _dofs[i] = dofs[_joint->_index[i]];
        _dofs_vel[i] = dofs_vel[_joint->_index[i]];
    }
}

void ActuatorMotor::computeForce(VectorX& fm, VectorX& fr) {
    // _debug_count += 1;
    if (_control_mode == ControlMode::FORCE) {
        VectorX u = _u.cwiseMin(VectorX::Ones(_joint->_ndof)).cwiseMax(VectorX::Ones(_joint->_ndof) * -1.);
        fr.segment(_joint->_index[0], _joint->_ndof) += map_value(u, VectorX::Constant(_joint->_ndof, -1), VectorX::Ones(_joint->_ndof, 1), _ctrl_min, _ctrl_max);
        // fr.segment(_joint->_index[0], _joint->_ndof) += ((u + VectorX::Ones(_joint->_ndof)) / 2.).cwiseProduct(_ctrl_max - _ctrl_min) + _ctrl_min;
    } else {
        _pos_error.head(3) = _u.head(3) - _dofs.head(3);
        _vel_error.head(3) = - _dofs_vel.head(3);
        if (_ndof == 6) {
            _pos_error.tail(3) = math::mat2rotvec(math::euler2mat(_u.tail(3)) * math::euler2mat(_dofs.tail(3)).transpose());
            _vel_error.tail(3) = -_dofs_vel.tail(3);
        }
        // if (_debug_count % 100 == 0) {
        //     std::cerr << "[DEBUG PD]" << std::endl;
        //     std::cerr << std::setprecision(4) << _pos_error.transpose() << std::endl;
        //     std::cerr << std::setprecision(4) << _vel_error.transpose() << std::endl;
        // }
        _integral_error += _pos_error * _h; // NOTE: to be checked
        fr.segment(_joint->_index[0], _joint->_ndof) += (_ctrl_P.cwiseProduct(_pos_error) + _ctrl_D.cwiseProduct(_vel_error) + _ctrl_I.cwiseProduct(_integral_error)).cwiseMin(_ctrl_max).cwiseMax(_ctrl_min);
    }
}

void ActuatorMotor::computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr) {
    // VectorX u = _u.cwiseMin(VectorX::Ones(_joint->_ndof)).cwiseMax(VectorX::Ones(_joint->_ndof) * -1.);
    // fr.segment(_joint->_index[0], _joint->_ndof) += ((u + VectorX::Ones(_joint->_ndof)) / 2.).cwiseProduct(_ctrl_max - _ctrl_min) + _ctrl_min;
    computeForce(fm, fr);
}

void ActuatorMotor::compute_dfdu(MatrixX& dfm_du, MatrixX& dfr_du) {
    if (_control_mode == ControlMode::FORCE) {
        for (int i = 0;i < _joint->_ndof;i++)
            if (_u[i] >= -1. && _u[i] <= 1.) {
                dfr_du(_joint->_index[i], _index[i]) += (_ctrl_max[i] - _ctrl_min[i]) / 2.;
            }
    } else { // TODO: to be tested
        for (int i = 0;i < _joint->_ndof;i++) {
            dtype f = _ctrl_P[i] * _pos_error[i] + _ctrl_D[i] * _vel_error[i];
            if (f >= _ctrl_min[i] && f <= _ctrl_max[i])
                dfr_du(_joint->_index[i], _index[i]) += _ctrl_P[i];
        }
    }
}

void ActuatorMotor::set_u(const VectorX& u) {
    // _debug_count = 0;
    for (int i = 0;i < _ndof;i++) {
        _u[i] = u[_index[i]];
    }
    _integral_error = VectorX::Zero(_ndof);
}

bool ActuatorMotor::check_completed(dtype pos_tol, dtype vel_tol, bool verbose) {
    dtype pos_error = _pos_error.norm();
    dtype vel_error = _vel_error.norm();
    if (verbose) {
        std::cout << "[ActuatorMotor::check_completed] pos_error: " << pos_error << ", vel_error: " << vel_error << std::endl;
    }
    return pos_error < pos_tol && vel_error < vel_tol;
}

bool ActuatorMotor::check_completed(Vector2 pos_tol, Vector2 vel_tol, bool verbose) {
    dtype pos_t_error = _pos_error.head(3).norm();
    dtype vel_t_error = _vel_error.head(3).norm();
    dtype pos_r_error = Eigen::AngleAxisd(math::euler2mat(_pos_error.tail(3))).angle();
    dtype vel_r_error = Eigen::AngleAxisd(math::euler2mat(_vel_error.tail(3))).angle();
    if (verbose) {
        std::cout << "[ActuatorMotor::check_completed] pos_t_error: " << pos_t_error << ", vel_t_error: " << vel_t_error << ", pos_r_error: " << pos_r_error << ", vel_r_error: " << vel_r_error << std::endl;
        std::cout << "[ActuatorMotor::check_completed] pos_tol: " << pos_tol.transpose() << ", vel_tol: " << vel_tol.transpose() << std::endl;
    }
    return pos_t_error < pos_tol[0] && vel_t_error < vel_tol[0] && pos_r_error < pos_tol[1] && vel_r_error < vel_tol[1];
}

}