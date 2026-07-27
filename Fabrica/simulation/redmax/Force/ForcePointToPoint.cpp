#include "Force/ForcePointToPoint.h"
#include "Body/Body.h"
#include "Simulation.h"
#include "Joint/Joint.h"

namespace redmax {

ForcePointToPoint::ForcePointToPoint(Simulation* sim, dtype stiffness, dtype damping, std::string frame) : Force(sim) {
    _stiffness = stiffness;
    _damping = damping;
    _frame = frame;
    _f = VectorX::Zero(12);
}

void ForcePointToPoint::set_stiffness(dtype stiffness) { 
    _stiffness = stiffness;
}

void ForcePointToPoint::set_damping(dtype damping) { 
    _damping = damping;
}

void ForcePointToPoint::addBodies(Body* body0, Body* body1) {
    _body0 = body0;
    _body1 = body1;
}

void ForcePointToPoint::addPoints(Vector3 xl0, Vector3 xl1) {
    if (_frame == "body") {
        _xls0.push_back(xl0);
        _xls1.push_back(xl1);
    } else if (_frame == "joint") {
        Matrix4 E0ij = _body0->_E_ij;
        Matrix4 E1ij = _body1->_E_ij;
        Matrix3 R0 = E0ij.topLeftCorner(3, 3);
        Matrix3 R1 = E1ij.topLeftCorner(3, 3);
        Vector3 p0 = E0ij.topRightCorner(3, 1);
        Vector3 p1 = E1ij.topRightCorner(3, 1);
        _xls0.push_back(R0 * xl0 + p0);
        _xls1.push_back(R1 * xl1 + p1);
    } else {
        throw_error("ForcePointToPoint: unknown frame");
    }
}

void ForcePointToPoint::computeForce(VectorX& fm, VectorX& fr, bool verbose) {
    int n = _xls0.size();
    Matrix4 E0 = _body0->_E_0i;
    Matrix4 E1 = _body1->_E_0i;
    Vector6 phi0 = _body0->_phi;
    Vector6 phi1 = _body1->_phi;
    Matrix3 R0 = E0.topLeftCorner(3, 3);
    Matrix3 R1 = E1.topLeftCorner(3, 3);
    Vector3 p0 = E0.topRightCorner(3, 1);
    Vector3 p1 = E1.topRightCorner(3, 1);
    std::vector<MatrixX> gamma0, gamma1, J0, J1;
    std::vector<Vector3> xw0, xw1, vw0, vw1;
    for (int k = 0;k < n;k++) {
        gamma0.push_back(math::gamma(_xls0[k]));
        gamma1.push_back(math::gamma(_xls1[k]));
        J0.push_back(R0 * gamma0[k]);
        J1.push_back(R1 * gamma1[k]);
        xw0.push_back(R0 * _xls0[k] + p0);
        xw1.push_back(R1 * _xls1[k] + p1);
        vw0.push_back(J0[k] * phi0);
        vw1.push_back(J1[k] * phi1);
    }

    // compute force vector
    VectorX fs = VectorX::Zero(12);
    VectorX fd = VectorX::Zero(12);
    for (int k = 0;k < n;k++) {
        Vector3 dx = xw1[k] - xw0[k];
        Vector3 dx_dot = vw1[k] - vw0[k];
        VectorX fsk(12); fsk << J0[k].transpose() * dx, -J1[k].transpose() * dx;
        VectorX fdk(12); fdk << J0[k].transpose() * dx_dot, -J1[k].transpose() * dx_dot;
        fs += fsk;
        fd += fdk;
    }
    
    // compute f = ks * fs + kd * fd
    VectorX f = _stiffness * fs + _damping * fd;
    _f = f;

    // fill fm
    fm.segment(_body0->_index[0], 6) += f.segment(0, 6);
    fm.segment(_body1->_index[0], 6) += f.segment(6, 6);
}

void ForcePointToPoint::computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr, bool verbose) {
    int n = _xls0.size();
    Matrix4 E0 = _body0->_E_0i;
    Matrix4 E1 = _body1->_E_0i;
    Vector6 phi0 = _body0->_phi;
    Vector6 phi1 = _body1->_phi;
    Matrix3 R0 = E0.topLeftCorner(3, 3);
    Matrix3 R1 = E1.topLeftCorner(3, 3);
    Matrix3 R0T = R0.transpose();
    Matrix3 R1T = R1.transpose();
    Vector3 p0 = E0.topRightCorner(3, 1);
    Vector3 p1 = E1.topRightCorner(3, 1);
    std::vector<MatrixX> gamma0, gamma1, J0, J1;
    std::vector<Vector3> xw0, xw1, vw0, vw1, vl0, vl1;
    for (int k = 0;k < n;k++) {
        gamma0.push_back(math::gamma(_xls0[k]));
        gamma1.push_back(math::gamma(_xls1[k]));
        J0.push_back(R0 * gamma0[k]);
        J1.push_back(R1 * gamma1[k]);
        xw0.push_back(R0 * _xls0[k] + p0);
        xw1.push_back(R1 * _xls1[k] + p1);
        vw0.push_back(J0[k] * phi0);
        vw1.push_back(J1[k] * phi1);
        vl0.push_back(gamma0[k] * phi0);
        vl1.push_back(gamma1[k] * phi1);
    }

    // compute force vector
    VectorX fs = VectorX::Zero(12);
    VectorX fd = VectorX::Zero(12);
    for (int k = 0;k < n;k++) {
        Vector3 dx = xw1[k] - xw0[k];
        Vector3 dx_dot = vw1[k] - vw0[k];
        VectorX fsk(12); fsk << J0[k].transpose() * dx, -J1[k].transpose() * dx;
        VectorX fdk(12); fdk << J0[k].transpose() * dx_dot, -J1[k].transpose() * dx_dot;
        fs += fsk;
        fd += fdk;
    }
    
    // compute f = ks * fs + kd * fd
    VectorX f = _stiffness * fs + _damping * fd;
    _f = f;

    // fill fm
    fm.segment(_body0->_index[0], 6) += f.segment(0, 6);
    fm.segment(_body1->_index[0], 6) += f.segment(6, 6);

    // compute K and D
    // K = ks * Ks + kd * Kd
    // D = kd * Dd
    Matrix3 I = Matrix3::Identity();
    MatrixX Zero63 = MatrixX::Zero(6, 3);
    MatrixX Ks = MatrixX::Zero(12, 12);
    MatrixX Kd = MatrixX::Zero(12, 12);
    MatrixX Dd = MatrixX::Zero(12, 12);

    Matrix3 R1R0 = R1.transpose() * R0;
    Matrix3 R0R1 = R1R0.transpose();

    for (int k = 0;k < n;k++) {
        Vector3 dx = xw1[k] - xw0[k];
        Vector3 dx_dot = vw1[k] - vw0[k];
        Matrix3 x0_skew = math::skew(_xls0[k]);
        Matrix3 x1_skew = math::skew(_xls1[k]);
        Matrix3 x0_dot_skew = math::skew(vl0[k]);
        Matrix3 x1_dot_skew = math::skew(vl1[k]);
        MatrixX gamma0T = gamma0[k].transpose();
        MatrixX gamma1T = gamma1[k].transpose();

        // Ks
        MatrixX Ksk(12, 12);
        Ksk.block(0, 0, 6, 3) = gamma0T * math::skew(R0T * (xw1[k] - p0));
        Ksk.block(0, 3, 6, 3) = -gamma0T;
        Ksk.block(0, 6, 6, 3) = -gamma0T * R0R1 * x1_skew;
        Ksk.block(0, 9, 6, 3) = gamma0T * R0R1;
        Ksk.block(6, 0, 6, 3) = -gamma1T * R1R0 * x0_skew;
        Ksk.block(6, 3, 6, 3) = gamma1T * R1R0;
        Ksk.block(6, 6, 6, 3) = gamma1T * math::skew(R1T * (xw0[k] - p1));
        Ksk.block(6, 9, 6, 3) = -gamma1T;
        Ks += Ksk;

        // Kd
        MatrixX Kdk(12, 12);
        Kdk.block(0, 0, 6, 3) = gamma0T * math::skew(R0T * vw1[k]);
        Kdk.block(0, 3, 6, 3) = Zero63;
        Kdk.block(0, 6, 6, 3) = -gamma0T * R0R1 * x1_dot_skew;
        Kdk.block(0, 9, 6, 3) = Zero63;
        Kdk.block(6, 0, 6, 3) = -gamma1T * R1R0 * x0_dot_skew;
        Kdk.block(6, 3, 6, 3) = Zero63;
        Kdk.block(6, 6, 6, 3) = gamma1T * math::skew(R1T * vw0[k]);
        Kdk.block(6, 9, 6, 3) = Zero63;
        Kd += Kdk;

        // D
        MatrixX Ddk(12, 12);
        Ddk.block(0, 0, 6, 6) = -gamma0T * gamma0[k];
        Ddk.block(0, 6, 6, 6) = gamma0T * R0R1 * gamma1[k];
        Ddk.block(6, 0, 6, 6) = gamma1T * R1R0 * gamma0[k];
        Ddk.block(6, 6, 6, 6) = -gamma1T * gamma1[k];
        Dd += Ddk;
    }

    MatrixX K = _stiffness * Ks + _damping * Kd;
    MatrixX D = _damping * Dd;

    // copy to global
    Km.block(_body0->_index[0], _body0->_index[0], 6, 6).noalias() += K.block(0, 0, 6, 6);
    Km.block(_body0->_index[0], _body1->_index[0], 6, 6).noalias() += K.block(0, 6, 6, 6);
    Km.block(_body1->_index[0], _body0->_index[0], 6, 6).noalias() += K.block(6, 0, 6, 6);
    Km.block(_body1->_index[0], _body1->_index[0], 6, 6).noalias() += K.block(6, 6, 6, 6);

    Dm.block(_body0->_index[0], _body0->_index[0], 6, 6).noalias() += D.block(0, 0, 6, 6);
    Dm.block(_body0->_index[0], _body1->_index[0], 6, 6).noalias() += D.block(0, 6, 6, 6);
    Dm.block(_body1->_index[0], _body0->_index[0], 6, 6).noalias() += D.block(6, 0, 6, 6);
    Dm.block(_body1->_index[0], _body1->_index[0], 6, 6).noalias() += D.block(6, 6, 6, 6);
}

}