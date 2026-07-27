#pragma once
#include "Force/Force.h"

namespace redmax {

class Body;

class ForcePointToPoint : public Force {
public:
    dtype _stiffness;
    dtype _damping;
    std::string _frame;
    VectorX _f;

    Body* _body0; // body to apply the forces.
    std::vector<Vector3> _xls0; // application locations in the local coordinate.
    Body* _body1; // body to apply the forces.
    std::vector<Vector3> _xls1; // application locations in the local coordinate.

    ForcePointToPoint(Simulation* sim, dtype stiffness = 1., dtype damping = 1., std::string frame = "joint");

    void set_stiffness(dtype stiffness);
    void set_damping(dtype damping);

    void addBodies(Body* body0, Body* body1);
    void addPoints(Vector3 loc0, Vector3 loc1);

    void computeForce(VectorX& fm, VectorX& fr, bool verbose = false);
    void computeForceWithDerivative(VectorX& fm, VectorX& fr, MatrixX& Km, MatrixX& Dm, MatrixX& Kr, MatrixX& Dr, bool verbose = false);

    VectorX getComputedForce() { return _f; };
};

}