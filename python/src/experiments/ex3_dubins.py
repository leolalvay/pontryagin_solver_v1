"""
Example 3: Dubins car with bounded turn rate.

The Dubins car has state (x, y, θ) and control u controlling the turn rate.
We seek to drive the car from (0,0,0) to (1,1,π/2) while minimizing a running cost
of the form 0.1 + u^2 and a soft terminal penalty.  Controls satisfy |u| ≤ 1.
"""
import numpy as np

from core.problem import OCPProblem
from core.adaptivity import solve_optimal_control
from core.hamiltonian import compute_H


def run_example():
    # dynamics for Dubins car
    def dynamics(x, u, t):
        theta = x[2]
        v = 1.0
        return np.array([v * np.cos(theta), v * np.sin(theta), u[0]])
    # stage cost: soft time and quadratic control
    def stage_cost(x, u, t):
        return 0.1 + u[0] ** 2
    # terminal target and penalty
    target = np.array([1.0, 1.0, np.pi / 2])
    penalty_weight = 10.0
    def terminal_cost(x):
        diff = x - target
        return penalty_weight * diff.dot(diff)
    # initial state and horizon guess
    x0 = np.array([0.0, 0.0, 0.0])
    T = 3.0
    # control bounds
    u_min = np.array([-1.0])
    u_max = np.array([1.0])
    # no state constraints
    prob = OCPProblem(dynamics, stage_cost, terminal_cost, x0, T,
                      control_bounds=(u_min, u_max), state_bounds=None)
    # initial mesh
    t_nodes = np.linspace(0.0, T, 31)
    # solve adaptively
    result = solve_optimal_control(prob, t_nodes, tol_time=5e-3, tol_PA=1e-3, tol_delta=1e-3, max_iters=10, delta0=0.2)
    X = result['X']
    P = result['P']
    mesh = result['t_nodes']
    bundle = result['bundle']
    # approximate controls and count switches
    controls = []
    for i in range(len(mesh)):
        _, u_star = compute_H(prob, P[i], X[i], mesh[i], bundle.controls, restricted=True)
        controls.append(u_star)
    controls = np.asarray(controls)
    # count sign changes
    signs = np.sign(controls[:, 0])
    sign_changes = np.sum(np.abs(np.diff(signs)) > 1e-6)
    print("Dubins Car Example")
    print(f"Mesh points: {len(mesh)}")
    print(f"Planes: {bundle.num_planes()}")
    print(f"Number of control switches: {sign_changes}")
    print("Indicator history:")
    for entry in result['log']:
        print(entry)
    return result


if __name__ == '__main__':
    run_example()
