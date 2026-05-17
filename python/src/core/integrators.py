import numpy as np
from typing import Tuple
from scipy.sparse import coo_matrix
from .smoothing import eval_H_smooth
from .hamiltonian import compute_H

def _hamiltonian_gradients(problem, bundle, p, x, t, delta, dt=None, use_explicit_gradients=False):
    if use_explicit_gradients and problem.hamiltonian_grad_fn is not None:
        return None, *problem.hamiltonian_gradients(x, p, t)
    return eval_H_smooth(problem, bundle, p, x, t, delta, dt=dt)



def pack_unknowns(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    """
    Flatten state and costate trajectories into a single vector.

    X has shape (N+1, n) and contains x_0,...,x_N.
    P has shape (N+1, n) and contains p_0,...,p_N.
    x_0 is fixed by the problem and should not be included in the unknown vector.

    The returned vector z concatenates x_1,...,x_N followed by p_0,...,p_N.
    """
    N_plus_1, n = X.shape
    N = N_plus_1 - 1
    z = np.zeros((N * n + (N + 1) * n,))
    # pack x_1,...,x_N
    z[0:N * n] = X[1:, :].reshape(N * n)
    # pack p_0,...,p_N
    z[N * n:] = P.reshape((N + 1) * n)
    return z

def unpack_unknowns(z: np.ndarray, x0: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse of pack_unknowns.

    Parameters
    ----------
    z : np.ndarray
        Flattened vector of unknowns with length N*n + (N+1)*n.
    x0 : np.ndarray
        The initial state (x_0), which is fixed and not included in z.

    Returns
    -------
    X : np.ndarray
        State trajectory array of shape (N+1, n).
    P : np.ndarray
        Costate trajectory array of shape (N+1, n).
    """
    n = x0.size
    # deduce N such that z.size = (2*N + 1) * n
    total = z.size // n
    # total = 2*N + 1 -> N = (total - 1) // 2
    N = (total - 1) // 2
    N_plus_1 = N + 1
    X = np.zeros((N_plus_1, n))
    P = np.zeros((N_plus_1, n))
    # assign x_0
    X[0, :] = x0
    # x_1,...,x_N
    X[1:, :] = z[0:N * n].reshape((N, n))
    # p_0,...,p_N
    P[:, :] = z[N * n:].reshape((N_plus_1, n))
    return X, P

def assemble_residual(problem, t_nodes: np.ndarray, X: np.ndarray, P: np.ndarray, bundle, delta: float, use_explicit_gradients: bool = False) -> np.ndarray:
    """
    Assemble the residual vector for the symplectic Euler discretisation.

    Parameters
    ----------
    problem : OCPProblem
        Optimal control problem instance.
    t_nodes : np.ndarray
        Array of time nodes of length N+1.
    X : np.ndarray
        Array of shape (N+1, n) containing state trajectory (including x_0).
    P : np.ndarray
        Array of shape (N+1, n) containing costate trajectory.
    bundle : PABundle
        Bundle used for smoothing in the Hamiltonian.
    delta : float
        Smoothing parameter.

    Returns
    -------
    np.ndarray
        Residual vector of length 2*N*n + n.
    """
    N_plus_1 = t_nodes.size # number of time nodes = N+1
    N = N_plus_1 - 1 #number of steps
    n = X.shape[1] #dimension of the state
    residual = np.zeros((2 * N * n + n,))
    offset = 0
    for i in range(N):
        dt = t_nodes[i + 1] - t_nodes[i]
        x_i = X[i]
        x_ip1 = X[i + 1]
        p_i = P[i]
        p_ip1 = P[i + 1]
        # # gradients evaluated at (p_{i+1}, x_i, t_i)
        _, grad_p, grad_x = _hamiltonian_gradients(problem, bundle, p_ip1, x_i, t_nodes[i], delta, dt=dt, use_explicit_gradients=use_explicit_gradients)
        # state residual r_x = x_i + dt * grad_p - x_{i+1}
        r_x = x_ip1 - x_i - dt * grad_p  
        residual[offset:offset + n] = r_x
        offset += n
        # costate residual r_p = p_{i+1} + dt * grad_x_ip1 - p_i
        r_p = p_i - p_ip1 - dt * grad_x
        residual[offset:offset + n] = r_p
        offset += n
    # terminal boundary condition: p_N + ∇g(x_N) = 0
    x_N = X[-1]
    p_N = P[-1]
    # gradient of g by finite difference
    g_grad = np.zeros_like(p_N)
    eps = 1e-6
    for j in range(n):
        x_plus = x_N.copy()
        x_minus = x_N.copy()
        x_plus[j] += eps
        x_minus[j] -= eps
        g_plus = problem.g(x_plus)
        g_minus = problem.g(x_minus)
        g_grad[j] = (g_plus - g_minus) / (2 * eps)
    r_bc = p_N - g_grad
    residual[offset:] = r_bc
    return residual

def assemble_jacobian(problem, t_nodes: np.ndarray, X: np.ndarray, P: np.ndarray, bundle, delta: float, use_explicit_gradients: bool = False):
    """
    Assemble the Jacobian matrix dF/dz exploiting locality (block stencil) for
    the symplectic Euler residual used in assemble_residual.

    Unknown vector order (repo):
        z = (x_1,...,x_N, p_0,...,p_N)  in R^{(2N+1)n}

    Residual order:
        F = (r_x^0, r_p^0, r_x^1, r_p^1, ..., r_x^{N-1}, r_p^{N-1}, r_bc)

    This routine builds the SAME Jacobian as the dense implementation, but
    returns a sparse matrix (CSR). The sparsity pattern follows the local
    dependencies:
        r_x^i depends on (x_i, x_{i+1}, p_{i+1})
        r_p^i depends on (x_i, p_i, p_{i+1})
        r_bc depends on (x_N, p_N)
    """
    

    N_plus_1 = t_nodes.size
    N = N_plus_1 - 1
    n = X.shape[1]

    m = (2 * N + 1) * n  # number of unknowns = number of equations
    #eps = 1e-7  # FD step for local Jacobian blocks

    #================= ADDED-=====================
    eps_x = 1e-7  # FD step for perturbing X
    eps_p = 1e-7  # FD step for perturbing P

    #if use_explicit_gradients:
        #for the paper regularization, p-sensitivity is on scale δ near p≈0
        #eps_p = max(1e-14, min(eps_p, 1e-2 * float(delta)))
    #===========================================================
    # -------------------------
    # Index maps (repo order)
    # -------------------------
    def col_x(k: int) -> int:
        """Start column (0-based) of block x_k in z. Only valid for k=1..N."""
        return (k - 1) * n

    def col_p(j: int) -> int:
        """Start column (0-based) of block p_j in z. Valid for j=0..N."""
        return N * n + j * n

    def row_rx(i: int) -> int:
        """Start row of block r_x^i in F."""
        return (2 * i) * n

    def row_rp(i: int) -> int:
        """Start row of block r_p^i in F."""
        return (2 * i + 1) * n

    row_bc = (2 * N) * n

    # -------------------------
    # Local nonlinear parts
    # -------------------------
    def phi(i: int) -> np.ndarray:
        """
        Nonlinear part of r_x^i:
            r_x^i = (x_{i+1} - x_i) + phi
            phi = -dt * grad_p H_delta(p_{i+1}, x_i, t_i)
        """
        dt = t_nodes[i + 1] - t_nodes[i]
        _, grad_p, _ = _hamiltonian_gradients(problem, bundle, P[i + 1], X[i], t_nodes[i], delta, dt=dt, use_explicit_gradients=use_explicit_gradients)
        return -dt * grad_p

    def psi(i: int) -> np.ndarray:
        """
        Nonlinear part of r_p^i:
            r_p^i = (p_i - p_{i+1}) + psi
            psi = -dt * grad_x H_delta(p_{i+1}, x_i, t_i)
        """
        dt = t_nodes[i + 1] - t_nodes[i]
        _, _, grad_x = _hamiltonian_gradients(problem, bundle, P[i + 1], X[i], t_nodes[i], delta, dt=dt, use_explicit_gradients=use_explicit_gradients)
        return -dt * grad_x

    def bc_block() -> np.ndarray:
        """
        Boundary residual block:
            r_bc = p_N - grad g(x_N)
        grad g computed by central differences (same style as assemble_residual).
        """
        xN = X[-1]
        pN = P[-1]
        g_grad = np.zeros_like(pN)
        epsg = 1e-6
        for j in range(n):
            x_plus = xN.copy()
            x_minus = xN.copy()
            x_plus[j] += epsg
            x_minus[j] -= epsg
            g_plus = problem.g(x_plus)
            g_minus = problem.g(x_minus)
            g_grad[j] = (g_plus - g_minus) / (2 * epsg)
        return pN - g_grad

    # ============================================================
    # Build sparse matrix via triplets (COO), then convert to CSR.
    # This avoids fragile sparse slicing/+= behavior.
    # ============================================================
    rows = []
    cols = []
    data = []

    def add_I(rr0: int, cc0: int, sign: float):
        """Add sign*I_n block at (rr0, cc0)."""
        for r in range(n):
            rows.append(rr0 + r)
            cols.append(cc0 + r)
            data.append(sign)

    for i in range(N):
        rr_x = row_rx(i)
        rr_p = row_rp(i)

        # ---------- r_x^i linear blocks ----------
        # d r_x^i / d x_{i+1} = I
        cx_ip1 = col_x(i + 1)
        add_I(rr_x, cx_ip1, +1.0)

        # d r_x^i / d x_i has linear part -I (only if x_i is an unknown => i>=1)
        if i >= 1:
            cx_i = col_x(i)
            add_I(rr_x, cx_i, -1.0)

        # ---------- r_p^i linear blocks ----------
        # d r_p^i / d p_i = I
        cp_i = col_p(i)
        add_I(rr_p, cp_i, +1.0)

        # d r_p^i / d p_{i+1} has linear part -I
        cp_ip1 = col_p(i + 1)
        add_I(rr_p, cp_ip1, -1.0)

        # ---------- local FD: add nonlinear contributions ----------
        # dphi/dx_i and dpsi/dx_i (only if x_i unknown)
        if i >= 1:
            cx_i = col_x(i)
            for ell in range(n):
                old = X[i, ell]

                X[i, ell] = old + eps_x
                phi_p = phi(i)
                psi_p = psi(i)

                X[i, ell] = old - eps_x
                phi_m = phi(i)
                psi_m = psi(i)

                X[i, ell] = old

                dphi = (phi_p - phi_m) / (2 * eps_x)  # column ell of dphi/dx_i
                dpsi = (psi_p - psi_m) / (2 * eps_x)  # column ell of dpsi/dx_i

                c = cx_i + ell
                for r in range(n):
                    rows.append(rr_x + r); cols.append(c); data.append(float(dphi[r]))
                    rows.append(rr_p + r); cols.append(c); data.append(float(dpsi[r]))

        # dphi/dp_{i+1} and dpsi/dp_{i+1} (p_{i+1} always unknown)
        for ell in range(n):
            old = P[i + 1, ell]

            P[i + 1, ell] = old + eps_p
            phi_p = phi(i)
            psi_p = psi(i)

            P[i + 1, ell] = old - eps_p
            phi_m = phi(i)
            psi_m = psi(i)

            P[i + 1, ell] = old

            dphi = (phi_p - phi_m) / (2 * eps_p)  # column ell of dphi/dp_{i+1}
            dpsi = (psi_p - psi_m) / (2 * eps_p)  # column ell of dpsi/dp_{i+1}

            c = cp_ip1 + ell
            for r in range(n):
                rows.append(rr_x + r); cols.append(c); data.append(float(dphi[r]))
                rows.append(rr_p + r); cols.append(c); data.append(float(dpsi[r]))

    # ---------- boundary condition blocks ----------
    # d r_bc / d p_N = I
    cp_N = col_p(N)
    add_I(row_bc, cp_N, +1.0)

    # d r_bc / d x_N by FD
    cx_N = col_x(N)
    for ell in range(n):
        old = X[N, ell]

        X[N, ell] = old + eps_x
        fp = bc_block()

        X[N, ell] = old - eps_x
        fm = bc_block()

        X[N, ell] = old

        dcol = (fp - fm) / (2 * eps_x)
        c = cx_N + ell
        for r in range(n):
            rows.append(row_bc + r)
            cols.append(c)
            data.append(float(dcol[r]))

    J = coo_matrix((data, (rows, cols)), shape=(m, m)).tocsr()
    J.sum_duplicates()
    return J



""" def assemble_jacobian(problem, t_nodes: np.ndarray, X: np.ndarray, P: np.ndarray, bundle, delta: float) -> np.ndarray:

    #Assemble the Jacobian matrix of the residual with respect to the unknowns.

    #The unknown vector consists of x_1,...,x_N, p_0,...,p_N.  The Jacobian
    #therefore has shape ((2*N*n + n), (2*N*n + n)).  We compute it by
    #finite differences on the residual function.  This is expensive for large
    #problems but suffices for moderate N and n.

    #Returns
    -------
    #np.ndarray
        #Full Jacobian matrix.
    
    # pack current unknown vector
    z = pack_unknowns(X, P)
    # function to compute residual
    def res_fun(z_vec: np.ndarray) -> np.ndarray:
        X_new, P_new = unpack_unknowns(z_vec, X[0])
        return assemble_residual(problem, t_nodes, X_new, P_new, bundle, delta)
    F0 = res_fun(z)
    m = z.size
    k = F0.size
    J = np.zeros((k, m))
    eps = 1e-6
    # finite differences
    for j in range(m):
        z_plus = z.copy()
        z_minus = z.copy()
        z_plus[j] += eps
        z_minus[j] -= eps
        F_plus = res_fun(z_plus)
        F_minus = res_fun(z_minus)
        J[:, j] = (F_plus - F_minus) / (2 * eps)
    return J """
