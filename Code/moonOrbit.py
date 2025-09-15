import numpy as np
from scipy.integrate import solve_bvp, solve_ivp
from scipy.interpolate import interp1d
from scipy.linalg import solve_continuous_are


# PMP -----------------------------------------------------
def ivp_dynamics(t, state, GM, eps=1e-6):
    r, _, vr, w = state
    r_safe = max(r, eps)
    drdt = vr
    dthetadt = w
    dvrdt = -GM / r_safe**2 + r_safe * w**2
    dwdt = -2.0 * vr * w / r_safe
    return [drdt, dthetadt, dvrdt, dwdt]


def PMP_dynamics(t, z, GM, eps=1e-6, alpha=0.01):
    # z has shape (7, n)
    r = z[0, :]
    vr = z[2, :]
    w = z[3, :]
    lam_r = z[4, :]
    lam_vr = z[5, :]
    lam_w = z[6, :]

    # safe r
    r_safe = np.maximum(r, eps)

    # optimal controls (smooth)
    u_r = -lam_vr / (2.0 * alpha)
    u_theta = -lam_w / (2.0 * alpha * r_safe)

    # state derivatives
    drdt = vr
    dthetadt = w
    dvrdt = u_r - GM / r_safe**2 + r_safe * w**2
    dwdt = (u_theta - 2.0 * vr * w) / r_safe

    # costate derivatives
    term1 = lam_vr * (2.0 * GM / (r_safe**3) + w**2)
    term2 = -lam_w * (u_theta - 2.0 * vr * w) / (r_safe**2)
    dlam_rdt = -(term1 + term2)

    dlam_vrdt = -lam_r + 2.0 * lam_w * w / r_safe
    dlam_wdt = -(2.0 * lam_vr * r_safe * w - 2.0 * lam_w * vr / r_safe)

    return np.vstack([drdt, dthetadt, dvrdt, dwdt, dlam_rdt, dlam_vrdt, dlam_wdt])


def bc(za, zb, r0, theta0, vr0, w0, r1, GM):
    # za = z(0), zb = z(T)
    # initial state constraints
    c0 = za[0] - r0
    c1 = za[1] - theta0
    c2 = za[2] - vr0
    c3 = za[3] - w0

    # terminal costate constraints
    c4 = zb[4] - 2.0 * (zb[0] - r1)  # lambda_r(T)
    c5 = zb[5] - 2.0 * zb[2]  # lambda_vr(T)
    c6 = zb[3] - np.sqrt(GM / r1**3)  # lambda_w(T)

    return np.array([c0, c1, c2, c3, c4, c5, c6])


def solvePMP(GM, T, r0, r1, theta0, w0):
    N = 400
    t = np.linspace(0.0, T, N)
    z_guess = np.zeros((7, N))

    # sensible guesses, linear
    circ_w = np.sqrt(GM / (r1**3))
    z_guess[0, :] = np.linspace(r0, r1, N)
    z_guess[1, :] = theta0 + circ_w * t
    z_guess[2, :] = np.linspace(0.0, 0.0, N)
    z_guess[3, :] = np.linspace(w0, circ_w, N)

    # costate guesses of zero
    z_guess[4:, :] = 0.0

    sol = solve_bvp(
        lambda x, y: PMP_dynamics(x, y, GM=GM),
        lambda x, y: bc(x, y, r0=r0, theta0=theta0, vr0=0, w0=w0, r1=r1, GM=GM),
        t,
        z_guess,
        verbose=2,
        max_nodes=20000,
        tol=1e-6,
    )
    print("solve_bvp success:", sol.success, "| message:", sol.message)
    return sol.sol


def getControlPMP_for_t(sol, t, eps=1e-6, alpha=0.01):
    v_arr = sol(t)
    # extract solution
    r = v_arr[0]
    vr = v_arr[2]
    w = v_arr[3]
    lam_vr = v_arr[5]
    lam_w = v_arr[6]

    # controls
    u_r = -lam_vr / (2.0 * alpha)
    u_theta = -lam_w / (2.0 * alpha * np.maximum(r, eps))
    return u_r, u_theta, r, vr, w


# HJB ----------------------------------------------------------------------------


def get_control_K(GM, r1, w1, use_vr=False):
    q = 10.0
    # Create matrix:
    A = np.array([[0, 1, 0], [GM / (r1**2), 0, 2 * w1], [0, -w1, 0]])
    B = np.array([[0, 0], [1, 0], [0, 1]])
    Q = np.diag([q, 0.1 * use_vr * q, 0.1 * q])
    R = np.diag([0.1, 0.1])
    P = solve_continuous_are(A, B, Q, R)
    K = np.linalg.inv(R) @ B.T @ P
    return K


def dynamics(state, mu, u):
    r, v_r, v_t = state
    r_dot = v_r
    v_r_dot = (v_t**2) / r - mu / (r**2) + u[0]
    v_t_dot = -(v_r * v_t) / r + u[1]
    theta_dot = v_t / r
    return np.array([r_dot, v_r_dot, v_t_dot]), theta_dot


# time vary --------------------------------------------------------------------


def build_time_varying_lqr_controller(
    ref_state_fn,
    ref_control_fn,
    t_grid,
    R,
    Q,
    S,
    mu=1,
    thruster_limits=None,
):
    t_grid = np.asarray(t_grid)

    Rinv = np.linalg.inv(R)

    B = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    # Linearization about ref state
    def A_of_t(t):
        r, vr, vt = map(float, ref_state_fn(t))
        A = np.zeros((3, 3))
        A[0, 0] = 0.0
        A[0, 1] = 1.0
        A[0, 2] = 0.0
        A[1, 0] = -(vt**2) / r**2 + 2.0 * mu / r**3
        A[1, 1] = 0.0
        A[1, 2] = 2.0 * vt / r
        A[2, 0] = vr * vt / r**2
        A[2, 1] = -vt / r
        A[2, 2] = -vr / r
        return A

    # if no feedforward supplied, use zero feedforward
    if ref_control_fn is None:

        def ref_control_fn_local(t):
            return np.zeros(2)
    else:
        ref_control_fn_local = ref_control_fn

    # RDE RHS:
    def rde_rhs(t, Pflat):
        P = Pflat.reshape(3, 3)
        A = A_of_t(t)
        RHS = A.T @ P + P @ A - P @ B @ Rinv @ B.T @ P + Q
        dPdt = -RHS
        return dPdt.reshape(-1)

    # integrate backward from TF to T0 using t_grid reversed
    T0, TF = t_grid[0], t_grid[-1]
    P0_flat = S.reshape(-1)
    sol = solve_ivp(
        rde_rhs,
        (TF, T0),
        P0_flat,
        t_eval=t_grid[::-1],
        method="RK45",
        atol=1e-8,
        rtol=1e-6,
    )

    # reorder solution to ascending time t_grid
    nT = t_grid.size
    P_stack = np.zeros((nT, 3, 3))
    for i in range(nT):
        idx = nT - 1 - i
        P_stack[i] = sol.y[:, idx].reshape(3, 3)

    # build smooth interpolants for P entries so we can query P(t) cheaply
    P_interps = []
    for i in range(3):
        for j in range(3):
            P_interps.append(
                interp1d(
                    t_grid, P_stack[:, i, j], kind="cubic", fill_value="extrapolate"
                )
            )

    def P_of_t(t):
        M = np.zeros((3, 3))
        k = 0
        for i in range(3):
            for j in range(3):
                M[i, j] = float(P_interps[k](t))
                k += 1
        # ensure symmetry
        return 0.5 * (M + M.T)

    def K_of_t(t):
        P = P_of_t(t)
        return Rinv @ B.T @ P

    # controller:
    def controller(t, x_abs):
        x_abs = np.asarray(x_abs).reshape(
            3,
        )
        xref = np.asarray(ref_state_fn(t)).reshape(
            3,
        )
        uref = np.asarray(ref_control_fn_local(t)).reshape(
            2,
        )
        Kt = K_of_t(t)
        u = uref - Kt.dot(x_abs - xref)
        if thruster_limits is not None:
            lim = np.asarray(thruster_limits)
            u = np.clip(u, -lim, lim)
        return u

    return controller, K_of_t


def reference_state(t, sol):
    v_arr = sol(t)
    # extract solution
    r = v_arr[0]
    vr = v_arr[2]
    w = v_arr[3]
    return np.array([r, vr, w])


# Simulation --------------------------------------------------------------
def simulate_orbit(
    r1,
    mu,
    x0,
    dt=0.1,
    T=600,
    K=None,
    u_max=None,
    perturb=None,
    usePMP=False,
    PMP_sol=None,
    combination=False,
    time_vary=False,
    controller=None,
    overTake=0,
):
    N = int(T // dt) + 1
    t = np.linspace(0.0, T, N)
    use_perturb = perturb is not None
    use_K = K is not None

    # initial absolute state
    r0, theta0, v_r0, w0 = x0
    w1 = np.sqrt(mu / r1)

    # state vector:
    state = np.array([r0, v_r0, w0])

    x = np.zeros((3, N))  # deviations
    u_arr = np.zeros((2, N))
    r_arr, theta_arr, v_r_arr, v_t_arr = (
        np.zeros(N),
        np.zeros(N),
        np.zeros(N),
        np.zeros(N),
    )

    # store initial
    x[:, 0] = np.array([r0 - r1, v_r0, w0 - w1])
    r_arr[0], v_r_arr[0], v_t_arr[0] = state
    theta_arr[0] = theta0

    for k in range(N - 1):
        # deviation for feedback
        dev = np.array([state[0] - r1, state[1], state[2] - w1])

        # control
        u = np.zeros(2)
        r_pmp, w_pmp = None, None
        if use_K:
            u = -K @ dev
        if usePMP:
            u_r, u_theta, r_pmp, _, w_pmp = getControlPMP_for_t(PMP_sol, k * dt)
            u = [u_r, u_theta]
        if combination:
            K = get_control_K(mu, r_pmp, w_pmp, use_vr=False)
            u = -K @ dev
        if time_vary:
            u = controller(0, state)
            if (r_pmp / r1) < overTake:
                K = get_control_K(mu, r_pmp, w_pmp, use_vr=False)
                u = -K @ dev
        if u_max is not None:
            u = np.clip(u, -u_max, u_max)
        u_arr[:, k] = u

        # RK4 integration
        k1, theta_dot_1 = dynamics(state, mu, u)
        k2, theta_dot_2 = dynamics(state + 0.5 * dt * k1, mu, u)
        k3, theta_dot_3 = dynamics(state + 0.5 * dt * k2, mu, u)
        k4, theta_dot_4 = dynamics(state + dt * k3, mu, u)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if use_perturb:
            state[1] = state[1] + perturb[0, k]
            state[2] = state[2] + perturb[1, k]

        # log
        r_arr[k + 1], v_r_arr[k + 1], v_t_arr[k + 1] = state
        theta_arr[k + 1] = theta_arr[k] + (dt / 6.0) * (
            theta_dot_1 + 2 * theta_dot_2 + 2 * theta_dot_3 + theta_dot_4
        )

    u_arr[:, -1] = u_arr[:, -2]
    return t, x, u_arr, r_arr, theta_arr, v_r_arr, v_t_arr
