"""Boundary value problem solver."""

import jax
import jax.numpy as jnp


# ------------------------------------------------------------------------------------------
# Linear solver for bordered almost block diagonal (BABD) systems
# ------------------------------------------------------------------------------------------
# Implementation as described in [1] Section 2.1 (structural orthogonal factorization).
# [1] M. Dessole and F. Marcuzzi, "A massively parallel algorithm for Bordered Almost Block
#     Diagonal Systems on GPUs", 2022.

def getQR(T, S):
    """ Equation (9) """
    A = jnp.r_[T, S]  # (2n, n)
    Q, R = jnp.linalg.qr(A, mode='complete')
    U = R[:R.shape[0] // 2]
    return Q, U


def forward_reduce_sys(S, T):
    """ Forward reduces until Equation (12), LHS """

    def body(carry, x):
        W_hat, V_hat = carry
        Si, Ti = x

        n = Si.shape[-1]
        Q, U = getQR(W_hat, Si)

        V = Q[:n].T @ V_hat
        W = Q[n:].T @ Ti

        carry = (W[n:], V[n:])
        y = (Q, [U, V[:n], W[:n]])
        return carry, y

    init = (T[0], S[0])
    xs = (S[1:], T[1:])
    return jax.lax.scan(body, init, xs)


def forward_reduce_b(Q, b):
    """ Forward reduces until Equation (12), RHS using the
        QR decompositions computed from `forward_reduce_sys`.
    """

    def body(f_hat, x):
        Q, bi = x
        f = Q.T @ jnp.r_[f_hat, bi]

        n = f_hat.shape[0]
        return f[n:], f[:n]

    init = b[0]
    xs = (Q, b[1:])
    return jax.lax.scan(body, init, xs)


def back_substitute(params, x0, xn):
    """ Recursively solves Equation (14) """

    def body(last_x, params):
        U, V, W, f = params
        b = f - V @ x0 - W @ last_x
        x = jax.scipy.linalg.solve_triangular(U, b)
        return x, x

    _, x = jax.lax.scan(body, xn, params, reverse=True)
    return x


def BABD_factor(S, T, Ba, Bb):
    """Factor the BABD system until its reduced form system.

    The entire system is reduced as it is (i.e., only one slice, P=1)
    since I found no computational advantage to vmapping the forward
    reduction procedure over more slices P > 1.

    Returns `(lu_dec, (Q, back_params))`, where `lu_dec` is the LU
    decomposition of the reduced system, `Q` is the set of QR
    decompositions used to reduce the system (needed when reducing
    the vector `b` in `forward_reduce_b`), and `back_params` is used
    for the back subsitution in `back_substitute` when solving.

    Parameters
    ----------
    S : ndarray, shape (m, n, n)
        Block diagonal.
    T : ndarray, shape (m, n, n)
        Block off-diagonal.
    Ba, Bb : ndarray, shape (n, n)
        Blocks at the last block row, first and last block columns.

    References
    ----------
    ..  [1] M. Dessole and F. Marcuzzi, "A massively parallel algorithm
            for Bordered Almost Block Diagonal Systems on GPUs", 2022.
    """
    # Reduce the system so that only (2n x 2n) has to be solved
    (Tr, Sr), others = forward_reduce_sys(S, T)

    # Cholesky decomposition of the reduced system
    M = jnp.r_[jnp.c_[Sr, Tr], jnp.c_[Ba, Bb]]
    lu_dec = jax.scipy.linalg.lu_factor(M)

    return lu_dec, others


def BABD_solve(factorization, b):
    """ Solve the BABD system Ax=b, given the factorization of A.

    Parameters
    ----------
    factorization : tuple
        As returned by `BABD_factor`.
    b : ndarray, shape (m*n, ) or (m, n)
        Values for which to solve the BABD system Ax=b.
    """
    lu_dec, (Q, back_params) = factorization
    b = b.reshape(-1, Q.shape[-1] // 2)

    # Apply the same transformations to b
    br, f = forward_reduce_b(Q, b[:-1])

    # Solve the reduced system
    br = jnp.r_[br, b[-1]]
    x_r = jax.scipy.linalg.lu_solve(lu_dec, br)

    x0 = x_r[:x_r.shape[0] // 2]
    xn = x_r[x_r.shape[0] // 2:]
    x_m = back_substitute(back_params + [f], x0, xn)

    x = jnp.r_[x0, x_m.reshape(-1), xn]
    return x


# ------------------------------------------------------------------------------------------
# Functions to compute the sparse Jacobian of the collocation system
# ------------------------------------------------------------------------------------------
# Naturally, significantly more efficent that jax.jacfwd on the collocation objective.

def construct_jac(h, df_dy, df_dy_middle, dbc_dya, dbc_dyb):
    """Construct the Jacobian of the collocation system.

    There are m * d functions: m - 1 collocations residuals, each
    containing d components, followed by d boundary condition residuals.

    There are m * d variables: m vectors of y, each containing d components.

    For example, let m = 4, and d = 2 then the Jacobian will have
    the following sparsity structure, named bordered almost block diagonal (BABD):

        1 1 2 2 0 0 0 0
        1 1 2 2 0 0 0 0
        0 0 1 1 2 2 0 0
        0 0 1 1 2 2 0 0
        0 0 0 0 1 1 2 2
        0 0 0 0 1 1 2 2

        3 3 0 0 0 0 4 4
        3 3 0 0 0 0 4 4

    Zeros denote identically zero values, other values denote different kinds
    of blocks in the matrix (see below). The blank row indicates the separation
    of collocation residuals from boundary conditions. And the blank column
    indicates the separation of y values from p values.

    Refer to [1]_  (p. 306) for the formula of n x n blocks for derivatives
    of collocation residuals with respect to y.

    We computed the structured orthogonal factorization of the BABD system as
    described in [2]_ (Section 2.1). This is much more efficient that the linear
    solvers provided by jax, and competitive with the sparse LU decomposition
    used in Scipy's implementation.

    Parameters
    ----------
    df_dy : ndarray, shape (m, n, n)
        Jacobian of f with respect to y computed at the mesh nodes.
        Corresponds to the block diagonal (represented as 1s above).
    df_dy_middle : ndarray, shape (m-1, d, d)
        Jacobian of f with respect to y computed at the middle between the mesh nodes.
        Coresponds to the block off-diagonal (represented as 2s above).
    dbc_dya, dbc_dyb : ndarray, shape (d, d)
        Jacobian of bc with respect to ya and yb.
        Corresponds to the last block row, represented as (3s and 4s above).

    Returns
    -------
    factorization : tuple
        Strcutred orthogonal factorization of the BABD system. See `BABD_factor`.

    References
    ----------
    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
       Number 3, pp. 299-316, 2001.
    .. [2] M. Dessole and F. Marcuzzi, "A massively parallel algorithm for Bordered
       Almost Block Diagonal Systems on GPUs", 2022.
    """
    m, n, _ = df_dy.shape

    h = h[:, jnp.newaxis, jnp.newaxis]  # (m-1, 1, 1)

    # Computing diagonal d x d blocks.
    dPhi_dy_0 = -jnp.identity(n)[jnp.newaxis].repeat(m - 1, axis=0)
    dPhi_dy_0 = dPhi_dy_0 - h / 6 * (df_dy[:-1] + 2 * df_dy_middle)
    T = df_dy_middle @ df_dy[:-1]
    dPhi_dy_0 = dPhi_dy_0 - h ** 2 / 12 * T

    # Computing off-diagonal d x d blocks.
    dPhi_dy_1 = jnp.identity(n)[jnp.newaxis].repeat(m - 1, axis=0)
    dPhi_dy_1 = dPhi_dy_1 - h / 6 * (df_dy[1:] + 2 * df_dy_middle)
    T = df_dy_middle @ df_dy[1:]
    dPhi_dy_1 = dPhi_dy_1 + h ** 2 / 12 * T

    return BABD_factor(dPhi_dy_0, dPhi_dy_1, dbc_dya, dbc_dyb)


def prepare_jac(fun, bc):
    """Returns a function which evaluates the Jacobian of the collocation system."""
    fun_jac = jax.vmap(jax.jacfwd(fun, argnums=1))
    bc_jac = jax.jacfwd(bc, argnums=([0, 1]))

    def sys_jac(x, h, y, y_middle):
        """Evaluates the Jacobian of the collocation system.

        Note that by requiring y_middle as an argument, the system must be evaluated
        one additional time at the start of the Newton solve.

        Returns a factorization of the Jacobian as detailed in `construct_jac`.

        Parameters
        ---------
        x : ndarray, shape (m, )
            Nodes of the mesh.
        h : ndarray, shape (m-1, )
            Increment between mesh nodes (i.e., np.diff(x)).
        y : ndarray, shape (m, n)
            Solution values at the mesh nodes.
        y_middle : ndarray, shape (m-1, n)
            Solution values at the mid-points of each mesh interval.
        """
        m = x.shape[0]

        # Compute the derivatives at the mesh points and its midpoints
        x_middle = x[:-1] + 0.5 * h
        x_xm = jnp.r_[x, x_middle]
        y_ym = jnp.r_[y, y_middle]

        df_dyym = fun_jac(x_xm, y_ym)
        dbc_dya, dbc_dyb = bc_jac(y[0], y[-1])

        return construct_jac(h, df_dyym[:m], df_dyym[m:], dbc_dya, dbc_dyb)

    return sys_jac


# ------------------------------------------------------------------------------------------
# Functions to evaluate the collocation residuals
# ------------------------------------------------------------------------------------------

def compute_ymiddle(fun, y, x, h):
    """Evaluate the solution at the middle points of the mesh intervals.

    Note that the solution to the BVP is sought as a cubic C1 continous spline with
    derivatives matching the ODE rhs at given nodes `x`.

    This function is needed to evaluate the Jacobian of the collocation system.

    For the Parameters and Returns, see `collocation_fun`.
    """
    h = h[:, jnp.newaxis]
    f = jax.vmap(fun)(x, y)
    y_middle = (0.5 * (y[1:] + y[:-1]) - 0.125 * h * (f[1:] - f[:-1]))
    return y_middle, f


def collocation_fun(fun, y, x, h):
    """Evaluate collocation residuals.

    The solution to the BVP is sought as a cubic C1 continuous spline with
    derivatives matching the ODE rhs at given nodes `x`. Collocation conditions
    are formed from the equality of the spline derivatives and rhs of the ODE
    system in the middle points between nodes.

    Such method is classified to Lobbato IIIA family in ODE literature.
    Refer to [1]_ for the formula and some discussion.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(x, y)``.
        All arguments are ndarray: ``x`` with shape (,), ``y`` with shape (n,).
        The return value must be an array with shape (n,).
    y : ndarray, shape (m, n)
        Solution values at the mesh nodes.
    x : ndarray, shape (m,)
        Nodes of the mesh.
    h : ndarray with shape (m-1,)
        Increment between the mesh nodes, that is, jnp.diff(x).

    Returns
    -------
    col_res : ndarray, shape (m-1, n)
        Collocation residuals at the middle points of the mesh intervals.
    y_middle : ndarray, shape (m-1, n)
        Values of the cubic spline evaluated at the middle points of the mesh intervals.
    f : ndarray, shape (m, n)
        RHS of the ODE system evaluated at the mesh nodes.
    f_middle : ndarray, shape (m-1, n)
        RHS of the ODE system evaluated at the middle points of the mesh
        intervals (and using `y_middle`).

    References
    ----------
    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
           Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
           Number 3, pp. 299-316, 2001.
    """
    # y_middle: (m-1, d), f: (m, d)   with h as 1-D here
    y_middle, f = compute_ymiddle(fun, y, x, h)

    # Use 1-D h to form midpoints, then expand h only for residual formula
    x_mid = x[:-1] + 0.5 * h            # (m-1,)
    f_middle = jax.vmap(fun)(x_mid, y_middle)  # (m-1, d)

    h_col = h[:, jnp.newaxis]           # (m-1, 1) for broadcasting in residuals
    col_res = y[1:] - y[:-1] - h_col / 6 * (f[:-1] + f[1:] + 4 * f_middle)

    return col_res, (y_middle, f, f_middle)



# ------------------------------------------------------------------------------------------
# Functions to fit cubic splines
# ------------------------------------------------------------------------------------------
# The solution to the BVP is sought as a cubic spline.


def fit_cubic_spline_coeffs(y, yp, h):
    """ Fit the parameters of a cubic spline.

    Formulas for the coefficients are taken from scipy.interpolate.CubicSpline.

    Parameters
    ---------
    y : ndarray, shape (m, n)
        Solution values at the mesh nodes.
    yp : ndarray, shape (m, n)
        ODE system evaluated at the mesh nodes.
    h : ndarray, shape (m-1, )
        Increment between mesh nodes, that is, jnp.diff(x)

    Returns
    -------
    c : ndarray with shape (m-1, n, 4)
    """
    h = h[:, jnp.newaxis]
    slope = (y[1:] - y[:-1]) / h
    t = (yp[:-1] + yp[1:] - 2 * slope) / h

    c0 = y[:-1]
    c1 = yp[:-1]
    c2 = (slope - yp[:-1]) / h - t
    c3 = t / h

    return jnp.stack([c0, c1, c2, c3], axis=-1)


def eval_cubic_spline(x, coeffs, xs):
    """ Evaluate a cubic spline and its first derivative at ``xs``.

    Parameters
    ----------
    x : ndarray with shape (m, )
        Nodes of the spline.
    coeffs : ndarray with shape (m, n, 4)
        Coefficients of the spline, as returned by `fit_cubic_spline_coeffs`.
    xs : ndarray with shape (t, )
        Where to evaluate the spline.

    Returns
    -------
    c : ndarray with shape (t, n)
        Evaluation of the spline at ``xs``.
    dc : ndarray with shape (t, n)
        First derivative of the spline at ``xs``.
    """
    ind = jnp.digitize(xs, x) - 1  # determine the interval in x
    ind = jnp.clip(ind, 0, len(x) - 2)  # include the right endpoint
    c = coeffs[ind]  # use the relevant spline coefficients

    t = (xs - x[ind])[:, jnp.newaxis, jnp.newaxis]
    t_powers = jnp.power(t, jnp.arange(4))

    eval = jnp.sum(c * t_powers, axis=-1)
    deval = jnp.sum(c[..., 1:] * t_powers[..., :-1]
                    * jnp.array([[[1, 2, 3.]]]), axis=-1)

    return eval, deval


# ------------------------------------------------------------------------------------------
# Damped Newton method
# ------------------------------------------------------------------------------------------


class BacktrackLineSearch:
    def __init__(self, sigma=0.2, tau=0.5, n_trials=4, jit=False):
        """ Backtracking line search.

        Parameters
        ----------
        sigma : float
            Minimum relative improvement of the criterion function to accept the
            step (Armijo constant).
        tau : float
            Step size decrease factor for backtracking.
        n_trials : int
            Maximum number of backtracking steps, the minimum step is then tau ** n_trial.
        jit : bool
            Whether to jit-compile the optimization loop (as a jax.lax.while_loop)
        """
        def _run(cost_fnc, y, step, cost, init_extras):
            """ Iteratively reduces step size until cost is sufficiently decreased.

            c(y - alpha * step) < (1 - 2 * alpha * sigma) c(y)

            Parameters
            ----------
            cost_fnc : callabe s.t. cost, extras = cost_fnc(y)
                Cost function taking as input some `y`, and returning both some
                float `cost` and some `extra` arguments.
            y : ndarray, shape (m, n)
                Base point from which steps are taken.
            step : ndarray, shape (m, n)
                Direction for the updates.
            cost : float
                The cost function `cost_fnc` evaluated at the base point `y`.
            init_extras : tuple
                Intitialization of the extra parameters returned by the cost function,
                since we assume cost, extras = cost_fcn(y). Required for jax.lax.while_loop.

            Returns
            -------
            iterations : int
                Number of backtracking steps taken.
            y_new : ndarray, shape (m, n)
                Point that meets the backtracking critetion (hopefully)
            cost_new, extras = cost_fnc(y_new)
            """

            # Conditions for termination, either
            #   - the backtracking condition is fulfilled (cost sufficiently low)
            #   - the maximum number of steps is taken
            def keep_going(val):
                iteration, (_, cost_new, _) = val

                alpha = tau ** (iteration - 1)
                iters_low = iteration < n_trials + 1
                cost_high = cost_new >= (1 - 2 * alpha * sigma) * cost
                return iters_low & cost_high

            def backtrack_step(val):
                iteration, _ = val

                # Update solution value
                alpha = tau ** iteration
                y_new = y - alpha * step

                # Compute the new cost
                cost_new, extras = cost_fnc(y_new)

                return iteration + 1, (y_new, cost_new, extras)

            # Hard code the shape of the extra returns, this could be better
            m, n = y.shape
            val = 0, (jnp.zeros_like(y), jnp.array(jnp.inf), init_extras)

            if jit:
                val = jax.lax.while_loop(keep_going, backtrack_step, val)
            else:
                while keep_going(val):
                    val = backtrack_step(val)

            return val

        if jit:
            self.run = jax.jit(_run, static_argnums=0)
        else:
            self.run = _run


class BacktrackingNewton:
    def __init__(self, col_obj, get_jac, init_extras, bvp_tol, bc_tol, max_njev=4,
                 max_iter=8, jit=False):
        """ Simple Newton method with a backtracking line search.

        As advised in [1]_, an affine-invariant criterion function F = ||J^-1 r||^2
        is used, where J is the Jacobian matrix at the current iteration and r is
        the vector of collocation residuals (values of the system lhs).

        The method alters between full Newton iterations and fixed-Jacobian
        iterations. The Jacobian is recomputed if a full Newton step does not
        meet some backtracking criterion, otherwise the same Jacobian is reused.

        There are other tricks proposed in [1]_, but they are not used as they
        don't seem to improve anything significantly, and even break the
        convergence on some test problems I tried.

        Parameters
        ----------
        col_obj : callable
            Function computing collocation residuals, and some other extra
            parameters such that (res, extra) = col_obj(y, x, h)
        get_jac : callable
            Returns the Jacobian of the collocation objective w.r.t `y` and
            evaluated at the mesh points `x`.
        init_extras : callable
            Takes the same arguments as `col_obj` and returns initial values
            for the extra parameters returned by `col_obj`.
        bvp_tol : float
            Tolerance to which we want to solve the BVP.
        bc_tol : float
            Tolerance to which we want to satisfy the boundary conditions.
        max_njev : int
            Maximum allowed number of Jacobian evaluation and factorization, in
            other words, the maximum number of full Newton iterations. A small
             value is recommended in the literature.
        max_iter : int
            Maximum number of iterations, considering that some of them can be
            performed with the fixed Jacobian.
        jit : bool
            Whether to jit-compile the relevant loops.

        Returns
        -------
        y : ndarray, shape (m, n)
            Final iterate for the function values at the mesh nodes.
        res, extras = col_obj(y, x, h)

        References
        ----------
        .. [1]  U. Ascher, R. Mattheij and R. Russell "Numerical Solution of
           Boundary Value Problems for Ordinary Differential Equations"
        """

        def _loop(cond, body, val):
            if jit:
                return jax.lax.while_loop(cond, body, val)
            else:
                while cond(val):
                    val = body(val)
                return val

        line_search = BacktrackLineSearch(jit=jit)

        def solve(y, x, h):
            m, n = y.shape

            # Some initialization of the extra parameters returned by `col_obj` and
            # `backtrack_cost`. Required for jax.lax.do_loop. Must have consistent dims.
            colobj_extras = (jnp.zeros((m - 1, n)), jnp.zeros((m, n)), jnp.zeros((m - 1) * n))
            btrack_extras = (jnp.zeros(m * n), colobj_extras, jnp.zeros((m, n)))

            # We know that the solution residuals at the middle points of the mesh
            # are connected with collocation residuals  r_middle = 1.5 * col_res / h.
            # As our BVP solver tries to decrease relative residuals below a certain
            # tolerance, it seems reasonable to terminated Newton iterations by
            # comparison of r_middle / (1 + jnp.abs(f_middle)) with a certain threshold,
            # which we choose to be 1.5 orders lower than the BVP tolerance. We rewrite
            # the condition as col_res < tol_r * (1 + jnp.abs(f_middle)), then tol_r
            # should be computed as follows:
            tol_r = 2 / 3 * h[:, jnp.newaxis] * 5e-2 * bvp_tol

            def continue_newton(val):
                (iteration, njev, terminate_tol), _, _ = val
                return (iteration < max_iter) & (njev <= max_njev) & (~terminate_tol)

            # At each outer step the Jacobian is recomputed, then a number of inner Newton
            # steps are taken using the same Jacobian, as long as the full Newton step meets
            # the backtracking condition (i.e., the affine-invariant criterion). If in the
            # other hand the step size is reduced in order to meet the backtracking condition,
            # then the Jacobian is recomputed.
            def newton_step(val):
                (iteration, njev, _), (y, _), extras = val

                # Recompute Jacobian
                y_middle, _, _ = extras
                jac = get_jac(x, h, y, y_middle)

                # The cost function is the affine-invariant criterion F = ||J^-1 r||^2, where
                # J is the Jacobian and r are the collocation residuals.
                def backtrack_cost(y):
                    res, extras = col_obj(y, x, h)  # compute collocation residuals

                    # Compute new step
                    step = BABD_solve(jac, res).reshape(y.shape)
                    cost = jnp.sum(step ** 2)
                    return cost, (res, extras, step)

                # Continue taking steps with the same Jacobian as long as the step size does
                # not need to be decreased to meet the affine-invariant criterion.
                def do_step(val):
                    (iteration, _, _), y_step_cost, _ = val
                    n_trials, (y, cost, extras) = line_search.run(backtrack_cost,
                                                                  *y_step_cost, btrack_extras)

                    res, extras, step = extras
                    _, _, f_middle = extras

                    col_res_cond = jnp.all(jnp.abs(res[:-n]) < tol_r * (1 + jnp.abs(f_middle)))
                    bc_res_cond = jnp.all(jnp.abs(res[-n:]) < bc_tol)
                    terminate_tol = col_res_cond & bc_res_cond

                    return (iteration + 1, n_trials, terminate_tol), (y, step, cost), (res, extras)

                def continue_stepping(val):
                    (iteration, n_trials, terminate_tol), _, _ = val
                    return (iteration < max_iter) & (n_trials <= 1) & (~terminate_tol)

                # Init step and cost
                cost, (_, _, step) = backtrack_cost(y)

                # Inner fixed-Jacobian loop
                res, extras = jnp.zeros(m * n), colobj_extras
                val = ((iteration, 0, jnp.array(False)), (y, step, cost), (res, extras))
                val = _loop(continue_stepping, do_step, val)

                (iteration, _, terminate_tol), (y, _, _), (res, extras) = val
                return (iteration, njev + 1, terminate_tol), (y, res), extras

            # Outer loop where the Jacobian is recomputed
            extras = init_extras(y, x, h)
            val = ((0, 0, jnp.array(False)), (y, jnp.zeros(m*n)), extras)
            (iterations, njevs, _), (y, res), extras = _loop(continue_newton, newton_step, val)

            return y, (res, extras)

        self.solve = solve


# ------------------------------------------------------------------------------------------
# Functions to estimate the relative residuals of the approximate solution
# ------------------------------------------------------------------------------------------
# 5-point Lobatto quadrature provides much more accurate estimatations of the relative
# residuals compared to Simpson's rule (3-point Lobatto quadrature), however it requires
# ~2*m additional evaluations of the ODE function, where m is the number of mesh points.


def estimate_rms_residuals_5lobatto(fun, coeffs, x, h, r_middle, f_middle):
    """Estimate rms values of collocation residuals using 5-point Lobatto quadrature.

    The residuals are defined as the difference between the derivatives of
    our solution and rhs of the ODE system. We use relative residuals, i.e.,
    normalized by 1 + jnp.abs(f). RMS values are computed as sqrt from the
    normalized integrals of the squared relative residuals over each interval.
    Integrals are estimated using 5-point Lobatto quadrature [1]_, we use the
    fact that residuals at the mesh nodes are identically zero.

    In [2] they don't normalize integrals by interval lengths, which gives
    a higher rate of convergence of the residuals by the factor of h**0.5.
    I chose to do such normalization for an ease of interpretation of return
    values as RMS estimates.

    Parameters
    ----------
    fun : callable
        ODE function.
    coeffs : ndarray, shape (m-1, n, 4)
        Coefficients of a cubic spline parametrizing the approximate solution.
    x : ndarray, shape (m, n)
        Nodes of the mesh.
    h : ndarray, shape (m-1, )
        Interval between the mesh nodes (i.e., np.diff(x)).
    r_middle : ndarray, shape (m-1, d)
        Residuals at the mid-point of each mesh interval.
    f_middle : ndarray, shape (m-1, d)
        Evaluation of the ODE function at the mid-point of each mesh interval.

    Returns
    -------
    rms_res : ndarray, shape (m-1,)
        Estimated rms values of the relative residuals over each mesh interval.

    References
    ----------
    .. [1] http://mathworld.wolfram.com/LobattoQuadrature.html
    .. [2] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
       Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
       Number 3, pp. 299-316, 2001.
    """
    x_middle = x[:-1] + 0.5 * h
    s = 0.5 * h * (3 / 7) ** 0.5

    xx = jnp.r_[x_middle + s, x_middle - s]
    y, yp = eval_cubic_spline(x, coeffs, xx)
    f = jax.vmap(fun)(xx, y)
    r = yp - f

    r /= 1 + jnp.abs(f)
    r_middle /= 1 + jnp.abs(f_middle)

    r = jnp.sum(r ** 2, axis=-1)
    r_middle = jnp.sum(r_middle ** 2, axis=-1)

    r_sum = r[:s.shape[0]] + r[s.shape[0]:]
    return (0.5 * (32 / 45 * r_middle + 49 / 90 * r_sum)) ** 0.5


def estimate_rms_residuals_simpson(r_middle, f_middle):
    """Estimate rms values of collocation residuals using Simpsons rule.

    The residuals are defined as the difference between the derivatives of
    our solution and rhs of the ODE system. We use relative residuals, i.e.,
    normalized by 1 + jnp.abs(f). RMS values are computed as sqrt from the
    normalized integrals of the squared relative residuals over each interval.

    Since this is precisely what the collocation objective solves for,
    using 5-point Lobatto quadrature gives significatly more accurate
    estimations of the residuals. However, 5-point Lobatto quadrature
    requires ~ 2*m more evaluations of the ODE function.

    Parameters
    ----------
    r_middle : ndarray, shape (m-1, d)
        Residuals at the mid-point of each mesh interval.
    f_middle : ndarray, shape (m-1, d)
        Evaluation of the ODE function at the mid-point of each mesh interval.

    Returns
    -------
    rms_res : ndarray, shape (m - 1,)
        Estimated rms values of the relative residuals over each interval.
    """
    # We use that residuals at the mesh nodes are identically zero.
    r_middle /= 1 + jnp.abs(f_middle)
    r_middle = jnp.sum(r_middle ** 2, axis=-1)
    return ((2 / 3.) * r_middle) ** 0.5


# ------------------------------------------------------------------------------------------
# Functions implementing the mesh selection strategy
# ------------------------------------------------------------------------------------------
# In scipy's bvp_solver implementation, the mesh is iteratively refined using a local
# criterion based on the estimated relative residuals of the approximate solution. However,
# JAX has poor support of arrays with dynamic size, and an efficient implementation of the
# refined strategy that is jittable and vmappable is non-trivial. We therefore use a global
# mesh selection strategy where node points are equidistributed based w.r.t. some picewise
# constant monitor function, without adding or removing any mesh nodes. This requires
# that are relatively fine initial mesh is used, and for the ODE function to be
# sufficiently smooth. The authors of MATLAB bvp4c [1] find global mesh selection strategies
# to be inferior to local strategies. Similarly, [2] Chapter 9.5 further argue that mesh
# refinement tends to be beneficial since a coarse initial mesh can be used, thus resulting
# in substantially more computationally effective solutions.
# [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual Control and the Maltab
#     PSE", ACM Trans. Math. Softw., Vol. 27, Number 3, pp. 299-316, 2001.
# [2] U. Ascher, R. Mattheij and R. Russell "Numerical Solution of  Boundary Value Problems
#     for Ordinary Differential Equations".


def monitor_fourth_derivative(x, h, coeffs):
    """ Approximates the fourth derivative of the BVP solution.

    The BVP solution is parametrized by a cubic spline. [1]_ Chapter 9.3.1 argues that
    higher order derivatives are more robust monitor function that relative residuals.
    According to the description of [1]_, I am unsure whether the third or fourth
    derivative should be used for the monitor function. According to preliminary tests
    I find the fourth derivative to result in solutions with lower estimated residuals.

    As described in [1]_, the fourth derivative of the BVP solution is approximated by
    fitting a picewise linear function v(x) of the third derivative f the cubic spline
    at the subinterval midpoints. The monitor function is then the  derivative v'(x).

    Parameters
    ----------
    x : ndarray, shape (m, 4)
        Mesh nodes.
    h : ndarray, shape (m-1, 4)
        Interval between the mesh nodes, that is, np.diff(x).
    coeffs : ndarray, shape (m-1, n, 4)
        Coefficients of the Cubic spline.

    Returns
    -------
    x_monitor: ndarray, shape (m+1, 4)
        Points of the mesh at which the monitor function is evaluated.
    monitor : ndarray, shape (m, )
        Evaluation of picewise monitor function at x_monitor.

    References
    ----------
    .. [1]  U. Ascher, R. Mattheij and R. Russell "Numerical Solution of
       Boundary Value Problems for Ordinary Differential Equations"
    """
    # Define the monitor function at the extreme points of the mesh and at the
    # midpoint of each subinterval
    midpoints = x[:-1] + h / 2.
    x_monitor = jnp.r_[x[0], midpoints, x[-1]]  # (m+1, )

    # Third derivative of the cubic spline
    third_der = coeffs[..., -1] * 6

    # [1] does not specify the value that v(x) should have at the extreme of the
    # mesh, so we assume that it is zero (which in general will not be the case).
    pad_zeros = jnp.zeros((1, third_der.shape[-1]))
    diffs = jnp.diff(jnp.r_[pad_zeros, third_der, pad_zeros], axis=0)  # (m, n)
    v_prime = diffs / jnp.diff(x_monitor)[:, None]

    # Evaluate monitor function according to (9.17). I believe that the max norm
    # should be used (refer to page 363).
    monitor = jnp.max(jnp.abs(v_prime), axis=-1) ** (1 / 4.)
    return x_monitor, monitor


def monitor_mazzia(x, coeffs, weight_infty=1., weight_1=1.):
    """ Combines the L_infinity norm of the BVP solution and the L1 of the derivatives.

    Monitor function described in [1]_ (p. 562), which is a linear combination of
    the L_infinity norm of the approximate solution and the L1 norm of the
    derivatives of the approximate solution.

    In my experience, `monitor_fourth_derivative` tends to work significantly better,
    however here we implement a very simplified version of the monitor function
    described in [1]_.

    Parameters
    ----------
    x : ndarray, shape (m, 4)
        Mesh nodes.
    coeffs : ndarray, shape (m-1, n, 4)
        Coefficients of a cubic spline parametrizing the approximate solution.
    weight_infty : float
        Weight given to the infinity norm of the approximate solution.
    weight_1 : float
        Weight given to the L1 norm of the derivatives of the approximate solution.

    Returns
    -------
    x_monitor: ndarray, shape (m+1, 4)
        Points of the mesh at which the monitor function is evaluated.
    monitor : ndarray, shape (m, )
        Evaluation of the monitor function at `x_monitor`.

    References
    ----------
    .. [1]  F. Mazzia, Mesh selection strategies of the code TOM for Boundary
       Value Problems, 2022.
    """
    y, yp = eval_cubic_spline(x, coeffs, x)

    norm_yp = jnp.sum(jnp.abs(yp), axis=-1)
    m_yp = norm_yp[:-1] + norm_yp[1:]

    norm_y = jnp.max(jnp.abs(y), axis=-1)
    m_y = jnp.abs(norm_y[1:] - norm_y[:-1])

    monitor = weight_infty * m_y + weight_1 * m_yp
    return x, monitor


def equidistribute_mesh(x_monitor, monitor, n_points):
    """ Solve for a new mesh by equidistributing a picewise constant monitor function.

    A picewise constant monitor function is used for simplicity and computational
    efficiency, since integration and reverse interpolation is then trivial [1]_.

    Parameters
    ----------
    x_monitor : ndarray, shape (m,)
        Mesh nodes at which the monitor function is evaluated.
    monitor: ndarray, shape (m-1,)
        Picewise constant monitor function, evaluated at x_monitor.
    n_points: int
        Number of mesh points in the new mesh.

    Returns
    -------
    x_new : ndarray, shape (n_points, )
        New mesh nodes.

    References
    ----------
    .. [1]  U. Ascher, R. Mattheij and R. Russell "Numerical Solution of
       Boundary Value Problems for Ordinary Differential Equations"
    """
    # Integrate the monitor function from x[0] to x[-1]
    integral = jnp.r_[jnp.array([0]), jnp.cumsum(jnp.diff(x_monitor) * monitor)]  # (9.18)

    # Equidistribute across the N intervals
    intervals = jnp.linspace(0, integral[-1], n_points)  # (9.19b)

    # Inverse interpolation (9.19) for a picewise constant function
    boxes = jnp.clip(jnp.digitize(intervals, integral) - 1, 0, n_points - 1)
    increment_needed = intervals - integral[boxes]
    new_x = x_monitor[boxes] + increment_needed / monitor[boxes]
    return new_x


def print_iteration_header():
    print("{:^15}{:^15}{:^15}{:^15}{:^15}".format(
        "Iteration", "Max residual", "Max BC residual", "Total nodes",
        "Nodes added"))


def print_iteration_progress(iteration, residual, bc_residual, total_nodes,
                             nodes_added):
    print("{:^15}{:^15.2e}{:^15.2e}{:^15}{:^15}".format(
        iteration, residual, bc_residual, total_nodes, nodes_added))


def solve_bvp(fun, bc, x, y, tol=1e-3, bc_tol=None, max_iterations=10,
              min_improv=0.05, eval_lobatto=True, verbose=False, jit=True):
    """Solve a boundary value problem for a system of ODEs.

    This function numerically solves a first order system of ODEs subject to
    two-point boundary conditions::

        dy / dx = f(x, y, p), a <= x <= b
        bc(y(a), y(b)) = 0

    Here x is a 1-D independent variable, y(x) is an N-D vector-valued function.
    For the problem to be determined, there must be n boundary conditions, i.e.,
    bc must be an n-D function.

    Parameters
    ----------
    fun : callable
        Right-hand side of the system. The calling signature is ``fun(x, y)``.
        All arguments are ndarray: ``x`` with shape (,), ``y`` with shape (d,).
        The return value must be an array with shape (d,).
    bc : callable
        Function evaluating residuals of the boundary conditions. The calling
        signature is ``bc(ya, yb)``. All arguments are ndarray: ``ya`` and
        ``yb`` with shape (n,). The return value must be an array with shape (n,).
    x : array_like, shape (m,)
        Initial mesh. Must be a strictly increasing sequence of real numbers
        with ``x[0]=a`` and ``x[-1]=b``. The initial mesh should be relatively
        fine since the initial mesh is not iteratively refined.
    y : array_like, shape (m, d)
        Initial guess for the function values at the mesh nodes.
    tol : float, optional
        Desired tolerance of the solution. If we define ``r = y' - f(x, y)``,
        where y is the found solution, then the solver tries to achieve on each
        mesh interval ``norm(r / (1 + abs(f)) < tol``, where ``norm`` is
        estimated in a root mean squared sense (using a numerical quadrature
        formula). Default is 1e-3.
    bc_tol : float, optional
        Desired absolute tolerance for the boundary condition residuals: `bc`
        value should satisfy ``abs(bc) < bc_tol`` component-wise.
        Equals to `tol` by default.
    max_iterations : int, optional
        Maximum number of iterations of the BVP solver.
    min_improv : float, optional
        Early stopping condition. Requires that the relative change in the
        maximum relative residual between two consecutive iterations be > than
        `min_improv`, otherwise the solution is returned. Since we do not
        refine the initial mesh, solutions often converge if the prescribed
        tolerance cannot be achieved with the given number of mesh nodes.
    eval_lobatto : bool, optional
        Whether to estimate the relative residuals using 5-point Lobatto
        quadrature as opposed to 3-point Lobatto quadrature. The former is
        significantly more accuracy, but requires ~2*m additional evaluations
        of `fun` per iteration.
    verbose : bool, optional
        Prints some helpful information. Cannot be used together with jit=True.
    jit : bool, optional
        Whether to jit compile the whole iteration procedure.

    Returns
    -------
    x : ndarray, shape (m,)
        Nodes of the final mesh.
    y : ndarray, shape (m, d)
        Solution values at the mesh nodes.
    iteration : int
        Number of iterations performed.
    max_rms_res : float
        Maximum estimated relative residual of the solution.
    max_bc_res : float
        Maximum residual of the boundary conditions.
    success : bool
        True if the algorithm converged to the desired accuracy (``status=0``).

    Notes
    -----
    This function implements a 4th order collocation algorithm with the
    control of residuals similar to [1]_. A collocation system is solved
    by a damped Newton method with an affine-invariant criterion function as
    described in [3]_. Note that in contrast to [1]_, we do not iteratively
    refine the mesh using a local criterion but rather iteratively
    equidistribute the mesh points based on some monitor function, since JAX
    has poor support for dynamic arrays.

    Note that in [1]_  integral residuals are defined without normalization
    by interval lengths. So, their definition is different by a multiplier of
    h**0.5 (h is an interval length) from the definition used here.

    References
    ----------
    .. [1] J. Kierzenka, L. F. Shampine, "A BVP Solver Based on Residual
           Control and the Maltab PSE", ACM Trans. Math. Softw., Vol. 27,
           Number 3, pp. 299-316, 2001.
    .. [2] L.F. Shampine, P. H. Muir and H. Xu, "A User-Friendly Fortran BVP
           Solver".
    .. [3] U. Ascher, R. Mattheij and R. Russell "Numerical Solution of
           Boundary Value Problems for Ordinary Differential Equations".
    """
    if bc_tol is None:
        bc_tol = tol

    # Concatenation of the collocation residuals and the boundary condition residuals.
    def col_obj(y, x, h):
        # Evaluate the collocation residuals
        y = y.reshape(x.shape[0], -1)
        col_res, (y_middle, f, f_middle) = collocation_fun(fun, y, x, h)

        # Evaluate the function residuals
        bc_res = bc(y[0], y[-1])

        res = jnp.hstack((col_res.ravel(), bc_res))
        extras = (y_middle, f, f_middle.ravel())
        return res, extras

    # Initial values for the extra parameters returned by `col_obj`.
    # Called once at each call of `newton_solver.solve`.
    def init_col_obj_extras(y, x, h):
        m, n = y.shape
        # As it currently stands, it is needed to compute y_middle in order to construct
        # the Jacobian of the collocation system. This is unfortunate since it amounts
        # to an additional evaluation of the ODE function, resulting in higher time
        # to jit compile the bvp solver.
        y_middle, _ = compute_ymiddle(fun, y, x, h)
        extras = (y_middle, jnp.zeros((m, n)), jnp.zeros((m - 1) * n))
        return extras

    get_jac = prepare_jac(fun, bc)
    newton_solver = BacktrackingNewton(col_obj, get_jac, init_col_obj_extras,
                                       tol, bc_tol, jit=jit)

    if verbose and not jit:
        print_iteration_header()

    def loop_continue(vals):
        (iteration, max_rms_res, max_bc_res, rms_change), _ = vals
        rms_cond = (max_rms_res > tol) & (rms_change > min_improv)
        return (iteration < max_iterations) & (rms_cond | (max_bc_res > bc_tol))

    def loop_body(vals):
        (iteration, prev_max_rms_res, _, _), (_, _, x, y) = vals

        h = jnp.diff(x)
        y, (res, (_, f, f_middle)) = newton_solver.solve(y, x, h)

        # Re-use the ODE function evaluations form inside the Newton solver
        f_middle = f_middle.reshape(-1, f.shape[-1])
        res = res.reshape(-1, f.shape[-1])
        bc_res = res[-1]
        col_res = res[:-1]

        # Fit a cubic spline as an approximate solution.
        spline_coeffs = fit_cubic_spline_coeffs(y, f, h)

        # Compute the residual at the mid-points of each interval.
        # This relation is not trivial, but can be verified.
        r_middle = 1.5 * col_res / h[:, jnp.newaxis]

        # Estimate the relative residuals of the approximate solution.
        if eval_lobatto:
            rms_res = estimate_rms_residuals_5lobatto(fun, spline_coeffs,
                                                      x, h, r_middle, f_middle)
        else:
            rms_res = estimate_rms_residuals_simpson(r_middle, f_middle)
        max_rms_res = jnp.max(rms_res)

        # Compute the ratio of improvement (condition for early stopping)
        rms_change = jnp.abs(max_rms_res - prev_max_rms_res) \
                     / jnp.minimum(max_rms_res, prev_max_rms_res)

        # Evaluate if the boundary condition is met
        max_bc_res = jnp.max(abs(bc_res))

        # Equdistribute the mesh according to the monitor function
        x_monitor, monitor = monitor_fourth_derivative(x, h, spline_coeffs)
        x_new = equidistribute_mesh(x_monitor, monitor, x.shape[0])
        y_new, _ = eval_cubic_spline(x, spline_coeffs, x_new)

        if verbose and not jit:
            print_iteration_progress(iteration, max_rms_res, max_bc_res, x.shape[0], 0)

        # Need to return both (x, y) and (x_new, y_new) in case the evaluated solution
        # meets the stopping criteria. Otherwise, the reported residuals would be inaccurate.
        return (iteration + 1, max_rms_res, max_bc_res, rms_change), (x, y, x_new, y_new)

    vals = ((0, jnp.array(jnp.inf), jnp.array(jnp.inf), jnp.array(1.)), (x, y, x, y))

    if jit:
        vals = jax.lax.while_loop(loop_continue, loop_body, vals)
    else:
        while loop_continue(vals):
            vals = loop_body(vals)

    (iteration, max_rms_res, max_bc_res, _), (x, y, _, _) = vals
    return x, y, (iteration, max_rms_res, max_bc_res)


if __name__ == "__main__":
    def fun(x, y):
        return jnp.array([y[1], -jnp.exp(y[0])])

    def bc(ya, yb):
        return jnp.array([ya[0], yb[0]])

    N = 100
    x = jnp.linspace(0, 1, N)
    y_a = jnp.zeros((x.size, 2))
    y_b = jnp.ones((x.size, 2)) * 3.

    solve = jax.jit(jax.vmap(lambda x, y: solve_bvp(fun, bc, x, y, tol=1e-5, jit=True)))

    xx = jnp.r_[x[jnp.newaxis], x[jnp.newaxis]]
    yy = jnp.r_[y_a[jnp.newaxis], y_b[jnp.newaxis]]

    _, _, (_, max_rms_res, max_bc_res) = solve(xx, yy)

    print('Max relative residuals: ', max_rms_res)
    print('Boundary condition residuals:', max_bc_res)