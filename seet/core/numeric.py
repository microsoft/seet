"""numeric.py.

User-defined package for numerical methods.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import itertools
import torch


DTYPE = torch.tensor(1.1).dtype

T0 = torch.tensor(0.0)
T1 = torch.tensor(1.0)
TPI = torch.tensor(3.14159265358979323846)
# Combined refractive index of cornea and aqueous humor
TETA = torch.tensor(1.375)

if (DTYPE == torch.float32):
    EPS = 1.192093e-07
    DPLACES = 5
else:
    EPS = 2.220446e-16
    DPLACES = 10

TEPS = torch.tensor(EPS)
UNIT_CUBE_ROOT = -0.5 + 0.8660254037844386j
NUMPY_DTYPE = T0.numpy().dtype


def deg_to_rad(val):
    # This function is meant to duplicate torch.deg2rad,
    # which has a bug when converting between float types.
    return val * TPI / 180


def rad_to_deg(val):
    # This function is meant to duplicate torch.rad2deg,
    # which has a bug when converting between float types.
    return val * 180 / TPI


def stack_tensors(tuple_of_rows):
    return torch.vstack(
        tuple(torch.hstack(tuple(row)) for row in tuple_of_rows)
    )


def compute_numeric_jacobian_from_tensors(x, fun, delta=TEPS * 100, device=None):
    """compute_numeric_jacobian_from_tensors.

    This function numerically compute the Jacobian of fun with respect to x.
    It is used for testing purposes only.

    Args:
        x (torch.Tensor): input to function fun.

        fun (callable): function taking torch Tensor x as input and returning a
        torch Tensor y.

        device (torch.device or str, optional): If specified, the computation is
        performed on this device. If None, the device of the input tensor x is used.

    Returns:
        torch.Tensor: torch.Tensor with shape of (y, x) corresponding to the
        Jacobian of fun with respect to x (y is the output of fun(x)).
    """
    if device is not None:
        target_device = torch.device(device) if isinstance(device, str) else device
        x = x.to(target_device)
    else:
        target_device = x.device
    
    if isinstance(delta, torch.Tensor):
        delta = delta.to(target_device)
    else:
        delta = torch.tensor(delta, device=target_device, dtype=x.dtype)

    y = fun(x)
    y_shape = list(y.shape)
    x_shape = list(x.shape)
    dy_dx = torch.empty(y_shape + x_shape, dtype=y.dtype, device=target_device, requires_grad=False)

    if dy_dx.ndim == 0:
        return (fun(x + delta) - fun(x)) / delta

    multi_range = [range(i) for i in x_shape]
    for multi_index in itertools.product(*multi_range):
        x_delta = x.clone()
        x_delta[multi_index] = x_delta[multi_index] + delta
        y_delta = fun(x_delta)
        dy_dx[..., multi_index] = \
            (y_delta - y).reshape(y_shape + [1, ]) / delta

    return dy_dx


def alt_compute_auto_jacobian_from_tensors(y, x, create_graph=False):
    """alt_compute_auto_jacobian_from_tensors.

    Generate the derivative of the tensor y with respect to the tensor x using
    PyTorch Autograd. Note that y and x must be on the same device.

    Parameters
    ----------
    y : torch.Tensor
        Torch tensor with shape y_shape = (Y1, Y2, ..., YM)
    x : torch.Tensor
        Torch tensor with shape x_shape = (X1, X2, ..., XN)
    create_graph : bool, default=False
        If true, create computation graph of result, so that higher order
        derivatives can be computed.

    Returns
    -------
    torch.Tensor
        A torch tensor with shape y_shape + x_shape = (Y1, Y2, ..., YM, X1,
        X2, ..., XN), such that the derivative of y[i1, i2, ..., iM] with
        respect to x[j1, j2, ..., jN] is out[i1, i2, ..., iM, j1, j2, ..., jN].
    """
    assert x.device == y.device
    device = x.device

    M = y.numel()
    y_shape = list(y.shape)
    x_shape = list(x.shape)

    # If Y is (b1 x ... x bM) and X is (a1 x ... x aN), then
    # dY_dX is (b1 x ... x bM x a1 x ... aN)
    basis = torch.eye(M, dtype=y.dtype, device=device).reshape([M, ] + y_shape)
    result = ()
    for i in range(M):
        dy_dx_i = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=basis[i, ...],
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True)[0]

        if dy_dx_i is None:
            # x is not in the computational graph of y, so dy_dx_i is zero.
            dy_dx_i = torch.zeros_like(x, device=device)

        if dy_dx_i.ndim == 0:
            # Result is a scalar. Fix it so concatenation works.
            dy_dx_i = dy_dx_i.view(1)

        result = result + (dy_dx_i,)

    return torch.cat(result, dim=0).reshape(y_shape + x_shape)


def compute_auto_jacobian_from_tensors(y, x, create_graph=False):
    """compute_auto_jacobian_from_tensors.

    Generate the derivative of the tensor y with respect to the tensor x using
    PyTorch Autograd. Note that y and x must be on the same device.

    Parameters
    ----------
    y : torch.Tensor
        Torch tensor with shape y_shape = (Y1, Y2, ..., YM)
    x : torch.Tensor
        Torch tensor with shape x_shape = (X1, X2, ..., XN)
    create_graph : bool, default=False
        If true, create computation graph of result, so that higher order
        derivatives can be computed.

    Returns
    -------
    torch.Tensor
        A torch tensor with shape y_shape + x_shape = (Y1, Y2, ..., YM, X1,
        X2, ..., XN), such that the derivative of y[i1, i2, ..., iM] with
        respect to x[j1, j2, ..., jN] is out[i1, i2, ..., iM, j1, j2, ..., jN].
    """
    assert x.device == y.device
    device = x.device
    
    M = y.numel()
    y_shape = list(y.shape)
    x_shape = list(x.shape)

    # If Y is (b1 x ... x bM) and X is (a1 x ... x aN), then
    # dY_dX is (b1 x ... x bM x a1 x ... aN)
    basis = torch.eye(M, dtype=y.dtype, device=device).reshape([M, ] + y_shape)
    dy_dx = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=(basis,),
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
        is_grads_batched=True)

    if dy_dx is None:
        # x is not used in the computation of y. Derivative must be zero.
        return torch.zeros(y_shape + x_shape, dtype=y.dtype, device=device)

    return dy_dx[0].view(y_shape + x_shape)


def poly_solver(p, force_real=True, tol=TEPS * 10) -> torch.Tensor:
    """poly_solver.

    Solves p[0] * (x ** 0) + ... + p[N-1] * (x ** (N - 1)) in x.
    Returns a list with up to N-1 roots in arbitrary order.

    Args:
        p (torch.Tensor): (p[0], p[1], ..., p[N-1]) torch.Tensor holding N
        coefficients of (N-1)-degree polynomial
        p[0] + p[1]*x + ... + p[N-1]*x^N-1.

        force_real (bool, optional): if True, returns only the real roots.
        Defaults to True.

    Returns:
        torch.Tensor: (x[0], x[1], ..., x[N-2]) torch.Tensor holding N-1 roots
        of (N-1)-degree polynomial, or up to N-1 real roots if force_real is
        True.
    """
    # Drop coefficients to ensure that the lead coefficient is not zero.
    while p[-1] == 0:
        p = p[:-1]
    p = p / p[-1]  # Division by lead non-zero coefficient preserves roots.

    # Rather than directly computing the roots of the polynomial using, e.g.,
    # numpy.roots, we create the polynomial companion matrix and compute its
    # eigenvalues using torch.linalg.eigvals (requires pytorch >= 1.9.0). This
    # allows for correct propagation of derivatives.
    N = len(p) - 1  # Degree of input polynomial.
    if N == 1:
        # Linear. Easy.
        roots = torch.stack((-p[0] / p[1],))

    elif N == 2:
        # Quadratic. Let's not use eigvals, as its derivative is finicky.
        delta = p[1]**2 - 4 * p[0]
        if delta < 0:
            factor = 0.0 + 1.0j
        else:
            factor = 1.0

        sqrt_delta = factor * torch.sqrt(torch.abs(delta))
        # Roots are (-b +/- sqrt(delta))/(2a)

        roots = \
            torch.stack(((-p[1] - sqrt_delta) / 2, (-p[1] + sqrt_delta) / 2))

    elif N == 3:
        # Cubic. Also allows for closed-form solution, see
        # https://en.wikipedia.org/wiki/Cubic_equation
        delta_0 = p[2]**2 - 3 * p[1]
        delta_1 = 2 * p[2]**3 - 9 * p[2] * p[1] + 27 * p[0]
        aux = delta_1**2 - 4 * delta_0**3
        if aux < 0:
            factor = 0.0 + 1.0j
            aux = -aux
        else:
            factor = 1.0

        sqrt_2 = factor * (aux**0.5)
        C_plus = ((delta_1 + sqrt_2) / 2)**(1.0 / 3)
        C_minus = ((delta_1 - sqrt_2) / 2)**(1.0 / 3)

        if abs(C_plus) > abs(C_minus):
            C = C_plus
        else:
            C = C_minus

        result = ()
        for k in range(3):
            epsilon_k = UNIT_CUBE_ROOT**k
            x_k = -(p[2] + epsilon_k * C + delta_0 / epsilon_k / C) / 3
            if torch.abs(x_k.imag) < tol:
                x_k = x_k.real
            result = result + (x_k,)

        roots = torch.stack(result)

    else:
        bottom = torch.eye(N - 1)
        top = torch.zeros((1, N - 1))
        companion = \
            torch.hstack(
                (torch.vstack((top, bottom)), -p[:-1].view(N, 1))
            )

        roots = torch.linalg.eigvals(companion)

    if force_real:
        roots = torch.real(roots[torch.isreal(roots)])

    return roots
