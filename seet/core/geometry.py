"""geometry.py.

User-defined package for computer vision geometry.
"""


__author__ = "Chris Aholt (chaholt@microsoft.com)"


from seet.core import numeric
import torch


def rotation_around_x(angle_deg):
    """rotation_around_x.

    Creates a 3x3 matrix that acts as a rotation around the x axis by a
    specified angle. Negative angles means rotation up, positive angles, down.

    A 90 degree rotation around x sends the positive y axis to the positive z
    axis and the positive z axis to the negative y axis.

    Args:
        angle_degrees (float type): rotation angle in degrees.

    Returns:
        torch.Tensor: 3 x 3 matrix acting as rotation around x axis.
    """
    theta_rad = numeric.deg_to_rad(angle_deg)
    cos_theta = torch.cos(theta_rad)
    sin_theta = torch.sin(theta_rad)
    return numeric.stack_tensors((
        (numeric.T1, numeric.T0, numeric.T0),
        (numeric.T0, cos_theta, -sin_theta),
        (numeric.T0, sin_theta, cos_theta)
    ))


def rotation_around_y(angle_degrees):
    theta_radians = numeric.deg_to_rad(angle_degrees)
    cos_theta = torch.cos(theta_radians)
    sin_theta = torch.sin(theta_radians)
    return numeric.stack_tensors((
        (cos_theta, numeric.T0, sin_theta),
        (numeric.T0, numeric.T1, numeric.T0),
        (-sin_theta, numeric.T0, cos_theta)
    ))


def rotation_around_z(angle_degrees):
    theta_radians = numeric.deg_to_rad(angle_degrees)
    cos_theta = torch.cos(theta_radians)
    sin_theta = torch.sin(theta_radians)
    return numeric.stack_tensors((
        (cos_theta, -sin_theta, numeric.T0),
        (sin_theta, cos_theta, numeric.T0),
        (numeric.T0, numeric.T0, numeric.T1)
    ))


def _translation(i, t):
    """_translation.

    Generic method to create translation vectors.

    Args:
        i (int): index of direction of the desired translation. 0 means
        translation in x axis, 1 means translation in y axis, 2 means
        translation in z axis.

        t (float or torch.float): magnitude of translation.

    Returns:
        torch.Tensor: (3,) tensor corresponding to a translation along "x",
        "y", or "z" by amount t.
    """
    if not torch.is_tensor(t):
        t = torch.tensor(t)

    stack = [t if j == i else numeric.T0 for j in range(3)]

    return torch.stack(stack)


def translation_in_x(x):
    """translation_in_x.

    Create a (3,) torch tensor corresponding to a translation in x direction by
    the input amount

    Args:
        x (float or torch.Tensor): magnitude of translation in x.

    Returns:
        torch.Tensor: (3,) tensor.
    """
    return _translation(0, x)


def translation_in_y(y):
    return _translation(1, y)


def translation_in_z(z):
    return _translation(2, z)


def get_yaw_pitch_angles_deg(x):
    """get_yaw_pitch_angles_deg.

    Get the pitch and yaw angles corresponding to a unit vector. Definition of
    pitch and yaw from https://en.wikipedia.org/wiki/Aircraft_principal_axes

    Args:
        x (3 x 1 vector): Input 3 x 1 vector.
    """
    yaw_rad = torch.arctan(-x[0] / x[2])
    pitch_rad = torch.arcsin(x[1])

    return torch.stack(
        (numeric.rad_to_deg(yaw_rad), numeric.rad_to_deg(pitch_rad))
    )


def dehomogenize(points):
    """dehomogenize.

    Appends the points array divided by its last row."""
    return points[:-1, ...] / points[-1, ...]


def homogenize(points):
    """homogenize.

    Appends a row of 1s to the points array."""

    # Special case of a single point when only one dimension.
    ones_size = (1, ) if len(points.size()) == 1 else (1, points.size()[1])
    ones_row = torch.ones(ones_size)
    return torch.cat((points, ones_row), dim=0)


def normalize(points):
    """normalize.

    Unitizes each column."""
    return points / torch.linalg.norm(points, dim=0, keepdim=True)


def expand_to_rotation_from_x_y_axes(
        x_axis=None,
        y_axis=None,
        z_axis=None,
        normalized=False):
    """
    Create a rotation matrix that rotates the identity matrix to a matrix
    (x_axis, y_axis, z_axis). Only two out of the first three arguments are
    provided, in which case the missing argument is computed as the cross-
    product of the other two, multiplied by -1 if needed.

    Args:
        x_axis (torch.tensor): (3, 1) or (3,) torch tensor corresponding to
        first colum of desired rotation matrix. If normalized is true, it is
        assumed that torch.linalg.norm(x_axis) is equal to 1.

        y_axis (torch.tensor): (3, 1) or (3,) torch tensor corresponding to
        second colum of desired rotation matrix. If normalized is true, it is
        assumed that torch.linalg.norm(x_axis) is equal to 1.

        z_axis (torch.tensor): (3, 1) or (3,) torch tensor corresponding to
        third colum of desired rotation matrix. If normalized is true, it is
        assumed that torch.linalg.norm(x_axis) is equal to 1.

        normalized (boolean): If true, other inputs are assumed to be
        normalized.
    """

    if x_axis is None:
        if y_axis is None or z_axis is None:
            # This is not a valid input.
            return torch.eye(3)

        x_axis = torch.cross(y_axis, z_axis)
    elif y_axis is None:
        if z_axis is None:
            # This is not a valid input.
            return torch.eye(3)

        y_axis = torch.cross(z_axis, x_axis)
    else:
        z_axis = torch.cross(x_axis, y_axis)

    if normalized is False:
        x_axis = x_axis / torch.linalg.norm(x_axis)
        y_axis = y_axis / torch.linalg.norm(y_axis)
        z_axis = z_axis / torch.linalg.norm(z_axis)

    return torch.hstack(
        (
            x_axis.view(3, 1),
            y_axis.view(3, 1),
            z_axis.view(3, 1)
        )
    )


def is_valid_rotation(R, dim=3, tol=numeric.EPS):
    """is_valid_rotation.

    Checks whether the input is a valid rotation matrix, i.e., R.T @ R = I and
    det(R) == 1

    Args:
        R (torch.Tensor): square matrix.
    """
    try:
        M, N = R.shape
        if M != N or M != dim:
            # Not square or wrong dimensions.
            return False
    except ValueError:
        print("Input is ill formed.")
        return False

    if torch.allclose(torch.linalg.det(R), numeric.T1, rtol=tol, atol=tol):
        identity = R.T @ R
        if torch.allclose(identity, torch.eye(dim), rtol=tol, atol=tol):
            return True

    return False


def enforce_rotation(R):
    """enforce_rotation.

    Forces a matrix which is close to being a rotation to be a rotation. The
    first row of the output is a normalized version of the original first row,
    and the other rows are minimally perturbed so that the resulting matrix is
    a rotation. See
    http://www.iri.upc.edu/files/scidoc/2288-On-Closed-Form-Formulas-for-the-3D-Nearest-Rotation-Matrix-Problem.pdf
    for a reference.

    Parameters
    ----------
    R : Tensor
        A 3 x 3 torch tensor which is "nearly" a rotation.

    Results
    -------
        A 3 x 3 torch tensor X that is orthonormal, i.e., X * X.T = Id, and
        det(X) = 1.
    """
    # Input matrix must be 3 x 3.
    M, N = R.shape
    assert (M == 3)
    assert (N == 3)

    R_near, _ = torch.linalg.qr(R)

    return R_near


def hat_so3(x):
    """hat_so3.

    Create from the 3-dimensional vector x the matrix X such that X * y =
    torch.cross(x, y) for every 3-dimensional vector y. This is the hat-map
    isomorphism allowing moving from real vector space to so(3).

    Parameters
    ----------
    x : Tensor
        3-dimensional torch tensor

    Returns
    -------
    Tensor
        3 x 3 torch matrix X such that X * y is equal to the cross product of
        the vector x and another 3-dimensional vector y.
    """
    top = torch.hstack([numeric.T0, -x[2], x[1]])
    mid = torch.hstack([x[2], numeric.T0, -x[0]])
    bot = torch.hstack([-x[1], x[0], numeric.T0])

    return numeric.stack_tensors((top, mid, bot))


def vee_so3(X):
    """vee_so3.

    Linear transformation from an element of the algebra of so3 to its natural
    parameterization in R3.
    """
    return torch.stack((-X[1, 2], X[0, 2], -X[0, 1]))


def rotation_matrix(x):
    """rotation_matrix.

    Create a rotation matrix from a screw axis.

    Parameters
    ----------
    x : Tensor
        Three-dimensional tensor with direction corresponding to the direction
        of rotation and magnitude corresponding to the angle of rotation in
        radians.

    Results
    -------
    Tensor
        Rotation matrix.
    """
    angle = torch.linalg.norm(x)
    Id3 = torch.eye(3)
    if angle == 0:
        # At first sight, this may look strange. Since x = 0, why not just
        # return R = id? The answer is that if we did so, R would not have any
        # dependence on x for x = 0, and therefore dR_dx would be zero. The
        # expression below is the first-order approximation for R as a function
        # of x, and it yields the correct dR_dx at x = 0.
        R = Id3 + hat_so3(x)
        return R

    # Rodrigues' Rotation Formula
    x = x / angle
    X = hat_so3(x)
    R = Id3 + torch.sin(angle) * X + \
        (numeric.T1 - torch.cos(angle)) * torch.matmul(X, X)

    return R


def rotation_axis(R):
    """rotation_axis.

    Computes the rotation axis given the rotation matrix R. This is the inverse
    of the function rotation_matrix.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    theta = torch.acos((trace - 1) / 2)
    if theta == 0:
        # R is identity, but let's return a differentiable expression. This is
        # the first order approximation of the log matrix.
        return vee_so3(R - torch.eye(3))

    omega = \
        torch.stack(
            (R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1])
        ) / 2 / torch.sin(theta)

    return omega * theta


def rotation_matrix_from_u_to_v(u, v):
    """rotation_matrix_from_u_to_v.

    Constructs a rotation matrix which rotates u to v.
    """
    norm_u = torch.linalg.norm(u)
    norm_v = torch.linalg.norm(v)

    # Rotate around a vector orthogonal to both u and v.
    u_normalized = u / norm_u
    v_normalized = v / norm_v
    cross_prod = torch.cross(u_normalized, v_normalized)
    norm_cross_prod = torch.linalg.norm(cross_prod)
    if torch.abs(norm_cross_prod) < numeric.TEPS:
        # The vectors are parallel. We return a first-order approximation of
        # the result (which will be accurate due to the "< TEPS" condition)
        # which correctly propagates derivatives.
        return torch.eye(3) + hat_so3(cross_prod)

    # Rescale the vector so its norm is the angle between u and v. Use cosine
    # to understand angles larger than 90 degrees.
    cos_angle = torch.dot(u_normalized, v_normalized)
    angle = torch.acos(cos_angle)

    return rotation_matrix(cross_prod * (angle / norm_cross_prod))
