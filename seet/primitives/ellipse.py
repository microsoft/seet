"""ellipse.py

User-defined package defining and ellipse primitives.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import torch


class Ellipse(core.Node):
    """Ellipse.

    Represents and ellipse in 3D, which has a center, a normal, axes, and
    an angle.

    Args:
        node (Node): Ellipse is a subclass of Node.
    """

    def __init__(
        self,
        parent,
        transform_toParent_fromSelf,
        x_radius,
        y_radius,
        name="",
        requires_grad=False
    ):
        """
        In its own coordinate system, the origin is (0, 0, 0) and the normal
        is (0, 0, 1).

        Args:
            parent (Node): parent object in the pose graph.

            transform_toParent_fromSelf (SE3): SE3 object representing the
            transformation from the ellipse coordinate system to its parent's.

            x_radius (torch.Tensor): length of the semi-axis along x direction.

            y_radius (torch.Tensor): length of the semi-axis along y direction.

            name (string, optional): name of ellipse object. Defaults to "".
        """

        super().__init__(
            parent,
            transform_toParent_fromSelf,
            name=name,
            requires_grad=requires_grad
        )

        self.center = torch.zeros(3)
        self.normal = torch.Tensor([0.0, 0.0, 1.0])
        self.x_radius = x_radius
        self.y_radius = y_radius

    def get_args(self):
        """Augment base-class method.

        Returns:
            list: list of required arguments, in the order of init.
        """

        return super().get_args() + (self.x_radius, self.y_radius)

    @classmethod
    def create_from_origin_angle_and_axes_inPlane(
        cls,
        plane_node,
        center_in2DPlane,
        angle_deg,
        x_radius,
        y_radius,
        name=""
    ):
        """
        Create an ellipse from its origin, orientation of its x-semi axis,
        and lengths of its x and y semi-axes in the canonical coordinate system
        of plane_node.

        Args:
            plane_node (Plane): parent node of ellipse. Establishes the 2D
            coordinate system in which other input parameters are represented.

            center_in2DPlane (torch.Tensor): (2,) torch Tensor representing the
            coordinates of the ellipse center in the plane's internal 2D
            coordinate system. In the plane's own 3D coordinate system this
            point is given by plane_node.orthonormal * origin_in2DPlane.

            angle_deg (float or torch.Tensor): angle in degree of the positive
            x-semi axis of the ellipse with respect to the positive x axis of
            the plane's internal 2D coordinate system.

            x_radius (torch.Tensor): length of x-semi axis.

            y_radius (torch.Tensor): length of y-semi axis.

        Returns:
            Ellipse: Ellipse object as a child of the input plane.
        """
        # We need to create the 3D transform to plane from ellipse to plane.
        # Its matrix T is partitioned as
        #
        #     [rotation_to2DPlane_fromEllipse, 0, center_in2DPlane]
        # T = [                             0, 1,                0] [ 0, 0, 1]
        angle_rad = core.deg_to_rad(angle_deg)
        c = torch.cos(angle_rad)
        s = torch.sin(angle_rad)
        rotation_to2DPlane_fromEllipse = \
            core.stack_tensors(
                ((c, -s),
                 (s, c))

            )
        top_2x4 = \
            torch.hstack(
                (
                    rotation_to2DPlane_fromEllipse,
                    torch.zeros(2, 1),
                    center_in2DPlane.view(2, 1)
                )
            )
        bottom_2x4 = \
            core.stack_tensors(
                (
                    torch.tensor([0.0, 0.0, 1.0, 0.0]),
                    torch.tensor([0.0, 0.0, 0.0, 1.0])
                )
            )
        transform_matrix_to3DPlane_fromEllipse = \
            torch.vstack((top_2x4, bottom_2x4))
        transform_to3DPlane_fromEllipse = \
            core.SE3(transform_matrix_to3DPlane_fromEllipse)

        return cls(
            plane_node,
            transform_to3DPlane_fromEllipse,
            x_radius,
            y_radius,
            name=name
        )

    @classmethod
    def create_from_homogeneous_matrix_inPlane(
        cls, plane_node, C_in2DPlane, name=""
    ):
        """
        Create an ellipse from the symmetric matrix C representing the
        equation of the ellipse in the coordinate system of the ellipse's
        parent plane.

        Args:
            C_in2DPlane (torch.Tensor): (3, 3) symmetric torch.Tensor
            representing the ellipse in homogeneous coordinates in the
            coordinate system of the ellipse's parent plane.

            name (string, optional): name of ellipse object.
        """
        # Ensure that C_inParent is symmetric.
        C_in2DPlane = C_in2DPlane + C_in2DPlane.T

        # In the plane's two-dimensional coordinate system, a point u lies on
        # the ellipse if u_h, the representation of u in homogeneous
        # coordinates satisfies u_h.T * C_in2DPlane * u_h.

        # We need to obtain the center, orientation, and lengths of the
        # semi-axes of the ellipse in the plane's coordinate system.

        # Find the center of the ellipse in the plane's coordinate system.
        center_in2DPlane = \
            -torch.linalg.solve(C_in2DPlane[:2, :2], C_in2DPlane[:2, -1])

        # Normalize the ellipse matrix to make life easier.
        aux = center_in2DPlane @ C_in2DPlane[:2, :2] @ center_in2DPlane
        # aux - C_inPlane[-1, -1] must be 1.
        scale = aux - C_in2DPlane[-1, -1]
        C_in2DPlane = C_in2DPlane / scale

        # SVD returns, U, L, V.T, where V.T is a rotation from 2D plane to the
        # ellipse coordinates.
        inverse_square_semi_axes_length, rotation_toEllipse_from2DPlane = \
            torch.linalg.eigh(C_in2DPlane[:2, :2])

        if torch.linalg.det(rotation_toEllipse_from2DPlane) < 0:
            rotation_toEllipse_from2DPlane = \
                torch.diag(torch.tensor([-1.0, 1.0])) @ \
                rotation_toEllipse_from2DPlane
        rotation_to2DPlane_fromEllipse = rotation_toEllipse_from2DPlane.T

        # We need to create the 3D transform to plane from ellipse to plane.
        # Its matrix T is partitioned as
        #
        #     [rotation_to2DPlane_fromEllipse, 0, center_in2DPlane] <--- top
        # T = [                             0, 1,                0] <-| [
        #     0, 0,                1] <-|- bottom
        top_2x4 = \
            torch.hstack(
                (
                    rotation_to2DPlane_fromEllipse,
                    torch.zeros(2, 1),
                    center_in2DPlane.view(2, 1)
                )
            )
        bottom_2x4 = \
            core.stack_tensors(
                (
                    torch.tensor([0.0, 0.0, 1.0, 0.0]),
                    torch.tensor([0.0, 0.0, 0.0, 1.0])
                )
            )
        transform_matrix_to3DPlane_fromEllipse = \
            torch.vstack((top_2x4, bottom_2x4))
        transform_to3DPlane_fromEllipse = \
            core.SE3(transform_matrix_to3DPlane_fromEllipse)

        # Eigenvalues are in ascending order, so the larger axis is the square
        # root of the inverse of the first eigenvalue.
        x_radius, y_radius = 1 / torch.sqrt(inverse_square_semi_axes_length)

        return cls(
            plane_node,
            transform_to3DPlane_fromEllipse,
            x_radius,
            y_radius,
            name=name
        )

    def get_homogeneous_matrix_inPlane(self):
        """get_homogeneous_matrix_inPlane.

        Get the symmetric matrix representing the ellipse in homogeneous
        coordinates in the coordinate system of the ellipse's parent plane.
        """

        matrix_inEllipse = \
            torch.diag(
                1 /
                torch.stack(
                    (
                        self.x_radius * self.x_radius,
                        self.y_radius * self.y_radius,
                        torch.tensor(-1.0)
                    )
                )
            )

        # We assume that the parent is a plane. In this case, the
        # transformation T from the ellipse to the plane coordinate system has
        # a special form:
        #
        #       (4 x 2)
        #          |
        #          V
        #        [Rxy | 0 cxy] -> (2 x 4)
        #        [----+------]
        # T_3D = [0 0 | 1   0] -> (1 x 4)
        #        [----+------]
        #        [0 0 | 0   1] -> (1 x 4)
        #                 ^
        #                 |
        #              (4 x 2)
        #
        # This allows for the 2D to 2D transformation from the ellipse
        # coordinate system to the plane coordinate system to be expressed as
        #
        #        [Rxy | cxy]
        # T_2D = [----+----]
        #        [0 0 |   1]
        #
        # by deleting the third row and column of the original matrix.

        T_3D_toEllipse_fromPlane = \
            self.transform_toParent_fromSelf.inverse_transform_matrix
        T_2D_toEllipse_fromPlane = \
            torch.vstack(
                (
                    T_3D_toEllipse_fromPlane[:2, :],
                    T_3D_toEllipse_fromPlane[-1, :].view((1, 4))
                )
            )
        T_2D_toEllipse_fromPlane = \
            torch.hstack(
                (
                    T_2D_toEllipse_fromPlane[:, :2],
                    T_2D_toEllipse_fromPlane[:, -1].view((3, 1))
                )
            )
        matrix_inPlane = \
            T_2D_toEllipse_fromPlane.T @ \
            matrix_inEllipse @ T_2D_toEllipse_fromPlane

        return matrix_inPlane

    def set_radii(self, x_radius, y_radius):
        """set_radii.

        Set semi-axes to new values.

        Args:
            x_radius (torch.Tensor): new value of x_radius.

            y_radius (torch.Tensor): new value of y_radius.
        """
        self.x_radius = x_radius
        self.y_radius = y_radius

    def update_radii(self, x_update, y_update, update_mode="additive"):
        """update_radii.

        Apply an additive or multiplicative update to the lengths of the
        semi-axes.

        An additive update means that the new values of the semi-axes will be
        self.x_radius += x_update and self.y_radius += y_update.

        A multiplicative update means that the new values of the semi-axes will
        be self.x_radius *= exp(x_update) self.y_radius *= exp(y_update)

        Args:
            update_x (float or torch.Tensor): additive or multiplicative update
            to be applied to the x radius of the ellipse.

            update_y (float or torch.Tensor): analogous to update_x, but
            applied to the y radius.

            update_type (string, optional): Type of update to be applied.
            Additive update if update_type is "additive", multiplicative update
            if update_type is "multiplicative". Defaults to "additive".
        """
        if update_mode == "multiplicative":
            new_x_radius = self.x_radius * torch.exp(x_update)
            new_y_radius = self.y_radius * torch.exp(y_update)
        else:
            new_x_radius = self.x_radius + x_update
            new_y_radius = self.y_radius + y_update

        self.set_radii(new_x_radius, new_y_radius)

    def get_center_and_normal_inOther(self, other):
        """get_center_and_normal_inOther.

        Get the center and the normal of the ellipse in some arbitrary
        coordinate system.

        Args:
            other (Node): Node object in which to represent the coordinates of
            the center and normal to the ellipse.

        Returns:
            tuple: (2,) tuple (origin_inOther, normal_inOther) where each
            element is a (3,) torch.Tensor.
        """
        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)
        origin_inOther = transform_toOther_fromSelf.transform(self.center)
        normal_inOther = transform_toOther_fromSelf.rotation.transform(
            self.normal)

        return origin_inOther, normal_inOther

    def get_center_and_normal_inParent(self):
        """get_center_and_normal_inParent.

        Get the center and normal of the ellipse in the ellipse's parent
        coordinate system.

        Returns:
            tuple: (2,) tuple (origin_inParent, normal_inParent) where each
            element is a (3,) torch.Tensor.
        """
        return self.get_center_and_normal_inOther(self.parent)

    def get_points_at_angles_inEllipse(self, angles_rad):
        """get_points_at_angles_inEllipse.

        Get the points on the ellipse at the given polar angles in radians.

        Args:
            angles_rad (torch.Tensor): angles in radians at which the points on
            the ellipse are sought.
        """

        # A point in polar coordinates is r*(cos(angle), sin(angle)). Plugging
        # this point on the ellipse equation we get
        #
        # r^2 = 1 / (cos^2(angle)/a^2 + sin^2(angle)/b^2).

        c = torch.cos(angles_rad)
        c_over_a = c / self.x_radius
        s = torch.sin(angles_rad)
        s_over_b = s / self.y_radius

        r2 = 1 / (c_over_a * c_over_a + s_over_b * s_over_b)
        r = torch.sqrt(r2)

        return torch.stack((r * c, r * s, torch.zeros_like(angles_rad)))

    def get_normal_at_angles_inEllipse(self, angles_rad):
        """get_normal_at_angles_inEllipse.

        Get the normal direction at the point in the ellipse with given
        polar angle. The directions are not normalized, i.e., their magnitudes
        do not have to equal 1.

        Args:
            angles_rad (torch.Tensor): polar angles in radians defining points
            on the ellipse at which normal directions are sought.
        """

        point_inEllipse = self.get_points_at_angles_inEllipse(angles_rad)
        angular_parameters = \
            torch.atan2(
                point_inEllipse[1, ...] / self.y_radius,
                point_inEllipse[0, ...] / self.x_radius
            )
        c = torch.cos(angular_parameters)
        s = torch.sin(angular_parameters)

        normal_direction = \
            torch.stack(
                (
                    self.y_radius * c,
                    self.x_radius * s,
                    torch.zeros_like(angles_rad)
                )
            )

        return normal_direction

    def sample_points_inEllipse(self, num_points=30):
        """sample_points_inEllipse.

        Sample points along the ellipse.

        Args:
            num_points (int, optional): number of points to be sampled.
            Defaults to 30.

        Returns:
            torch.Tensor: (3, num_points) torch.Tensor representing sampled
            points in the ellipse coordinate system.

            angles_rad: (num_points,) torch.Tensor containing the angles in
            radians that parameterize each sampled point.
        """

        # If derivatives with respect to shape parameters are required, assume
        # that derivatives with respect to angles are required as well.
        requires_grad = \
            self.x_radius.requires_grad or self.y_radius.requires_grad

        # Use num_points + 1 and remove the last point because torch.linspace
        # does not have an option to not include the last point, and we do not
        # want to do a complete wrap around.
        angles_rad_tensor = \
            torch.linspace(
                0,
                2 * core.TPI.item(),
                num_points + 1,
                requires_grad=requires_grad
            )[:-1]

        # Same operations can be vectorized if the angles are held on a single
        # tensor, whereas other operations, particularly computation of
        # derivatives, benefit from each angle begin represented as a single
        # tensor.
        angles_rad = torch.split(angles_rad_tensor, 1)

        # This reconstitutes angles_rad_tensor as a function of angle_rad. This
        # may look redundant, because we are breaking up the original tensor
        # and reconstituting it back again, but this step is important because
        # now tensors that depend on angles_rad_tensor will also depend on
        # angles_rad.
        angles_rad_tensor = torch.stack(angles_rad).flatten()

        return \
            self.get_points_at_angles_inEllipse(angles_rad_tensor), angles_rad

    def get_points_inOther(self, other, num_points=30):
        """get_points_inOther.

        Sample points along the ellipse and return their coordinates in the
        coordinate system of Node other.

        Args:
            other (Node): Node object in which to return the coordinates of the
            sampled points.

            num_points (int, optional): Number of points to be sampled.
            Defaults to self.num_points.

        Returns:
            torch.Tensor: (3, N) torch.Tensor holding coordinates of N sampled
            points in the coordinate system of node other.

            angles_rad: (num_points,) torch.Tensor containing the angles in
            radians that parameterize each sampled point.
        """
        points_inSelf, \
            angles_rad = self.sample_points_inEllipse(num_points=num_points)
        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)

        return \
            transform_toOther_fromSelf.transform(points_inSelf), angles_rad

    def get_points_inParent(self, num_points=30):
        """get_points_inParent.

        Get sampled points along the ellipse in ellipse's parent's
        coordinate system.

        Returns:
            torch.Tensor: (3, self.num_points) torch.Tensor representing points
            sampled along the ellipse, expressed in the coordinate system of
            the ellipses parent node.

            angles_rad: (num_points,) torch.Tensor containing the angles in
            radians that parameterize each sampled point.
        """
        points_inSelf, \
            angles_rad = self.sample_points_inEllipse(num_points=num_points)

        return \
            self.transform_toParent_fromSelf.transform(points_inSelf), \
            angles_rad

    def compute_angle_of_closest_point(self, point_inEllipse):
        """compute_angle_of_closest_point.

        Get the polar angle of a point in ellipse closest to the input
        point, provided in the ellipse's coordinate system. Correctly
        propagates gradients.

        Args:
            point_inEllipse (torch.Tensor): point for which the angle of the
            closest point on the ellipse is sought.

        Returns:
            torch.Tensor: polar angle of closest point on ellipse to input
            point.
        """

        # We need to solve for vector (x, y) in the ellipse that is closest to
        # input point (x0, y0).
        #
        # (x, y) = arg_min ||(x, y) - (x0, y0)||^2
        #          subj. to x^2/a^2 + y^2/b^2 - 1 = 0
        #
        # This yields the following polynomial on the Lagrange multiplier s:
        #
        # s^4 + 2*(a^2 + b^2)*s^3 +
        #   (a^4 + 4*a^2*b^2 + b^4 - a^2*x0^2 - b^2*y0^2)*s^2 +
        #       (2*a^4*b^2 + 2*a^2*b^4 - 2*a^2*b^2*x0^2 - 2*a^2*b^2*y0^2)*s +
        #           a^4*b^4 - a^2*b^4*x0^2 - a^4*b^2*y0^2 = 0

        # Let's use the notation above.
        a2 = self.x_radius * self.x_radius
        b2 = self.y_radius * self.y_radius
        a4 = a2 * a2
        b4 = b2 * b2
        a2b2 = a2 * b2
        a4b2 = a4 * b2
        a2b4 = a2 * b4

        x0 = point_inEllipse[0]
        x02 = x0 * x0
        y0 = point_inEllipse[1]
        y02 = y0 * y0

        p0 = a4 * b4 - a2b4 * x02 - a4b2 * y02
        p1 = 2 * (a4b2 + a2b4 - a2b2 * x02 - a2b2 * y02)
        p2 = (a4 + 4 * a2b2 + b4 - a2 * x02 - b2 * y02)
        p3 = 2 * (a2 + b2)
        p4 = torch.tensor(1.0)

        p = torch.hstack((p0, p1, p2, p3, p4))
        roots = core.poly_solver(p)

        # We get up to four solutions. We want the one corresponding to the
        # point closest to point_inEllipse.
        min_dist = torch.inf
        result_inEllipse = \
            torch.hstack((self.x_radius, self.y_radius)) / \
            torch.sqrt(torch.tensor(2.0))
        for s in roots:
            candidate_inEllipse = \
                torch.hstack((a2 / (s + a2) * x0, b2 / (s + b2) * y0))
            difference = candidate_inEllipse - point_inEllipse[:2]
            new_dist = difference @ difference
            if new_dist < min_dist:
                result_inEllipse = candidate_inEllipse
                min_dist = new_dist

        return torch.atan2(result_inEllipse[1], result_inEllipse[0])
        return \
            self._compute_angle_of_closest_point_internal.apply(
                point_inEllipse,
                torch.stack((self.x_radius, self.y_radius)),
                self
            )
