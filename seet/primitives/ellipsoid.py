"""ellipsoid.py

Class defining and ellipsoidal object.
"""


__author__ = "Chris Aholt (chaholt@microsoft.com)"


import seet.core as core
import seet.primitives as primitives
import torch


class Ellipsoid(core.Node):
    """Ellipsoid.

    Object representing and ellipsoid in 3D. Ellipsoid has a center and
    semi-axes of given length.

    Args:
        node (Node): Ellipsoid is a subclass of Node.
    """

    def __init__(
        self,
        parent,
        transform_toParent_fromSelf,
        shape_parameters,
        name="",
        requires_grad=False
    ):
        """
        Create an Ellipsoid object from a parent Node, a transformation from
        the coordinate system of the Ellipsoid to Node, shape parameters
        (x_axis, y_axis, z_axis) corresponding to the lengths of the semi-axes
        of the Ellipsoid.

        Args:
            parent (Node): parent Node of the Ellipsoid object in the pose
            graph.

            transform_toParent_fromSelf (SE3): SE3 object representing the
            transformation of coordinate system from the Ellipsoid to its
            parent Node.

            shape_parameters (torch.Tensor): (3,) or (3, 1) Tensor with lengths
            of x, y, and z semi-axes of the Ellipsoid.

            name (string, optional): name of Ellipsoid object. Defaults to
            None.

            requires_grad (bool, optional): whether node tensors require
            computation of gradient. Defaults to False.
        """
        super().__init__(
            parent,
            transform_toParent_fromSelf,
            name=name,
            requires_grad=requires_grad
        )

        self.reset_shape(shape_parameters)

    def get_args(self):
        """Augment base-class method.

        Returns:
            list: list of required arguments, in the order of init.
        """

        return super().get_args() + (self.shape_parameters,)

    def reset_shape(self, shape_parameters):
        self.shape_parameters = shape_parameters
        self.diagonal = \
            core.T1 / (self.shape_parameters * self.shape_parameters)

    def get_ellipsoid_matrix_inEllipsoid(self):
        """get_ellipsoid_matrix_inEllipsoid.

        Compute the matrix Q in the ellipsoid own coordinate system such
        that homogeneous points h_x in the ellipsoid coordinate system satisfy
        h_x @ Q @ h_x = 0.

        Returns:
            torch.Tensor: (4, 4) matrix representing ellipsoid in homogenous
            coordinates in in the ellipsoid own coordinate system.
        """

        return torch.diag(torch.hstack((self.diagonal, torch.tensor(-1.0))))

    def get_ellipsoid_matrix_inOther(self, other):
        """get_ellipsoid_matrix_inOther.

        Compute the matrix Q in representing the ellipsoid in homogeneous
        coordinate system in the coordinates of the node other in the pose
        graph.
        """

        Q_inEllipsoid = self.get_ellipsoid_matrix_inEllipsoid()
        transform_toOther_fromEllipsoid = self.get_transform_toOther_fromSelf(other)
        transform_matrix_toEllipsoid_fromOther = transform_toOther_fromEllipsoid.inverse_transform_matrix

        # Ensure all tensors are on the same device
        target_device = transform_matrix_toEllipsoid_fromOther.device
        if Q_inEllipsoid.device != target_device:
            Q_inEllipsoid = Q_inEllipsoid.to(target_device)

        return transform_matrix_toEllipsoid_fromOther.T @ Q_inEllipsoid @ transform_matrix_toEllipsoid_fromOther

    def intersect_from_origin_and_direction_inEllipsoid(
        self, origin_inEllipsoid, direction_inEllipsoid
    ):
        """
        Intersect with ellipsoid a ray from origin an direction in ellipsoid
        coordinate system.

        Args:
            origin_inEllipsoid (torch.Tensor): (3,) tensor corresponding to
            coordinates of origin of ray in coordinate system of ellipsoid.

            direction_inEllipsoid (torch.Tensor): (3,) tensor corresponding to
            direction of ray in coordinate system of ellipsoid.

        Returns:
            (torch.Tensor, torch.Tensor): pair of (3,) torch tensors
            corresponding to intersection points of ray from origing to
            destination and ellipsoid.
        """
        p0 = torch.sum((origin_inEllipsoid**2) * self.diagonal) - 1
        p1 = \
            2 * torch.sum(
                origin_inEllipsoid * direction_inEllipsoid * self.diagonal
            )
        p2 = torch.sum((direction_inEllipsoid**2) * self.diagonal)

        poly_coefficients = torch.stack((p0, p1, p2))
        t = core.poly_solver(poly_coefficients)

        if torch.any(torch.logical_not(torch.isreal(t))) or len(t) == 0:
            return None, None

        # There are two solutions, and we return both of them. The first
        # solution is the one closest to the origin of the ray.
        t0 = torch.tensor([1.0, 0.0]) @ t
        t1 = torch.tensor([0.0, 1.0]) @ t
        if torch.abs(t0) > torch.abs(t1):
            intersection_1_inEllipsoid = \
                origin_inEllipsoid + direction_inEllipsoid * t1
            intersection_2_inEllipsoid = \
                origin_inEllipsoid + direction_inEllipsoid * t0
        else:
            intersection_1_inEllipsoid = \
                origin_inEllipsoid + direction_inEllipsoid * t0
            intersection_2_inEllipsoid = \
                origin_inEllipsoid + direction_inEllipsoid * t1

        return intersection_1_inEllipsoid, intersection_2_inEllipsoid

    def intersect_ray_inEllipsoid(self, input_ray) -> tuple:
        """intersect_ray_inEllipsoid.

        Intersect a ray with the surface of the ellipsoid.

        Args:
            input_ray (Ray): ray object in the same pose graph as Ellipsoid.

        Returns:
            (torch.Tensor, torch.Tensor): tuple with two (3,) torch.Tensor
            objects each representing an intersection of ray with ellipsoid
            surface in the coordinate system of the ellipsoid.
        """
        origin_inEllipsoid, direction_inEllipsoid = \
            input_ray.get_origin_and_direction_inOther(self)

        return self.intersect_from_origin_and_direction_inEllipsoid(
            origin_inEllipsoid, direction_inEllipsoid
        )

    def compute_algebraic_distance_inEllipsoid(self, point_inEllipsoid):
        """compute_algebraic_distance_inEllipsoid.

        Computes the algebraic distance between the input point and the
        Ellipsoid object.

        The algebraic distance is the value of (x - x0)/a^2 + (y - y0)^2/b^2 +
        (z - z0)^2/c^2 - 1.

        Args:
            input_point (torch.Tensor): (3,) or (3, N) torch.Tensor
            representing one or more points in the pose graph of the ellipsoid.

        Returns:
            torch.Tensor: algebraic distance between the input point or points
            and the ellipsoid.
        """
        return \
            torch.sum((point_inEllipsoid ** 2) * self.diagonal) - core.T1

    def compute_algebraic_distance(self, input_point) -> torch.Tensor:
        """compute_algebraic_distance.

        Computes the algebraic distance between the input point and the
        Ellipsoid object.

        The algebraic distance is the value of (x - x0)/a^2 + (y - y0)^2/b^2 +
        (z - z0)^2/c^2 - 1.

        Args:
            input_point (torch.Tensor): (3,) or (3, N) torch.Tensor
            representing one or more points in the pose graph of the ellipsoid.

        Returns:
            torch.Tensor: algebraic distance between the input point or points
            and the ellipsoid.
        """
        point_inEllipsoid = input_point.get_coordinates_inOther(self)

        return self.compute_algebraic_distance_inEllipsoid(point_inEllipsoid)

    def compute_apex_inOther(self, other) -> torch.Tensor:
        """compute_apex_inOther.

        Compute the coordinates of the ellipsoid's apex in the coordinate
        system of the Node object other.

        Args:
            other (Node): Node object in the pose graph.

        Returns:
            torch.Tensor: (3,) tensor corresponding to the coordinates of the
            ellipsoid's apex in the coordinate system of Node object other.
        """
        apex_inEllipsoid = \
            torch.tensor([0.0, 0.0, 1.0]) * self.shape_parameters

        transform_toOther_fromSelf = \
            self.get_transform_toOther_fromSelf(other)

        return \
            transform_toOther_fromSelf.transform(apex_inEllipsoid)

    def compute_apex_inParent(self) -> torch.Tensor:
        """compute_apex_inParent.

        Compute the coordinate of the ellipsoid's apex in the coordinate
        system of the ellipsoid's parent.

        Returns:
            torch.Tensor: (3,) tensor corresponding to the coordinate of the
            ellipsoid's apex in the coordinate system of the ellipsoid's
            parent.
        """
        return self.compute_apex_inOther(self.parent)

    def compute_polar_plane_to_point_inEllipsoid(
        self, point_inEllipsoid
    ):
        """
        Compute the plane in homogeneous coordinates, polar to the input
        point. Both the input point and the output plane are in the coordinate
        system of the ellipsoid

        Args:
            point_inEllipsoid (torch.Tensor): (3,) tensor representing point in
            coordinate system of the ellipsoid.

        Returns:
            torch.Tensor: (4,) tensor representing homogeneous coordinates in
            the coordinate system of the ellipsoid of plane polar to the input
            point.
        """
        # The polar plane in homogeneous coordinates to homogeneous point x_h
        # is given by Q * x_h, where Q is the 4 x 4 matrix representation of
        # the ellipsoid in the same coordinate system as the point x_h. In the
        # ellipsoid's own coordinate system Q is diag(1/a^2, 1/b^2, 1/c^2, -1).
        homogeneous_plane_inEllipsoid = \
            torch.hstack(
                (point_inEllipsoid * self.diagonal, -core.T1.view(1))
            )

        return homogeneous_plane_inEllipsoid

    def compute_polar_plane_to_point(self, point_node):
        """compute_polar_plane_to_point.

        Compute the Plane object that is polar to the Point object
        point_node with respect to the ellipsoid.

        Args:
            point_node (Point): Point object whose polar plane with respect to
            the ellipsoid is to be computed.

        Returns:
            Plane: Plane object corresponding to polar plane to input node.
        """
        point_inEllipsoid = point_node.get_coordinates_inOther(self)

        homogeneous_plane_inEllipsoid = \
            self.compute_polar_plane_to_point_inEllipsoid(
                point_inEllipsoid
            )

        return primitives.Plane.create_from_homogeneous_coordinates_inParent(
            self, homogeneous_plane_inEllipsoid
        )

    def compute_polar_point_to_plane_inEllipsoid(
        self, homogeneous_plane_inEllipsoid
    ):
        """
        Compute point polar to plane in the ellipsoid coordinate system.

        Args:
            homogeneous_plane_inEllipsoid (torch.Tensor): (4,) tensor
            representing homogeneous coordinates of input plane in coordinate
            system of the ellipsoid.

        Returns:
            torch.Tensor: (3,) tensor representing coordinates of point polar
            to plane in coordinate system of ellipsoid.
        """
        homogenous_point_inEllipsoid = \
            torch.hstack(
                (
                    homogeneous_plane_inEllipsoid[:3] / self.diagonal,
                    -homogeneous_plane_inEllipsoid[-1]
                )
            )

        return core.dehomogenize(homogenous_point_inEllipsoid)

    # Note that rather than creating a node in the pose graph, which is
    # coordinate system independent, we output coordinates of the reflection
    # point in the coordinate system of the ellipsoid. The reason for this
    # distinction is that nodes are permanent, and remain attached to the pose
    # graph unless explicitly removed, whereas points are ephemeral, lasting
    # for only as long as we need them.
    def project_on_surface_inEllipsoid(
        self, point_inEllipsoid
    ) -> torch.Tensor:
        """
        Project point onto surface of ellipsoid.

        Assuming an ellipsoid Q = diag([1/a^2, 1/b^2, 1/c^2]) the orthogonal
        projection of a point y onto the surface of Q produces a point x such
        that

        x.T * Q * x = 1 x + Q * x * s = y

        where Q * x is the normal to the ellipsoid at x and s is an unknown
        scale factor. The scale factor should be positive if y is outside Q and
        negative otherwise.

        The second equation results in

        x = diag([a^2/(a^2 + s), b^2/(b^2 + s), c^2/(c^2 + s)]) * y

        and we need to solve for s such that the first equation is satisfied.
        This yields a 6-degree polynomial in s. We find the roots of this
        polynomial and plug the appropriate root into the expression above.

        Args:
            point_inEllipsoid (torch.Tensor): (3,) torch Tensor representing
            coordinates of the point to be projected in the coordinate system
            of the elliposid.

        Returns:
            torch.Tensor: (3,) tensor corresponding to coordinates of the
            projection of input_point onto the surface of the ellipsoid,
            expressed in the ellipsoid's coordinate system.
        """
        a2, b2, c2 = 1 / self.diagonal

        # The coefficients of the polynomial were obtained used sympy using the
        # following code:
        #
        # import sympy
        #
        # a2, b2, c2, y1, y2, y3, x = \
        # sympy.symbols("a2, b2, c2, y1, y2, y3, x")
        #
        # a4 = (x + a2)**2 b4 = (x + b2)**2 c4 = (x + c2)**2
        #
        # coeffs = sympy.Poly(a4 * b4 * c4 - y1**2 * a2 * b4 * c4 - \
        #   y2**2 * a4 * b2 * c4 - y3**2 * a4 * b4 * c2, x).all_coeffs()
        #
        # for i, co in enumerate(coeffs): print("i: {0}, factor: {1}".format(i,
        #   sympy.factor(co)))
        y1, y2, y3 = point_inEllipsoid
        poly = torch.stack(
            (
                -a2 * b2 * c2 * (-a2 * b2 * c2 + a2 * b2 * y3 **
                                 2 + a2 * c2 * y2**2 + b2 * c2 * y1**2),
                -2 * a2 * b2 * c2 * (-a2 * b2 - a2 * c2 + a2 * y2**2 +
                                     a2 * y3**2 - b2 * c2 + b2 * y1**2 +
                                     b2 * y3**2 + c2 * y1**2 + c2 * y2**2),
                a2**2 * b2**2 + 4 * a2**2 * b2 * c2 - a2**2 * b2 * y2**2 +
                a2**2 * c2**2 - a2**2 * c2 * y3**2 + 4 * a2 * b2**2 * c2 -
                a2 * b2**2 * y1**2 + 4 * a2 * b2 * c2**2 -
                4 * a2 * b2 * c2 * y1**2 - 4 * a2 * b2 * c2 * y2**2 -
                4 * a2 * b2 * c2 * y3**2 - a2 * c2**2 * y1**2 +
                b2**2 * c2**2 - b2**2 * c2 * y3**2 - b2 * c2**2 * y2**2,
                -2 * (-a2**2 * b2 - a2**2 * c2 - a2 * b2**2 -
                      4 * a2 * b2 * c2 + a2 * b2 * y1**2 + a2 * b2 * y2**2 -
                      a2 * c2**2 + a2 * c2 * y1**2 + a2 * c2 * y3**2 -
                      b2**2 * c2 - b2 * c2**2 + b2 * c2 * y2**2 +
                      b2 * c2 * y3**2),
                a2**2 + 4 * a2 * b2 + 4 * a2 * c2 - a2 * y1**2 + b2**2 +
                4 * b2 * c2 - b2 * y2**2 + c2**2 - c2 * y3**2,
                2 * (a2 + b2 + c2),
                core.T1
            )
        )

        roots = core.poly_solver(poly, force_real=True)
        dist = torch.tensor(float("inf"))
        x = point_inEllipsoid
        for root in roots:
            diag = torch.stack(
                (a2 / (a2 + root), b2 / (b2 + root), c2 / (c2 + root))
            )
            Q = torch.diag(diag)
            new_x = Q @ point_inEllipsoid

            # There will be up to six real solutions to the polynomial
            # equation. Keep the one that is closest to the input point.
            new_dist = torch.linalg.norm(new_x - point_inEllipsoid)
            if new_dist < dist:
                dist = new_dist
                x = new_x

        return x

    def compute_occluding_contour_inEllipsoid(
        self,
        point_inEllipsoid,
        tol=core.TEPS * 100,
        name=""
    ):
        """
        Compute the elliptical occluding contour from the point of view of
        point_inEllipsoid.

        Args:
            point_inEllipsoid (torch.Tensor): (3,) tensor representing
            coordinates of input point in coordinate system of ellipsoid.

            tol (float or torch.Tensor): tolerance in algebraic distance used
            to determine whether point_node is outside ellipsoid. Occluding
            contour is not defined for points that are not outside the
            ellipsoid.

            name (string, optional): name of Ellipse object representing the
            occluding contour.

        Returns:
            Ellipse: Ellipse object representing the occluding contour observed
            by point.
        """
        # Occluding contour is the set of points that satisfy the equation of
        # the ellipsoid and lie on the polar plane of the point with respect to
        # the ellipsoid. Given the polar plane p_h = (n, -d) in homogeneous
        # coordinates and the matrix Q of the quadric, the points X on the
        # occluding contour satisfy
        #
        # n.T * X = d X.T * Q * X = 1
        #
        # The solution of the first equation is X = n * d + N * q, where N is
        # the right null space of n and q is arbitrary. Plugging this solution
        # in the second equation we get
        #
        #          [N.T,      0] [Q,     0] [N,  n * d] [q]
        # [q.T, 1] [d * n.T,  1] [0.T,  -1] [0,      1] [1] = 0.
        #
        # We attach to the plane a coordinate system with origin n * d and
        # orthonormal basis N, where these entities are represented in the
        # coordinate system of the ellipsoid. The point q is a point in that
        # planar (2D) coordinate system.

        # If point_node is not outside of ellipsoid occluding contour is not
        # defined.
        if \
            self.compute_algebraic_distance_inEllipsoid(
                point_inEllipsoid) <= tol:
            return None

        Q = torch.diag(torch.hstack((self.diagonal, -core.T1.clone())))

        # Get the polar plane to the ellipsoid with respect to the input point.
        homogenous_plane_inEllipsoid = \
            self.compute_polar_plane_to_point_inEllipsoid(point_inEllipsoid)
        polar_plane = \
            primitives.Plane.create_from_homogeneous_coordinates_inParent(
                self, homogenous_plane_inEllipsoid
            )
        origin_inEllipsoid, _ = \
            polar_plane.get_origin_and_normal_inOther(self)
        orthonormal_inEllipsoid = polar_plane.get_orthonormal_inOther(self)

        # This is a strange transformation matrix, as it is 4 x 3. It maps from
        # a normalized coordinate system on the plane (2D) to the coordinate
        # system of the ellipsoid (3D).
        transform_toEllipsoid_from2DPlane = \
            torch.vstack(
                (
                    torch.hstack(
                        (
                            orthonormal_inEllipsoid,
                            origin_inEllipsoid.view(3, 1)
                        )
                    ),
                    torch.tensor([0.0, 0.0, 1.0])
                )
            )

        C_in2DPlane = \
            transform_toEllipsoid_from2DPlane.T @ \
            Q @ \
            transform_toEllipsoid_from2DPlane

        occluding_contour = \
            primitives.Ellipse.create_from_homogeneous_matrix_inPlane(
                polar_plane, C_in2DPlane, name=name
            )

        # Detach the occluding contour from its plane and attach it directly to
        # the ellipsoid.
        self.add_child(occluding_contour)

        # Do not bloat the pose graph.
        self.remove_child(polar_plane)

        return occluding_contour

    def compute_occluding_contour(
        self,
        point_node,
        tol=core.TEPS * 100,
        name=""
    ):
        """
        Compute the elliptical occluding contour from the point of view of
        point_node.

        Args:
            point_node (Point): Point object with respect to which we compute
            the occluding contour.

            tol (float or torch.Tensor): tolerance in algebraic distance used
            to determine whether point_node is outside ellipsoid. Occluding
            contour is not defined for points that are not outside the
            ellipsoid.

            name (string, optional): name of Ellipse object representing the
            occluding contour.

        Returns:
            Ellipse: Ellipse object representing the occluding contour observed
            by point.
        """
        # Occluding contour is the set of points that satisfy the equation of
        # the ellipsoid and lie on the polar plane of the point with respect to
        # the ellipsoid. Given the polar plane p_h = (n, -d) in homogeneous
        # coordinates and the matrix Q of the quadric, the points X on the
        # occluding contour satisfy
        #
        # n.T * X = d X.T * Q * X = 1
        #
        # The solution of the first equation is X = n * d + N * q, where N is
        # the right null space of n and q is arbitrary. Plugging this solution
        # in the second equation we get
        #
        #          [N.T,      0] [Q,     0] [N,  n * d] [q]
        # [q.T, 1] [d * n.T,  1] [0.T,  -1] [0,      1] [1] = 0.
        #
        # We attach to the plane a coordinate system with origin n * d and
        # orthonormal basis N, where these entities are represented in the
        # coordinate system of the ellipsoid. The point q is a point in that
        # planar (2D) coordinate system.

        # If point_node is not outside of ellipsoid occluding contour is not
        # defined.
        point_inEllipsoid = point_node.get_coordinates_inOther(self)

        return \
            self.compute_occluding_contour_inEllipsoid(
                point_inEllipsoid, tol=tol, name=name
            )

    def sample_level_sets(
        self,
        min_level=0.0,
        max_level=None,
        num_level_sets=7
    ):
        """
        Sample elliptical level sets orthogonal to the z axis of the
        ellipsoid.

        Args:
            min_level (torch.Tensor, optional): minimum level of level sets
            along z axis of ellipsoid. It must be between -c and c, where c is
            the length of the z semi-axis. Defaults to core.T0.

            max_level (torch.Tensor, optional): maximum level of level sets
            along z axis of ellipsoid. It must be between -c and c, where c is
            the length of the z semi-axis. Defaults to c.

            num_level_sets (int, optional): number of level sets to sample.
            Defaults to 7.

            num_points_per_level_set (int, optional): number of points per
            level set. Defaults to 30.
        """
        # Define planes along the ellipsoid axis
        if max_level is None:
            max_level = self.shape_parameters[-1].item()
        min_level = max(min_level, -self.shape_parameters[-1].item())

        # Add one to the number of levels so as not to create a degenerate
        # contour at the apex.
        levels = torch.linspace(min_level, max_level, num_level_sets + 1)

        occluding_contours = []
        for level in levels[:-1]:  # Do not use last, extra level.
            # Create plane.
            normal_inEllipsoid = \
                torch.sign(level) * torch.tensor([0.0, 0.0, 1.0])
            homogeneous_plane_inEllipsoid = \
                torch.hstack((normal_inEllipsoid, -torch.abs(level).view(1)))

            # Create point
            point_inEllipsoid = \
                self.compute_polar_point_to_plane_inEllipsoid(
                    homogeneous_plane_inEllipsoid
                )

            # Create level set.
            contour = \
                self.compute_occluding_contour_inEllipsoid(
                    point_inEllipsoid
                )
            occluding_contours = occluding_contours + [contour, ]

        return occluding_contours

    def reflect_from_origin_and_direction_inEllipsoid(
        self, origin_inEllipsoid, direction_inEllipsoid
    ):
        """
        Reflect ray starting from origin and with given direction, bot in
        the coordinate system of the ellipsoid.

        Args:
            origin_inEllipsoid (torch.Tensor): (3,) tensor representing
            coordinates of origin of ray in the coordinate system of the
            ellipsoid.

            direction_inEllipsoid (torch.Tensor): (3,) tensor representing
            coordinates of direction of ray in the coordinate system of the
            ellipsoid.
        """
        intersection_inEllipsoid, _ = \
            self.intersect_from_origin_and_direction_inEllipsoid(
                origin_inEllipsoid, direction_inEllipsoid
            )

        if intersection_inEllipsoid is None:
            return origin_inEllipsoid, direction_inEllipsoid, False

        # Find the tangent plane at the intersection
        homogeneous_plane_inEllipsoid = \
            self.compute_polar_plane_to_point_inEllipsoid(
                intersection_inEllipsoid
            )

        # Flip the origin of the ray with respect to the plane's normal.
        normal_ray_inEllipsoid = \
            core.normalize(homogeneous_plane_inEllipsoid[:3])
        projection_length = \
            normal_ray_inEllipsoid @ \
            (origin_inEllipsoid - intersection_inEllipsoid)
        projection_along_normal_inEllipsoid = \
            projection_length * normal_ray_inEllipsoid + \
            intersection_inEllipsoid
        flipped_origin_inEllipsoid = \
            projection_along_normal_inEllipsoid + \
            (projection_along_normal_inEllipsoid - origin_inEllipsoid)
        reflection_direction_inEllipsoid = \
            flipped_origin_inEllipsoid - intersection_inEllipsoid

        return intersection_inEllipsoid, reflection_direction_inEllipsoid, True

    def reflect_from_origin_and_direction_inParent(
        self, origin_inParent, direction_inParent
    ):
        """
        Reflect ray starting from origin and with given direction, both in
        the coordinate system of the ellipsoid's parent.

        Args:
            origin_inParent (torch.Tensor): (3,) tensor representing
            coordinates of origin of ray in the coordinate system of the
            ellipsoid's parent.

            direction_inParent (torch.Tensor): (3,) tensor representing
            coordinates of direction of ray in the coordinate system of the
            ellipsoid's parent.

        Returns:
            tuple of torch.Tensor: (2,) tuple of (3,) tensors corresponding to
            coordinates of reflection point and direction of reflected ray in
            the coordinate system of the ellipsoid's parent.

            If the ray does not intersect with the ellipsoid, return the origin
            and direction inputs.
        """
        origin_inEllipsoid = \
            self.transform_toParent_fromSelf.inverse_transform(origin_inParent)
        direction_inEllipsoid = \
            self.transform_toParent_fromSelf.rotation.inverse_transform(
                direction_inParent
            )

        reflection_origin_inEllipsoid, \
            reflection_direction_inEllipsoid, \
            _ = \
            self.reflect_from_origin_and_direction_inEllipsoid(
                origin_inEllipsoid, direction_inEllipsoid
            )

        # Transform the results to the coordinate system of the ellipsoid's
        # parent.
        reflection_origin_inParent = \
            self.transform_toParent_fromSelf.transform(
                reflection_origin_inEllipsoid
            )
        reflection_direction_inParent = \
            self.transform_toParent_fromSelf.rotation.transform(
                reflection_direction_inEllipsoid
            )

        return reflection_origin_inParent, reflection_direction_inParent

    def reflect_ray(self, ray_node):
        """reflect_ray.

        Reflect ray on surface of ellipsoid. If the ray does not hit the
        surface, return the input ray

        Args:
            ray_node (Ray): ray that may reflect off the surface of the
            ellipsoid.

        Returns:
            Ray: reflected ray, or, if the input ray does not hit the surface
            of the ellipsoid, the input ray.
        """
        # Find the intersection of the ray with the ellipsoid.
        intersection_inEllipsoid, _ = self.intersect_ray_inEllipsoid(ray_node)

        if intersection_inEllipsoid is None:
            return ray_node

        # Find the tangent plane at the intersection
        homogeneous_plane_inEllipsoid = \
            self.compute_polar_plane_to_point_inEllipsoid(
                intersection_inEllipsoid
            )

        # Flip the origin of the ray with respect to the plane's normal.
        normal_ray = primitives.Ray.create_from_origin_and_dir_inParent(
            self, intersection_inEllipsoid, homogeneous_plane_inEllipsoid[:3]
        )
        origin_inEllipsoid, _ = \
            ray_node.get_origin_and_direction_inOther(self)
        closest_inEllipsoid = \
            normal_ray.project_to_ray_inParent(origin_inEllipsoid)

        # Clear house: remove normal ray from pose graph.
        self.remove_child(normal_ray)

        flipped_inEllipsoid = \
            closest_inEllipsoid + (closest_inEllipsoid - origin_inEllipsoid)
        direction_inEllipsoid = flipped_inEllipsoid - intersection_inEllipsoid

        return \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self, intersection_inEllipsoid, direction_inEllipsoid
            )

    def refract_from_origin_and_direction_inEllipsoid(
        self,
        origin_inEllipsoid,
        direction_inEllipsoid,
        eta_at_origin=core.T1.clone(),
        eta_at_destination=core.TETA.clone()
    ):
        """
        Refract ray on surface of ellipsoid. If the ray does not hit the
        ellipsoid, return the input origin and direction.

        Args:
            origin_inEllipsoid (torch.Tensor): (3,) tensor with coordinates of
            origin of ray in ellipsoid's coordinate system.

            direction_inEllipsoid (torch.Tensor): (3,) tensor with coordinates
            of direction of ray in ellipsoid's coordinate system.

            eta_at_origin (torch.float, optional): refractive index of medium
            at origin. Defaults to 1.

            eta_at_destination (torch.float, optional): refractive index of
            medium at destination. Defaults to core.TETA.
        """
        intersection_inEllipsoid, _ = \
            self.intersect_from_origin_and_direction_inEllipsoid(
                origin_inEllipsoid, direction_inEllipsoid
            )

        if intersection_inEllipsoid is None:
            return origin_inEllipsoid, direction_inEllipsoid, False

        # Follow Snell's law to get the refracted ray via the diagram below.
        #
        #                  normal-> |  / <-refracted_dir
        #                           | /
        #      eta_2                |/ <-theta_2
        # ----Surface----------------------------------------- eta_1 /|
        #      <-theta_1 /   | dir_normalized ->  /      |
        #
        # Write
        #
        # refracted_dir = alpha * dir_normalized + beta * normal, and solve:
        #
        # |refracted_dir|**2 = 1
        # dir_normalized.T * normal = cos(theta_1) refracted_dir.T * normal =
        # cos(theta_2)

        inner_normal_inEllipsoid = \
            -self.compute_polar_plane_to_point_inEllipsoid(
                intersection_inEllipsoid
            )[:3]
        inner_normal_inEllipsoid = core.normalize(inner_normal_inEllipsoid)
        direction_inEllipsoid = core.normalize(direction_inEllipsoid)

        alpha = eta_at_origin / eta_at_destination
        cos_theta_1 = \
            torch.dot(direction_inEllipsoid, inner_normal_inEllipsoid)

        discriminant = 1.0 - alpha**2 * (1.0 - cos_theta_1**2)
        if discriminant < 0:
            # No refraction (angle too high). Total internal reflection.
            return \
                self.reflect_from_origin_and_direction_inEllipsoid(
                    origin_inEllipsoid, direction_inEllipsoid
                )

        beta = -alpha * cos_theta_1 + torch.sqrt(discriminant)
        refracted_direction_inEllipsoid = \
            alpha * direction_inEllipsoid + beta * inner_normal_inEllipsoid

        return intersection_inEllipsoid, refracted_direction_inEllipsoid, True

    def refract_from_origin_and_direction_inParent(
        self,
        origin_inParent,
        direction_inParent,
        eta_at_origin=core.T1.clone(),
        eta_at_destination=core.TETA.clone()
    ):
        """
        Refract ray from origin and direction of ray in coordinate system of
        ellipsoid's parent.

        Args:
            origin_inParent (torch.Tensor): (3,) tensor with coordinates of
            ray's origin in coordinate system of ellipsoid's parent.

            direction_inParent (torch.Tensor): (3,) tensor with coordinates of
            ray's direction in coordinate system of ellipsoid's parent.

            eta_at_origin (torch.float, optional): refractive index of medium
            at origin. Defaults to core.TETA.

            eta_at_destination (torch.float, optional): refractive index of
            medium at destination. Defaults to core.T1.
        """
        origin_inEllipsoid = \
            self.transform_toParent_fromSelf.inverse_transform(origin_inParent)
        direction_inEllipsoid = \
            self.transform_toParent_fromSelf.rotation.inverse_transform(
                direction_inParent
            )

        refraction_origin_inEllipsoid, \
            refraction_direction_inEllipsoid, \
            success = \
            self.refract_from_origin_and_direction_inEllipsoid(
                origin_inEllipsoid,
                direction_inEllipsoid,
                eta_at_origin=eta_at_origin,
                eta_at_destination=eta_at_destination
            )

        refraction_origin_inParent = \
            self.transform_toParent_fromSelf.transform(
                refraction_origin_inEllipsoid
            )
        refraction_direction_inParent = \
            self.transform_toParent_fromSelf.rotation.transform(
                refraction_direction_inEllipsoid
            )

        return \
            refraction_origin_inParent, refraction_direction_inParent, success

    def refract_ray(
        self,
        ray_node,
        eta_at_origin=core.T1.clone(),
        eta_at_destination=core.TETA.clone()
    ):
        """
        Refract ray on surface of ellipsoid. If the ray does not hit the
        ellipsoid, return the input ray.

        Args:
            ray_node (Ray): ray to be refracted.

            eta_at_origin (torch.Tensor, optional): refractive index at the
            origin of the ray. Defaults to core.TETA.

            eta_at_destination (torch.Tensor, optional): refractive index at
            the destination of the ray. Defaults to core.T1.

        Returns:
            Ray: refracted ray, or input ray if input ray does not hit the
            ellipsoid.
        """
        origin_inEllipsoid, direction_inEllipsoid = \
            ray_node.get_origin_and_direction_inOther(self)

        refraction_origin_inEllipsoid, \
            refraction_direction_inEllipsoid, \
            _ = \
            self.refract_from_origin_and_direction_inEllipsoid(
                origin_inEllipsoid,
                direction_inEllipsoid,
                eta_at_origin=eta_at_origin,
                eta_at_destination=eta_at_destination
            )

        # Create the ray.
        ray = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self,
                refraction_origin_inEllipsoid,
                refraction_direction_inEllipsoid
            )

        return ray

    # Contrary to norm but in order to simplify the notation we do not add the
    # suffix _inEllipsoid to indicate the coordinate system of the geometric
    # primitivess, to simplify the notation. We keep it on function names,
    # though! Note also that rather than creating a node in the pose graph,
    # which is coordinate system independent, we output coordinates of the
    # reflection point in the coordinate system of the ellipsoid. The reason
    # for this distinction is that nodes are permanent, and remain attached to
    # the pose graph unless explicitly removed, whereas points are ephemeral,
    # lasting for only as long as we need them.
    def _compute_reflection_point_no_grad_inEllipsoid(
        self, origin, destination, tol=core.TEPS * 100
    ):
        """
        Compute the point on the surface of the ellipsoid such that a ray
        originating on point origin will reflect and go through point
        destination.

        The approach consists into finding an interpolating point along the
        origin and destination in the form (origin * (1 - t) + destination * t)
        with t in [0, 1], for which the normal of the ellipsoid at the
        orthogonal projection of the interpolating point onto the ellipsoid
        bisects the angle formed by the vectors (origin - projection) and
        (destination - projection).

        Args:
            origin (torch.Tensor): (3,) torch tensor representing coordinates
            of origin of reflected ray in the coordinate system of the
            ellipsoid.

            destination (torch.Tensor): (3,) torch tensor representing
            coordinates of destination of reflected ray in the coordinate
            system of the ellipsoid.

            tol (torch.Tensor, optional): tolerance of fixed point iteration.
            Defaults to core.TEPS*100.

        Returns:
            torch.Tensor: point on the surface of ellipsoid at which an
            incident ray departing from the origin will be reflected towards
            the destination.
        """

        #
        #  origin               interpolator
        #     * __________[t0_____t*______t1]____ * destination \
        #      /            _/ \               /          _/ \     normal ^
        #        _/ \         /      _/ \      /    _/ \   /   / \ reflection
        #          _____*________ /                \
        #           /    ellipsoid     \
        #

        with torch.no_grad():
            # Initialize variables. Bracket the interpolating parameter t.
            bracket = torch.tensor([0.0, 1.0])
            delta = core.T1.clone() + tol
            projection = None
            while delta > tol:
                # We shrink the width of the bracket by a factor of 2 at each
                # iteration of the while loop. This is fast.
                t = torch.mean(bracket)
                interpolator = (1 - t) * origin + t * destination

                # Projection of interpolator onto surface of ellipsoid. We
                # iterate until this projection converges to the reflection
                # point.
                projection = self.project_on_surface_inEllipsoid(interpolator)

                # The next blocks ending on a break take care of edge cases.
                # They should not happen, having been taken care of when we
                # tested whether the ray from origin to destination is tangent
                # to the ellipsoid. But this is an numerically unstable
                # condition, so we check for it again inside the loop.

                origin_minus_projection = origin - projection
                norm_origin_minus_projection = torch.linalg.norm(
                    origin_minus_projection)

                # If norm_projection_minus_origin is zero, we are in a
                # degenerate situation where origin is on the ellipsoid
                # surface, and therefore the result is origin.
                if norm_origin_minus_projection < tol:
                    projection = origin
                    break

                destination_minus_projection = destination - projection
                norm_destination_minus_projection = torch.linalg.norm(
                    destination_minus_projection)

                # If norm_projection_minus_destination is zero, we are in a
                # degenerate situation where destination is on the ellipsoid
                # surface, and therefore the result is destination.
                if norm_destination_minus_projection < tol:
                    projection = destination
                    break

                # Compute the normalized vectors (origin - projection) and
                # (destination - projection).
                origin_minus_projection_norm = \
                    origin_minus_projection / norm_origin_minus_projection

                destination_minus_projection_norm = \
                    destination_minus_projection / \
                    norm_destination_minus_projection

                # Compute the normal to the ellipsoid at the projection point.
                # Because we have projected the interpolator onto the
                # ellipsoid, this is simply (interpolator - projection). The
                # magnitude of this vector is not relevant.
                interpolator_minus_projection = interpolator - projection

                # Compute the scaled cosine of the angle between (interpolator
                # - projection) and (projection - origin), and the scaled
                # cosine of the angle between (interpolator - projection) and
                # (projection - origin). The scale factor is common to the two
                # computations, so its value is irrelevant.
                cos_origin = interpolator_minus_projection.dot(
                    origin_minus_projection_norm)
                cos_destination = interpolator_minus_projection.dot(
                    destination_minus_projection_norm)

                # We have two stopping criteria. First, we use the difference
                # between the angles. This helps when we get a good
                # projection/interpolator, but the bracket is still large.
                if torch.abs(cos_destination - cos_origin) < tol:
                    break

                # If the angle with respect to destination is smaller than that
                # with respect to origin, the interpolating point is between t0
                # and the current value t. Otherwise, the interpolating point
                # is between the current value t and t1.
                if cos_destination > cos_origin:
                    bracket = core.stack_tensors(
                        (bracket[0].view(1), t.view(1)))
                else:
                    bracket = core.stack_tensors(
                        (t.view(1), bracket[1].view(1)))

                delta = (bracket[1] - bracket[0])

            return projection

    class _compute_reflection_point_internal(torch.autograd.Function):
        """_compute_reflection_point_internal.

        Because the computation of the reflection point is the output of an
        iterative process, the automatic computation of derivatives does not
        work. Therefore, we implement the derivatives ourselves.
        """

        @staticmethod
        def forward(
            ctx,
            origin,
            destination,
            shape_parameters,
            ellipsoid_node,
            tol=core.TEPS * 100
        ):
            """
            Computation of reflection point as an autograd function, for
            correct propagation of derivatives.

            Assume inputs are in the the coordinate system of the ellipsoid. We
            should also benefit from the stable iterative method we have
            already implemented.

            We explicitly provide the diagonal of the ellipsoid's shape
            parameters because the ellipsoid is not a torch.Tensor, as required
            by the forward and backward methods.

            Args:
                ctx (context): structure for saving computations needed for for
                computation of gradient.

                origin (torch.Tensor): origin of reflected ray in coordinate
                system of the ellipsoid.

                destination (torch.Tensor): destination of reflected ray in
                coordinate system of the ellipsoid.

                shape_parameters (torch.Tensor): half lengths of semi axes of
                ellipsoid. This is provided as a separate parameter because we
                can compute derivatives only with respect to tensors, and the
                ellipsoid is not a tensor but a node.

                ellipsoid_node (Ellipsoid): ellipsoid node on which reflection
                point is computed.

                tol (torch.Tensor, optional): tolerance of fixed-point
                iteration for computation of reflection point. Defaults to
                core.TEPS*100.

            Returns:
                torch.Tensor: reflection point represented as a (3,) torch
                Tensor in the coordinate system of the ellipsoid.
            """
            reflection = \
                ellipsoid_node._compute_reflection_point_no_grad_inEllipsoid(
                    origin, destination, tol=tol
                )

            ctx.save_for_backward(
                reflection, origin, destination, shape_parameters
            )

            # Since we pass on the ellipsoid, let's save it too to save some
            # computation.
            ctx.ellipsoid_node = ellipsoid_node

            return reflection

        @staticmethod
        def backward(ctx, grad_output):
            """backward.

            Should return:
                grad_output @ d_reflection_d_origin grad_output
                grad_output @ d_reflection_d_destination grad_output
                grad_output @ d_reflection_d_shape_parameters
                None (no gradient with respect to ellipsoid)
                None (no gradient with respect to tol)
            """
            # We perform computations in the ellipse coordinate system, and
            # then move the results to the world coordinate system if required.

            # Recover the required data.
            reflection, origin, destination, shape_parameters = \
                ctx.saved_tensors
            s = ctx.ellipsoid_node.diagonal

            # Algebra
            #
            # The ellipsoid is the locus of points x such that x.T * A * x = 1
            #
            # s := shape_matrix_diag = diag(A) (from matrix to vector) x1 :=
            # origin_inEllipsoid x2 := destination_inEllipsoid
            #
            # The solution is x = x(s, x1, x2) such that
            #
            # x = A^(-1) * y / x.dot(y),
            #
            # where
            #
            # y = y(x, x1, x2) = (x - x1)/norm(x - x1) + (x - x2)/norm(x - x2).
            #
            # x.dot(y) will not be zero unless we are in a degenerate
            # configuration.

            A = torch.diag(s)
            x = reflection
            x1 = origin
            x2 = destination

            x_minus_x1 = x - x1
            x_minus_x2 = x - x2
            norm_x_minus_x1 = torch.linalg.norm(x_minus_x1)
            norm_x_minus_x2 = torch.linalg.norm(x_minus_x2)
            x_minus_x1_norm = x_minus_x1 / norm_x_minus_x1
            x_minus_x2_norm = x_minus_x2 / norm_x_minus_x2

            y = x_minus_x1_norm + x_minus_x2_norm

            # Calculus
            #
            # We compute the total derivative of F(x, s, y) = x - A^(-1) * y /
            # x.dot(y).
            #
            # We must remember that x = x(x, x1, x2), and y = y(x, x1, x2).

            x_outer_y = torch.outer(x, y)
            x_dot_y = x.dot(y)
            Id = torch.eye(3)

            dy_dx1 = -(Id - torch.outer(x_minus_x1_norm,
                       x_minus_x1_norm)) / norm_x_minus_x1
            dy_dx2 = -(Id - torch.outer(x_minus_x2_norm,
                       x_minus_x2_norm)) / norm_x_minus_x2

            # Better names?
            x_outer_y_norm = x_outer_y / x_dot_y
            aux = (Id - x_outer_y_norm.T) / x_dot_y
            aux_x1 = aux.mm(dy_dx1)
            aux_x2 = aux.mm(dy_dx2)
            aux_x = aux_x1 + aux_x2
            aux_s = -torch.diag(x)

            dF_dx = A.mm(Id + x_outer_y_norm) + aux_x
            inv_dF_dx = torch.inverse(dF_dx)
            dx_dx1 = inv_dF_dx.mm(aux_x1)
            dx_dx2 = inv_dF_dx.mm(aux_x2)
            dx_ds = inv_dF_dx.mm(aux_s)

            # Let y' = T * y and x' = T * x. Then
            #
            # dy'_dx' = dy'_dy * dy_dx * dx_dx' =      T * dy_dx * T^(-1) =
            #         (T^(-t) * (T * dy_dx)^(t))^(t)
            #
            d_reflection_d_origin = dx_dx1
            d_reflection_d_destination = dx_dx2
            d_reflection_d_diagonal = dx_ds

            # d_(1/a^2, 1/b^2, 1/c^2)_d(a, b, c) = (-2/a^3, -2/b^3, -2/c^3)
            d_diagonal_d_shape_parameters = \
                -2 * torch.diag(s / shape_parameters)
            d_reflection_d_shape_parameters = \
                d_reflection_d_diagonal @ d_diagonal_d_shape_parameters

            grad_origin = grad_output @ d_reflection_d_origin
            grad_destination = grad_output @ d_reflection_d_destination
            grad_shape_parameters = \
                grad_output @  d_reflection_d_shape_parameters

            return \
                grad_origin, \
                grad_destination, \
                grad_shape_parameters, \
                None, \
                None

    # This is the public API which correctly propagates gradients.
    def compute_reflection_point_inEllipsoid(
        self, origin, destination, tol=core.TEPS * 100
    ):
        return \
            self._compute_reflection_point_internal.apply(
                origin, destination, self.shape_parameters, self, tol
            )

    def _iteration_for_refraction(
        self,
        refraction,
        origin,
        destination,
        eta_at_origin,
        eta_at_destination
    ):
        """
        Iterative function used in computation of refraction point through a
        fixed-point iteration.

        Args:
            refraction (torch.Tensor): (3,) tensor corresponding to current
            value of refraction point.

            origin (torch.Tensor): (3,) origin of refracted ray in coordinate
            system of ellipsoid.

            destination (torch.Tensor): (3,) destination of refracted ray in
            coordinate system of ellipsoid.

            eta_at_origin (torch.float): refractive index of medium at origin.

            eta_at_destination (torch.float): refractive index of medium at
            destination.

        Returns:
            torch.Tensor: (3,) corresponding to refined value of refraction
            point in coordinate system of ellipsoid.
        """
        in_ray_norm = torch.linalg.norm(refraction - origin)
        out_ray_norm = torch.linalg.norm(destination - refraction)
        eta_at_destination_times_in_ray_norm = \
            eta_at_destination * in_ray_norm
        t = \
            eta_at_destination_times_in_ray_norm / \
            (
                eta_at_destination_times_in_ray_norm +
                eta_at_origin * out_ray_norm
            )
        interpolator = (1 - t) * origin + t * destination

        return self.project_on_surface_inEllipsoid(interpolator)

    def _compute_refraction_point_no_grad_inEllipsoid(
        self,
        origin,
        destination,
        eta_at_origin,
        eta_at_destination,
        tol,
        max_num_iter
    ):
        """
        Internal method for computation of refraction point in ellipsoid.
        This method utilizes a fixed-point iteration, therefore its computation
        of gradient is incorrect, and we deactivate it anyway.

        Args:
            origin (torch.Tensor): (3,) tensor representing coordinates of
            origin of refracted ray in coordinate system of ellipsoid.

            destination (torch.Tensor): (3,) tensor representing destination of
            refracted ray in coordinate system of ellipsoid.

            eta_at_origin (torch.float): Refractive index of medium at origin.

            eta_at_destination (torch.float): Refractive index of medium at
            destination.TETA.

            tol (torch.float): tolerance in fixed-point iteration. Stopping
            criterion is whether norm of refraction point changes by less than
            tol.

            max_num_inter (int): maximum number of fixed-point iterations.

        Returns:
            torch.Tensor: (3,) tensor corresponding to coordinates of
            refraction point in the coordinate system of the ellipsoid.
        """
        with torch.no_grad():
            # Initialization
            t = torch.tensor(0.5)
            interpolator = (1 - t) * origin + t * destination
            refraction = self.project_on_surface_inEllipsoid(interpolator)

            # Fixed point iteration.
            delta = core.T1.clone()
            iter = 0
            while delta > tol and iter < max_num_iter:
                new_refraction = \
                    self._iteration_for_refraction(
                        refraction,
                        origin,
                        destination,
                        eta_at_origin,
                        eta_at_destination
                    )

                iter += 1
                delta = torch.linalg.norm(new_refraction - refraction)
                refraction = new_refraction

            return refraction

    class _compute_refraction_point_internal(torch.autograd.Function):
        """_compute_refraction_point_internal.

        Because the computation of the refraction point is the output of an
        iterative process, the automatic computation of derivatives does not
        work. Therefore, we implement the derivatives ourselves.
        """
        @staticmethod
        def forward(
            ctx,
            origin,
            destination,
            shape_parameters,  # Dummy var needed for the gradient signature.
            eta_at_origin,
            eta_at_destination,
            ellipsoid_node,
            tol,
            max_num_iter,
            create_graph
        ):
            """
            Computation of refraction point as an autograd function, for
            correct propagation of derivatives.

            Assume inputs are in the coordinate system of the ellipsoid. We
            should also benefit from the stable iterative method we have
            already implemented.

            We explicitly provide the shape parameters of the ellipsoid because
            the ellipsoid is not a torch.Tensor, as required by the forward and
            backward methods.

            Args:
                ctx (context): structure for saving computations needed for for
                computation of gradient.

                origin (torch.Tensor): origin of refracted ray in coordinate
                system of the ellipsoid.

                destination (torch.Tensor): destination of refracted ray in
                coordinate system of the ellipsoid.

                shape_parameters (torch.Tensor): half lengths of semi axes of
                ellipsoid. This is provided as a separate parameter because we
                can compute derivatives only with respect to tensors, and the
                ellipsoid is not a tensor but a node.

                ellipsoid_node (Ellipsoid): ellipsoid node on which refraction
                point is computed.

                eta_at_origin (torch.float): refractive index of medium at
                origin.

                eta_at_destination (torch.float): refractive index of medium at
                destination.

                tol (torch.Tensor): tolerance of fixed-point iteration for
                computation of refraction point. Defaults to core.TEPS*100.

                max_num_iter (int): maximum number of fixed-point iterations.

                create_graph (bool): if True, derivatives are added to
                computational graph. Useful for computation of Hessians.

            Returns:
                torch.Tensor: (3,) torch tensor corresponding to coordinates of
                refraction point in coordinate system of ellipsoid.
            """
            refraction = \
                ellipsoid_node._compute_refraction_point_no_grad_inEllipsoid(
                    origin,
                    destination,
                    eta_at_origin,
                    eta_at_destination,
                    tol,
                    max_num_iter
                )

            ctx.save_for_backward(
                refraction,
                origin,
                destination,
                shape_parameters,
                eta_at_origin,
                eta_at_destination
            )
            ctx.ellipsoid_node = ellipsoid_node
            ctx.create_graph = create_graph

            return refraction

        @staticmethod
        def backward(ctx, grad_output):
            """backward.

            Should return:
                grad_output @ d_refraction_d_origin grad_output @
                d_refraction_d_destination grad_output @ d_refraction_d_shape
                grad_output @ d_refraction_d_eta_at_origin grad_output @
                d_refraction_d_eta_at_destination None (ellipsoid node is not a
                torch tensor) None (no gradient with respect to tolerance) None
                (no gradient with respect to number of iterations) None (no
                gradient with respect to boolean flag)

            Args:
                ctx (context): context with saved data from forward method.

                grad_output (torch.Tensor): derivative with respect to
                refraction, for gradient propagation
            """
            # The refraction point is defined implicitly by a function
            #
            # refraction = f(refraction, origin, destination, shape, indices).
            # Thus,
            #
            # d_refraction_d_origin = d_f_d_refraction * d_refraction_d_origin
            #   + d_f_d_origin
            #
            # d_refraction_d_dest = d_f_d_refraction * d_refraction_d_dest +
            #   d_f_d_dest
            #
            # d_refraction_d_shape = d_f_d_refraction * d_refraction_d_shape +
            #   d_f_d_shape
            #
            # d_refraction_d_index = d_f_d_refraction * d_refraction_d_index +
            #   d_f_d_index
            refraction, \
                origin, \
                destination, \
                _, \
                eta_at_origin, \
                eta_at_destination = \
                ctx.saved_tensors

            ellipsoid_node = ctx.ellipsoid_node
            create_graph = ctx.create_graph

            # Forward computation.
            fresh_refraction = refraction.clone().detach()
            fresh_refraction.requires_grad = True
            with torch.enable_grad():
                refraction_ = \
                    ellipsoid_node._iteration_for_refraction(
                        fresh_refraction,
                        origin,
                        destination,
                        eta_at_origin,
                        eta_at_destination
                    )

            d_f_d_refraction = \
                core.compute_auto_jacobian_from_tensors(
                    refraction_,
                    fresh_refraction,
                    create_graph=create_graph  # For second derivatives
                )
            tmp = torch.eye(3) - d_f_d_refraction

            if origin.requires_grad:
                d_f_d_origin = \
                    core.compute_auto_jacobian_from_tensors(
                        refraction_,
                        origin,
                        create_graph=create_graph  # For second derivatives
                    )
                d_refraction_d_origin = \
                    grad_output @ torch.linalg.solve(tmp, d_f_d_origin)
            else:
                d_refraction_d_origin = None

            if destination.requires_grad:
                d_f_d_destination = \
                    core.compute_auto_jacobian_from_tensors(
                        refraction_,
                        destination,
                        create_graph=create_graph  # For second derivatives
                    )
                d_refraction_d_destination = \
                    grad_output @ torch.linalg.solve(tmp, d_f_d_destination)
            else:
                d_refraction_d_destination = None

            if ellipsoid_node.shape_parameters.requires_grad:
                d_f_d_shape = \
                    core.compute_auto_jacobian_from_tensors(
                        refraction_,
                        ellipsoid_node.shape_parameters,
                        create_graph=create_graph  # For second derivatives
                    )
                d_refraction_d_shape = grad_output @ \
                    torch.linalg.solve(tmp, d_f_d_shape)
            else:
                d_refraction_d_shape = None

            if eta_at_origin.requires_grad:
                d_f_d_eta_at_origin = \
                    core.compute_auto_jacobian_from_tensors(
                        refraction_,
                        eta_at_origin,
                        create_graph=create_graph  # For second derivatives
                    )
                d_refraction_d_eta_at_origin = \
                    grad_output @ \
                    torch.linalg.solve(tmp, d_f_d_eta_at_origin.view(-1, 1))
            else:
                d_refraction_d_eta_at_origin = None

            if eta_at_destination.requires_grad:
                d_f_d_eta_at_destination = \
                    core.compute_auto_jacobian_from_tensors(
                        refraction_,
                        eta_at_destination,
                        create_graph=create_graph  # For second derivatives
                    )
                d_refraction_d_eta_at_destination = \
                    grad_output @ \
                    torch.linalg.solve(
                        tmp, d_f_d_eta_at_destination.view(-1, 1)
                    )
            else:
                d_refraction_d_eta_at_destination = None

            return \
                d_refraction_d_origin, \
                d_refraction_d_destination, \
                d_refraction_d_shape, \
                d_refraction_d_eta_at_origin, \
                d_refraction_d_eta_at_destination, \
                None, \
                None, \
                None, \
                None

    def compute_refraction_point_inEllipsoid(
        self,
        origin,
        destination,
        eta_at_origin=core.T1.clone(),
        eta_at_destination=core.TETA.clone(),
        tol=core.TEPS * 100,
        max_num_iter=10,
        create_graph=False
    ):
        """
        Computation of refraction point in ellipsoid, with correct
        propagation of gradients.

        Args:
            origin (torch.Tensor): (3,) tensor representing coordinates of
            origin of refracted ray in coordinate system of ellipsoid.

            destination (torch.Tensor): (3,) tensor representing destination of
            refracted ray in coordinate system of ellipsoid.

            eta_at_origin (torch.float, optional): Refractive index of medium
            at origin. Defaults to core.TETA.

            eta_at_destination (torch.float, optional): Refractive index of
            medium at destination. Defaults to core.T1.

            tol (torch.float, optional): tolerance in fixed-point iteration.
            Stopping criterion is whether norm of refraction point changes by
            less than tol. Defaults to core.TEPS.

            max_num_iter (int, optional): maximum number of fixed-point
            iterations. Defaults to 10.

            create_graph (bool, optional): if True, adds derivatives of
            function to computational graph. Useful for computation of
            Hessians.

        Returns:
            torch.Tensor: (3,) tensor corresponding to coordinates of
            refraction point in the coordinate system of the ellipsoid.
        """
        return \
            self._compute_refraction_point_internal.apply(
                origin,
                destination,
                self.shape_parameters,
                eta_at_origin,
                eta_at_destination,
                self,
                tol,
                max_num_iter,
                create_graph
            )
