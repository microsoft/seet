"""Unit tests from methods and classes in plane.py.

Unit tests from methods and classes in plane.py
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.primitives as primitives
from tests import tests_utils
import torch
import unittest


class TestEllipsoid(tests_utils.TestUtils):
    shape_parameters = torch.tensor([1.0, 2.0, 3.0])

    def create_point_inEllipsoid(self, ellipsoid_node, requires_grad=False):
        """create_point_inEllipsoid.

        Create a point on the input ellipsoid to facilitate testing.

        Args:
            ellipsoid_node (Ellipsoid): ellipsoid in which to create a point.

        Returns:
            torch.Tensor: (3,) tensor representing point on ellipsoid surface
            in ellipsoid's coordinate system.
        """

        # Create a point on the surface of the ellipsoid.
        a, b, c = ellipsoid_node.shape_parameters
        # Point in first octant.
        theta = core.deg_to_rad(13.0)  # Random angle.
        psi = core.deg_to_rad(37.0)    # Random angle.
        x = a * torch.cos(theta) * torch.cos(psi)
        y = b * torch.sin(theta) * torch.cos(psi)
        z = c * torch.sin(psi)

        return torch.tensor((x, y, z), requires_grad=requires_grad)

    def setUp(self):
        super().setUp()

        nodes = self.root.traverse()
        for parent_node in nodes:
            primitives.Ellipsoid(
                parent_node,
                self.transform,
                self.shape_parameters,
                name=f"ellipsoid in {parent_node.name}"
            )

    def test_init(self):
        """test_init.

        Test instantiation of new Ellipsoid object.
        """

        # Get an arbitrary ellipsoid.
        parent_node = self.root.children[0].children[0]
        ellipsoid_node = parent_node.children[-1]

        self.assertTrue(
            torch.allclose(
                ellipsoid_node.shape_parameters**2,
                1 / ellipsoid_node.diagonal
            )
        )

        shape_parameters = torch.tensor([3.0, 1.0, 2.0])
        ellipsoid_node.reset_shape(shape_parameters)

        self.assertTrue(
            torch.allclose(ellipsoid_node.shape_parameters, shape_parameters)
        )
        self.assertTrue(
            torch.allclose(
                ellipsoid_node.shape_parameters**2,
                1 / ellipsoid_node.diagonal
            )
        )

    def test_get_ellipsoid_matrix(self):
        """test_get_ellipsoid_matrix.

        Test creation of homogeneous matrix of ellipsoid.
        """

        # Get an arbitrary ellipsoid.
        parent_node = self.root.children[0].children[0]
        ellipsoid_node = parent_node.children[-1]

        Q = ellipsoid_node.get_ellipsoid_matrix_inEllipsoid()
        X_inEllipsoid = self.create_point_inEllipsoid(ellipsoid_node)
        X_h = core.homogenize(X_inEllipsoid)

        self.assertTrue(torch.allclose(X_h @ Q @ X_h, core.T0))

        Q = ellipsoid_node.get_ellipsoid_matrix_inOther(parent_node)
        transform_toParent_fromEllipsoid = \
            ellipsoid_node.get_transform_toParent_fromSelf()
        X_inParent = transform_toParent_fromEllipsoid.transform(X_inEllipsoid)
        X_h = core.homogenize(X_inParent)

        # Must relax pass criterion.
        self.assertTrue(torch.allclose(X_h @ Q @ X_h, core.T0, atol=1e-3))

    def test_intersect_from_origin_and_direction_inEllipsoid(self):
        """test_intersect_from_origin_and_direction_inEllipsoid.

        Test computation of intersection with ellipsoid of ray defined by
        origin and direction in coordinate system of ellipsoid.
        """

        # Get an arbitrary ellipsoid.
        parent_node = self.root
        ellipsoid_node = parent_node.children[-1]

        intersection_inEllipsoid = \
            self.create_point_inEllipsoid(ellipsoid_node)

        directions_inEllipsoid = torch.eye(3)
        for single_direction_inEllipsoid in directions_inEllipsoid:
            intersection_inEllipsoid_, _ = \
                ellipsoid_node.intersect_from_origin_and_direction_inEllipsoid(
                    intersection_inEllipsoid, single_direction_inEllipsoid
                )

            self.assertTrue(
                torch.allclose(
                    intersection_inEllipsoid,
                    intersection_inEllipsoid_,
                    rtol=core.EPS * 100,
                    atol=core.EPS * 100
                )
            )

            # Using the same ray, create another ray with a different origin.
            intersection_inEllipsoid_, _ = \
                ellipsoid_node.intersect_from_origin_and_direction_inEllipsoid(
                    intersection_inEllipsoid +
                    10 * single_direction_inEllipsoid,
                    single_direction_inEllipsoid
                )

            self.assertTrue(
                torch.allclose(
                    intersection_inEllipsoid,
                    intersection_inEllipsoid_,
                    rtol=core.EPS * 100,
                    atol=core.EPS * 100
                )
            )

    def test_intersect_ray_inEllipsoid(self):
        """test_intersect_ray_inEllipsoid.

        Test intersection of ray with ellipsoid.
        """

        # Get an arbitrary ellipsoid.
        parent_node = self.root
        ellipsoid_node = parent_node.children[-1]

        intersection_inEllipsoid = \
            self.create_point_inEllipsoid(ellipsoid_node)

        directions_inEllipsoid = torch.eye(3)
        for single_direction_inEllipsoid in directions_inEllipsoid:
            ray = \
                primitives.Ray.create_from_origin_and_dir_inParent(
                    ellipsoid_node,
                    intersection_inEllipsoid,
                    single_direction_inEllipsoid
                )

            intersection_inEllipsoid_, _ = \
                ellipsoid_node.intersect_ray_inEllipsoid(ray)

            self.assertTrue(
                torch.allclose(
                    intersection_inEllipsoid,
                    intersection_inEllipsoid_,
                    rtol=core.EPS * 100,
                    atol=core.EPS * 100
                )
            )

            # Using the same ray, create another ray with a different origin.
            ray = \
                primitives.Ray.create_from_origin_and_dir_inParent(
                    ellipsoid_node,
                    intersection_inEllipsoid +
                    10 * single_direction_inEllipsoid,
                    single_direction_inEllipsoid
                )

            intersection_inEllipsoid_, _ = \
                ellipsoid_node.intersect_ray_inEllipsoid(ray)

            self.assertTrue(
                torch.allclose(
                    intersection_inEllipsoid,
                    intersection_inEllipsoid_,
                    rtol=core.EPS * 100,
                    atol=core.EPS * 100
                )
            )

    def test_compute_algebraic_distance(self):
        """test_compute_algebraic_distance.

        Test computation of algebraic distance. It should be negative for a
        point inside the ellipsoid, positive for a point outside the ellipsoid,
        and very close to zero for a point on the surface.
        """

        # Get an arbitrary ellipsoid.
        parent_node = self.root
        ellipsoid_node = parent_node.children[-1]

        # Get a point inside the ellipsoid
        inside_inEllipsoid = torch.zeros(3)
        point_node = \
            primitives.Point.create_from_coordinates_inParent(
                ellipsoid_node, inside_inEllipsoid, name="point"
            )

        # Attach point to root. Not needed, but prevents the test form becoming
        # trivial.
        parent_node.add_child(point_node)

        # Transform the point to some other coordinate system.
        self.assertTrue(
            ellipsoid_node.compute_algebraic_distance(point_node) <
            core.T0
        )

        # Remove point_node from tree.
        ellipsoid_node.remove_child(point_node)

        # Get a point on the surface of the ellipsoid.
        apex_inEllipsoid = \
            torch.stack(
                (
                    core.T0,
                    core.T0,
                    ellipsoid_node.shape_parameters[-1]
                )
            )

        point_node = \
            primitives.Point.create_from_coordinates_inParent(
                ellipsoid_node, apex_inEllipsoid, name="point"
            )

        self.assertTrue(
            torch.allclose(
                ellipsoid_node.compute_algebraic_distance(point_node),
                core.T0
            )
        )

        # Remove point_node from tree.

        # Get a point outside the ellipsoid.
        outside_inEllipsoid = 2 * ellipsoid_node.shape_parameters

        point_node = \
            primitives.Point.create_from_coordinates_inParent(
                ellipsoid_node, outside_inEllipsoid, name="point"
            )

        self.assertTrue(
            ellipsoid_node.compute_algebraic_distance(point_node) >
            core.T0
        )

    def test_compute_apex_inOther(self):
        """test_compute_apex_inOther.

        Test the computation of the coordinates of the ellipsoid's apex in
        the coordinate system of an arbitrary node in the pose graph.
        """

        # Get an arbitrary ellipsoid from the tree.
        ellipsoid_node = self.root.children[0].children[1].children[-1]
        other_node = self.root
        apex_inRoot = ellipsoid_node.compute_apex_inOther(other_node)

        #  The tree looks like this:
        #
        #   root | \
        #    |   \
        #  T |     \ T4
        #    |       \
        #    |         \
        #  child 0    child 1 | \           \
        #    |   \           \ T5, T6, T7 T2 | T3  \           \
        #    |       \       (children 0, 1, 2 of child 1) |         \
        #  child 0   child 1 of child 0  of child 0 | T  | | ellipsoid

        # Coordinates of ellipsoid's apex in root coordinate system is
        # apex_inEllipsoid transformed by T five times.
        apex_inEllipsoid = \
            torch.stack(
                (
                    core.T0,
                    core.T0,
                    ellipsoid_node.shape_parameters[-1]
                )
            )

        apex_inRoot_ = apex_inEllipsoid  # Temporarily.
        for _ in range(5):
            apex_inRoot_ = self.transform.transform(apex_inRoot_)

        self.assertTrue(
            torch.allclose(apex_inRoot, apex_inRoot_)
        )

    def test_compute_apex_inParent(self):
        """test_compute_apex_inParent.

        Test computation of coordinates of ellipsoid's apex (in z direction)
        in the ellipsoid's parent coordinate system.
        """

        # Get an arbitrary ellipsoid from the tree.
        ellipsoid_node = self.root.children[0].children[1].children[-1]
        apex_inParent = ellipsoid_node.compute_apex_inParent()

        apex_inEllipsoid = \
            torch.stack(
                (
                    core.T0,
                    core.T0,
                    ellipsoid_node.shape_parameters[-1]
                )
            )

        apex_inParent_ = self.transform.transform(apex_inEllipsoid)

        self.assertTrue(
            torch.allclose(apex_inParent, apex_inParent_)
        )

    def test_compute_polar_plane_to_point_inEllipsoid(self):
        """test_compute_polar_plane_to_point_inEllipsoid.

        Test computation of homogeneous coordinates of polar plane to input
        point, in coordinate system of the ellipsoid.
        """

        # Get an arbitrary ellipsoid in the tree.
        ellipsoid_node = self.root.children[1].children[2].children[-1]
        point_inEllipsoid = self.create_point_inEllipsoid(ellipsoid_node)

        homogeneous_plane_inEllipsoid = \
            ellipsoid_node.compute_polar_plane_to_point_inEllipsoid(
                point_inEllipsoid
            )

        # Input point lies on plane.
        homogeneous_point_inEllipsoid = core.homogenize(point_inEllipsoid)

        self.assertTrue(
            torch.allclose(
                homogeneous_plane_inEllipsoid @ homogeneous_point_inEllipsoid,
                core.T0
            )
        )

        # Polar plane is tangent to ellipsoid.
        zero = \
            (
                homogeneous_plane_inEllipsoid[:3] / ellipsoid_node.diagonal
            ) @ \
            homogeneous_plane_inEllipsoid[:3] - \
            homogeneous_plane_inEllipsoid[-1]**2

        self.assertTrue(
            torch.allclose(
                zero,
                core.T0,
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

    def test_compute_polar_plane_to_point(self):
        """test_compute_polar_plane_to_point.

        Test computation of polar plane to a given point with respect to the
        ellipsoid.
        """

        # Get an arbitrary ellipsoid in the tree.
        ellipsoid_node = self.root.children[1].children[2].children[-1]
        point_inEllipsoid = self.create_point_inEllipsoid(ellipsoid_node)

        point_node = \
            primitives.Point.create_from_coordinates_inParent(
                ellipsoid_node, point_inEllipsoid
            )

        plane_node = ellipsoid_node.compute_polar_plane_to_point(point_node)

        # In homogeneous coordinates a plane p_h tangent to an ellipsoid
        # represented by the 4 x 4 matrix Q in the same coordinate system of
        # the point satisfies the equation p_h.T * inv(Q) * p_h = 0

        homogeneous_plane_inEllipsoid = \
            plane_node.get_homogeneous_coordinates_inOther(ellipsoid_node)

        zero = \
            (
                homogeneous_plane_inEllipsoid[:3] / ellipsoid_node.diagonal
            ) @ \
            homogeneous_plane_inEllipsoid[:3] - \
            homogeneous_plane_inEllipsoid[-1]**2

        self.assertTrue(
            torch.allclose(
                zero,
                core.T0,
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

        # The normal of the plane must be the same as the normal of the
        # ellipsoid at point_inEllipsoid.
        _, normal_inEllipsoid = \
            plane_node.get_origin_and_normal_inOther(ellipsoid_node)
        direction_inEllipsoid_ = homogeneous_plane_inEllipsoid[:3]
        normal_inEllipsoid_ = core.normalize(direction_inEllipsoid_)

        self.assertTrue(
            torch.allclose(normal_inEllipsoid, normal_inEllipsoid_)
        )

    def test_compute_polar_point_to_plane_inEllipsoid(self):
        """test_compute_polar_point_to_plane_inEllipsoid.

        Test computation of a polar point given the homogeneous coordinates
        of a plane in the coordinate system of the ellipsoid.
        """

        # Get an arbitrary ellipsoid in the tree.
        ellipsoid_node = self.root.children[1].children[2].children[-1]

        # Polar point to polar plane is original point even if point is on
        # ellipsoid.
        point_inEllipsoid = self.create_point_inEllipsoid(ellipsoid_node)
        homogeneous_plane_inEllipsoid = \
            ellipsoid_node.compute_polar_plane_to_point_inEllipsoid(
                point_inEllipsoid
            )

        point_inEllipsoid_ = \
            ellipsoid_node.compute_polar_point_to_plane_inEllipsoid(
                homogeneous_plane_inEllipsoid
            )

        self.assertTrue(torch.allclose(point_inEllipsoid, point_inEllipsoid_))

        # Polar point to polar plane is original point even if point is on
        # ellipsoid.
        point_inEllipsoid = torch.tensor([1.0, -3.0, 7.0])
        homogeneous_plane_inEllipsoid = \
            ellipsoid_node.compute_polar_plane_to_point_inEllipsoid(
                point_inEllipsoid
            )

        point_inEllipsoid_ = \
            ellipsoid_node.compute_polar_point_to_plane_inEllipsoid(
                homogeneous_plane_inEllipsoid
            )

        self.assertTrue(torch.allclose(point_inEllipsoid, point_inEllipsoid_))

    def test_project_on_surface_inEllipsoid(self):
        """test_project_on_surface_inEllipsoid.

        Test orthogonal projection of a point onto the ellipsoid's surface.
        """

        # Get an arbitrary ellipsoid in the tree.
        ellipsoid_node = self.root.children[1].children[2].children[-1]

        #  The tree looks like this:
        #
        #   root | \
        #    |   \
        #  T |     \ T4
        #    |       \
        #    |         \
        #  child 0    child 1 | \           \
        #    |   \           \ T5, T6, T7 T2 | T3  \           \
        #    |       \       (children 0, 1, 2 of child 1) |         \
        #    \
        #  child 0   child 1                 T \
        #  of child 0  of child 0               \
        #                                   ellipsoid

        # Projection of a point already on the surface of the ellipsoid should
        # be equal to the point itself. Create a point on the surface of the
        # ellipsoid.
        point_inEllipsoid = self.create_point_inEllipsoid(ellipsoid_node)

        point_inEllipsoid_ = \
            ellipsoid_node.project_on_surface_inEllipsoid(point_inEllipsoid)

        self.assertTrue(
            torch.allclose(point_inEllipsoid, point_inEllipsoid_)
        )

        # Move the projected point away from the surface of the ellipsoid along
        # the ellipsoid's normal direction at that point. The normal direction
        # at a point x_h in homogeneous coordinates on the surface of the
        # ellipsoid represented by the 4 x 4 matrix Q is Q * x_h in homogeneous
        # coordinates. In the ellipsoid's own coordinate system Q is given by
        # diag((1/a^2, 1/b^2, 1/c^2, -1))
        direction_inEllipsoid = point_inEllipsoid * ellipsoid_node.diagonal

        # Move point away from ellipsoid, outwards.
        new_point_inEllipsoid = point_inEllipsoid + 10 * direction_inEllipsoid

        point_inEllipsoid_ = \
            ellipsoid_node.project_on_surface_inEllipsoid(
                new_point_inEllipsoid
            )

        self.assertTrue(
            torch.allclose(point_inEllipsoid, point_inEllipsoid_)
        )

    def test_compute_occluding_contour(self):
        """test_compute_occluding_contour.

        Test computation of occluding contour observed by a point with
        respect to an ellipsoid.
        """

        # Get an arbitrary ellipsoid.
        parent_node = self.root.children[1].children[0]
        ellipsoid_node = parent_node.children[-1]

        # Get an arbitrary point outside the ellipsoid.
        point_inEllipsoid = torch.tensor([10.0, 11.0, 13.0])
        point_node = \
            primitives.Point.create_from_coordinates_inParent(
                ellipsoid_node, point_inEllipsoid, name="point"
            )

        # Compute occluding contour on ellipsoid with respect to point.
        ellipse_node = ellipsoid_node.compute_occluding_contour(point_node)
        # Compute polar plane to point with respect to ellipsoid.
        polar_plane = ellipsoid_node.compute_polar_plane_to_point(point_node)

        # Points on occluding contour must be simultaneously on the polar plane
        # and on the ellipsoid's surface.
        occluding_points_inEllipse, _ = ellipse_node.sample_points_inEllipse()
        for point_inEllipse in occluding_points_inEllipse.T:
            # Test that points are on the polar plane.
            self.assertTrue(
                polar_plane.is_point_on_plane_inOther(
                    point_inEllipse, ellipse_node
                )
            )

            # Test that points are on the ellipsoid.
            transform_toEllipse_fromEllipsoid = \
                ellipsoid_node.get_transform_toOther_fromSelf(ellipse_node)

            point_inEllipsoid = \
                transform_toEllipse_fromEllipsoid.inverse_transform(
                    point_inEllipse
                )
            self.assertTrue(
                torch.allclose(
                    ellipsoid_node.compute_algebraic_distance_inEllipsoid(
                        point_inEllipsoid
                    ),
                    core.T0,
                    rtol=core.EPS * 100,
                    atol=core.EPS * 100
                )
            )

    # Beyond this point we have the auxiliary methods and tests for reflection
    # on the surface of the ellipsoid. To simplify notation, and contrary to
    # norm, we omit the _inEllipsoid suffix - everything is represented in the
    # coordinate system of the ellipsoid.
    def generate_data(self, requires_grad=False):
        """generate_data.

        Generate data for testing computation of reflection point on surface
        of ellipsoid.

        Returns:
            tuple: (Ellipsoid, torch.Tensor, torch.Tensor, Torch.Tensor) tuple
            representing and ellipsoid, the reflection point in the ellipsoid
            coordinate system, and the origin and destination points in the
            ellipsoid coordinate system. Each torch.Tensor is of shape (3,).
        """

        # Get an arbitrary ellipsoid.
        parent_node = self.root.children[1].children[0]
        ellipsoid_node = parent_node.children[-1]

        # Create an arbitrary point on ellipsoid.
        projection = \
            self.create_point_inEllipsoid(
                ellipsoid_node, requires_grad=requires_grad
            )
        projection_node = \
            primitives.Point.create_from_coordinates_inParent(
                ellipsoid_node, projection
            )
        polar_plane_node = \
            ellipsoid_node.compute_polar_plane_to_point(projection_node)

        # Get normal at point in ellipsoid.
        _, normal = polar_plane_node.get_origin_and_normal_inParent()

        # Create origin and destination that will reflect on known point.
        interpolator = projection + 10 * normal

        orthonormal = \
            polar_plane_node.get_orthonormal_inOther(ellipsoid_node)
        orthogonal = orthonormal @ torch.tensor([1.0, 2.0])

        origin = interpolator - orthogonal
        destination_reflection = interpolator + orthogonal

        # Use Snell's law to create incident and refracted rays with known
        # directions.
        inner_normal = -normal
        normal_ray = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                ellipsoid_node, projection, inner_normal
            )
        closest_point = normal_ray.project_to_ray_inParent(origin)
        orthogonal_direction = core.normalize(closest_point - origin)

        incident_direction = core.normalize(projection - origin)
        sin_incident_angle = \
            torch.linalg.norm(torch.cross(inner_normal, incident_direction))
        sin_refracted_angle = sin_incident_angle / core.TETA.clone()
        cos_refracted_angle = torch.cos(torch.arcsin(sin_refracted_angle))

        refracted_direction = \
            inner_normal * cos_refracted_angle + \
            orthogonal_direction * sin_refracted_angle
        destination_refraction = projection + 1.0 * refracted_direction
        ####

        return \
            ellipsoid_node, \
            projection, \
            interpolator, \
            origin, \
            destination_reflection, \
            destination_refraction

    def test_reflect_from_origin_and_direction_inEllipsoid(self):
        """test_reflect_from_origin_and_direction_inEllipsoid.

        Test reflection of ray from origin and destination in coordinate
        system of ellipsoid.
        """

        ellipsoid, projection, _, origin, destination, _ = self.generate_data()

        projection_, direction_, _ = \
            ellipsoid.reflect_from_origin_and_direction_inEllipsoid(
                origin, projection - origin
            )

        self.assertTrue(torch.allclose(projection, projection_))

        direction = core.normalize(destination - projection)
        direction_ = core.normalize((direction_))
        self.assertTrue(torch.allclose(direction, direction_))

    def test_reflect_from_origin_and_direction_inParent(self):
        """test_reflect_from_origin_and_direction_inParent.

        Test reflection of ray from origin and destination in coordinate
        system of ellipsoid's parent.
        """

        ellipsoid, projection, _, origin, destination, _ = self.generate_data()

        projection_inParent = \
            ellipsoid.transform_toParent_fromSelf.transform(projection)
        destination_inParent = \
            ellipsoid.transform_toParent_fromSelf.transform(destination)
        origin_inParent = \
            ellipsoid.transform_toParent_fromSelf.transform(origin)
        direction_inParent = projection_inParent - origin_inParent

        reflection_origin_inParent_, reflection_direction_inParent_ = \
            ellipsoid.reflect_from_origin_and_direction_inParent(
                origin_inParent, direction_inParent
            )

        self.assertTrue(
            torch.allclose(projection_inParent, reflection_origin_inParent_)
        )

        reflection_direction_inParent = \
            core.normalize(destination_inParent - projection_inParent)
        reflection_direction_inParent_ = \
            core.normalize(reflection_direction_inParent_)
        self.assertTrue(
            torch.allclose(
                reflection_direction_inParent,
                reflection_direction_inParent_,
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

    def test_reflect_ray(self):
        """test_reflect_ray.

        Test reflection of ray on surface of ellipsoid.
        """

        ellipsoid_node, \
            projection, \
            _, \
            origin, \
            destination, \
            _ = \
            self.generate_data()

        # Create a ray from origin to projection.
        ray = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                ellipsoid_node, origin, projection - origin
            )

        reflected_ray = ellipsoid_node.reflect_ray(ray)

        # Reflected ray must not go through origin.
        self.assertFalse(reflected_ray.is_point_on_ray_inParent(origin))
        # But it must go through the reflection point.
        self.assertTrue(reflected_ray.is_point_on_ray_inParent(projection))
        # And it must go through destination.
        self.assertTrue(
            reflected_ray.is_point_on_ray_inParent(
                destination, tol=core.TEPS * 1000
            )
        )

    def test_refract_from_origin_and_direction_inEllipsoid(self):
        """test_refract_from_origin_and_direction_inEllipsoid.

        Test refraction of ray from origin and direction in the ellipsoid's
        coordinate system.
        """

        ellipsoid, projection, _, origin, _, destination = self.generate_data()

        projection_, direction_, _ = \
            ellipsoid.refract_from_origin_and_direction_inEllipsoid(
                origin, projection - origin
            )

        self.assertTrue(torch.allclose(projection, projection_))

        direction = core.normalize(destination - projection)
        direction_ = core.normalize((direction_))
        self.assertTrue(torch.allclose(direction, direction_))

    def test_refract_from_origin_and_direction_inParent(self):
        """test_refract_from_origin_and_direction_inParent.

        Test refraction of ray from origin and destination in coordinate
        system of ellipsoid's parent.
        """

        ellipsoid, projection, _, origin, _, destination = self.generate_data()

        projection_inParent = \
            ellipsoid.transform_toParent_fromSelf.transform(projection)
        destination_inParent = \
            ellipsoid.transform_toParent_fromSelf.transform(destination)
        origin_inParent = \
            ellipsoid.transform_toParent_fromSelf.transform(origin)
        direction_inParent = projection_inParent - origin_inParent

        reflection_origin_inParent_, reflection_direction_inParent_, _ = \
            ellipsoid.refract_from_origin_and_direction_inParent(
                origin_inParent, direction_inParent
            )

        self.assertTrue(
            torch.allclose(projection_inParent, reflection_origin_inParent_)
        )

        reflection_direction_inParent = \
            core.normalize(destination_inParent - projection_inParent)
        reflection_direction_inParent_ = \
            core.normalize(reflection_direction_inParent_)
        self.assertTrue(
            torch.allclose(
                reflection_direction_inParent,
                reflection_direction_inParent_,
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

    def test_refract_ray(self):
        """test_refract_ray.

        Test refraction of ray on surface of ellipsoid.
        """

        ellipsoid_node, projection, _, origin, _, destination = \
            self.generate_data()

        # Hit the ellipsoid right on, there should be no refraction.
        projection_ = \
            ellipsoid_node.project_on_surface_inEllipsoid(origin)
        incident_direction = projection_ - origin
        incident_direction = core.normalize(incident_direction)
        incident_ray = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                ellipsoid_node, origin, incident_direction
            )
        refracted_ray = ellipsoid_node.refract_ray(incident_ray)
        _, refracted_direction = \
            refracted_ray.get_origin_and_direction_inParent()
        ellipsoid_node.remove_child(refracted_ray)

        self.assertTrue(
            torch.allclose(incident_direction, refracted_direction)
        )

        # Use Snell's law to create incident and refracted rays with known
        # directions.
        incident_direction = core.normalize(projection - origin)
        incident_ray = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                ellipsoid_node, origin, incident_direction
            )
        refracted_ray_ = ellipsoid_node.refract_ray(incident_ray)

        # Verify that the incidence point is on the refracted ray.
        self.assertTrue(refracted_ray_.is_point_on_ray_inParent(projection))

        # Verify that the direction of the refracted ray agrees with Snell's.
        refracted_direction = core.normalize(destination - projection)
        _, refracted_direction_ = \
            refracted_ray_.get_origin_and_direction_inParent()
        self.assertTrue(
            torch.allclose(
                refracted_direction,
                refracted_direction_,
                rtol=core.EPS * 100,
                atol=core.EPS * 100)
        )

    def test_compute_reflection_point_inEllipsoid(self):
        """test_compute_reflection_point_inEllipsoid.

        Test computation of reflection point.
        """

        # The points are torch.Tensor representing coordinates in the
        # coordinate system of the ellipsoid.
        ellipsoid_node, projection, interpolator, origin, destination, _ = \
            self.generate_data()

        projection_ = ellipsoid_node.project_on_surface_inEllipsoid(
            interpolator
        )

        self.assertTrue(torch.allclose(projection, projection_))

        # Now, compute the projection point using the internal routine.
        projection_ = \
            ellipsoid_node.compute_reflection_point_inEllipsoid(
                origin, destination
            )

        self.assertTrue(torch.allclose(projection, projection_))

    def test_gradients_compute_reflection_point_inEllipsoid(self):
        """test_gradients_compute_reflection_point_inEllipsoid.

        Test computation of gradients for reflected points.
        """

        ellipsoid_node, projection, interpolator, origin, destination, _ = \
            self.generate_data(requires_grad=True)

        # Now, compute the projection point using the internal routine.
        projection_ = \
            ellipsoid_node.compute_reflection_point_inEllipsoid(
                origin, destination
            )

        self.assertTrue(projection_.requires_grad)

        # Compute derivative of projection with respect to origin:
        def fun_origin(origin):
            return ellipsoid_node.compute_reflection_point_inEllipsoid(
                origin, destination, tol=core.TEPS * 10
            )

        d_projection_d_origin_numeric = \
            core.compute_numeric_jacobian_from_tensors(
                origin, fun_origin, delta=torch.tensor(1e-2)
            )

        d_projection_d_origin_autograd = \
            core.compute_auto_jacobian_from_tensors(projection_, origin)

        # It is very difficult to get good precision with the numeric
        # derivatives, so we must be quite forgiving.
        self.assertTrue(
            torch.allclose(
                d_projection_d_origin_autograd,
                d_projection_d_origin_numeric,
                rtol=core.EPS * 1000,
                atol=core.EPS * 1000
            )
        )

        # Compute derivative of projection with respect to destination:
        def fun_destination(destination):
            return ellipsoid_node.compute_reflection_point_inEllipsoid(
                origin, destination, tol=core.TEPS * 10
            )

        d_projection_d_destination_numeric = \
            core.compute_numeric_jacobian_from_tensors(
                destination, fun_destination, delta=torch.tensor(1e-2)
            )

        d_projection_d_destination_autograd = \
            core.compute_auto_jacobian_from_tensors(
                projection_, destination
            )

        self.assertTrue(
            torch.allclose(
                d_projection_d_destination_autograd,
                d_projection_d_destination_numeric,
                rtol=core.EPS * 10000,
                atol=core.EPS * 10000
            )
        )

        # Compute derivative of projection with respect to shape parameters:
        shape_parameters = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        ellipsoid_node.reset_shape(shape_parameters)

        projection_ = \
            ellipsoid_node.compute_reflection_point_inEllipsoid(
                origin, destination
            )

        def fun_shape_parameters(shape_parameters):
            new_ellipsoid = \
                primitives.Ellipsoid(
                    ellipsoid_node.parent,
                    ellipsoid_node.transform_toParent_fromSelf,
                    shape_parameters)

            return new_ellipsoid.compute_reflection_point_inEllipsoid(
                origin, destination, tol=core.TEPS * 10
            )

        d_projection_d_shape_numeric = \
            core.compute_numeric_jacobian_from_tensors(
                shape_parameters, fun_shape_parameters, delta=torch.tensor(
                    1e-2)
            )

        d_projection_d_shape_autograd = \
            core.compute_auto_jacobian_from_tensors(
                projection_, shape_parameters
            )

        self.assertTrue(
            torch.allclose(
                d_projection_d_shape_autograd,
                d_projection_d_shape_numeric,
                rtol=core.EPS * 100000,
                atol=core.EPS * 100000
            )
        )

    def test_compute_refraction_point_no_grad_inEllipsoid(self):
        """test_compute_refraction_point_no_grad_inEllipsoid.

        Test public API for computation of refraction point.
        """

        ellipsoid_node, projection, _, origin, _, destination = \
            self.generate_data()

        refraction_point = \
            ellipsoid_node._compute_refraction_point_no_grad_inEllipsoid(
                origin,
                destination,
                eta_at_origin=core.T1.clone(),
                eta_at_destination=core.TETA.clone(),
                tol=core.TEPS * 100,
                max_num_iter=10
            )

        self.assertTrue(torch.allclose(projection, refraction_point))

    def test_gradients_compute_refraction_point_inEllipsoid(self):
        """test_gradients_compute_refraction_point_inEllipsoid.

        Test computation of gradients of refracted rays.
        """

        ellipsoid_node = self.generate_data()[0]

        # Create more manageable data. Same refraction indices, so there's
        # actually no refraction
        common_eta = torch.tensor(1.0)
        eta_at_origin = common_eta
        eta_at_destination = common_eta
        # Sphere:
        radius = torch.tensor(1.5, requires_grad=True)
        shape_parameters = torch.stack((radius, radius, radius))
        ellipsoid_node.reset_shape(shape_parameters)
        origin = torch.tensor([1.5, -2.0, 3.0], requires_grad=True)
        destination = torch.zeros(3, requires_grad=True)

        # Compute "refraction" and its derivatives using knowledge that the
        # refraction point is the intersection of the ray from origin to
        # destination.
        refraction_A = \
            ellipsoid_node.intersect_from_origin_and_direction_inEllipsoid(
                origin, destination - origin
            )[0]

        refraction_B = \
            ellipsoid_node.compute_refraction_point_inEllipsoid(
                origin,
                destination,
                eta_at_origin=eta_at_origin,
                eta_at_destination=eta_at_destination,
                create_graph=True
            )

        # The intersection/refraction point is origin/|origin| * radius.
        normal_origin = torch.linalg.norm(origin)
        refraction_C = origin / normal_origin * radius
        d_refraction_C_d_x = \
            (
                refraction_C,
                (
                    torch.eye(3) -
                    torch.outer(origin, origin) / normal_origin**2
                ) / normal_origin * radius
            )
        d2_refraction_C_d_xy = \
            (
                (
                    torch.zeros((3, 1, 1)),
                    (d_refraction_C_d_x[1] / radius)
                ),
                (
                    (d_refraction_C_d_x[1] / radius),
                    core.compute_auto_jacobian_from_tensors(
                        d_refraction_C_d_x[1].T, origin
                    )
                )
            )
        counter_x = 0
        x_parameters = (radius, origin)
        for x in x_parameters:
            counter_y = 0
            # Compute first derivatives.
            d_refraction_A_d_x = \
                core.compute_auto_jacobian_from_tensors(
                    refraction_A, x, create_graph=True
                )

            d_refraction_B_d_x = \
                core.compute_auto_jacobian_from_tensors(
                    refraction_B, x, create_graph=True
                )

            self.assertTrue(
                torch.allclose(
                    d_refraction_A_d_x,
                    d_refraction_B_d_x
                )
            )

            torch.allclose(
                d_refraction_B_d_x,
                d_refraction_C_d_x[counter_x]
            )

            # Compute second derivatives.
            for y in x_parameters:
                d2_refraction_A_d_xy = \
                    core.compute_auto_jacobian_from_tensors(
                        d_refraction_A_d_x, y
                    )

                # Regrettably, here we need to fall back on the old,
                # inefficient way to compute derivatives.
                d2_refraction_B_d_xy = \
                    core.alt_compute_auto_jacobian_from_tensors(
                        d_refraction_B_d_x, y
                    )

                self.assertTrue(
                    torch.allclose(
                        d2_refraction_A_d_xy,
                        d2_refraction_B_d_xy,
                        rtol=core.EPS * 1000,
                        atol=core.EPS * 1000
                    )
                )

                self.assertTrue(
                    torch.allclose(
                        d2_refraction_B_d_xy,
                        d2_refraction_C_d_xy[counter_x][counter_y],
                        rtol=core.EPS * 1000,
                        atol=core.EPS * 1000
                    )
                )

                counter_y += 1

            counter_x += 1


if __name__ == "__main__":
    unittest.main()
