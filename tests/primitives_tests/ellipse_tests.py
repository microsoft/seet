"""ellipse_tests.py

Unit tests for Ellipse class.
"""


__author__ = "Paulo R. S. Mendonca"


import seet.core as core
from tests import tests_utils
import seet.primitives as primitives
import torch
import unittest


class TestEllipse(tests_utils.TestUtils):
    x_radius = torch.tensor(2.0, requires_grad=True)
    y_radius = torch.tensor(1.0, requires_grad=True)

    def setUp(self):
        super().setUp()

        nodes = self.root.traverse()  # Depth first
        for parent_node in nodes:
            primitives.Ellipse(
                parent_node,
                self.transform,
                self.x_radius,
                self.y_radius,
                name=f"ellipse in {parent_node.name}"
            )

    def create_data_for_testing_homogeneous_matrix(self):
        """create_data_for_testing_homogeneous_matrix.

        Create plane and ellipse with which to test the creation of an
        ellipse from a homogeneous matrix and the extraction of the homogeneous
        matrix given an ellipse.
        """

        # Create an arbitrary plane in some node of the tree.
        parent_node = self.root.children[1].children[1]
        origin_inParent = torch.tensor([1.0, 2.0, 3.0])
        normal_inParent = torch.tensor([0.0, 1.0, 0.0])  # Parallel to xz.
        plane_node = primitives.Plane.create_from_origin_and_normal_inParent(
            parent_node,
            origin_inParent,
            normal_inParent,
            name="arbitrary plane"
        )

        # Create an ellipse with very long x semi-axis, rotated 30 degrees
        # counterclockwise, and centered at (-1, -1) in the plane's coordinate
        # system.
        angle_deg = 30
        angle_rad = core.deg_to_rad(angle_deg)
        c = torch.cos(angle_rad)
        s = torch.sin(angle_rad)
        rotation_to2DPlane_fromEllipse = \
            core.stack_tensors(
                ((c, -s),
                 (s, c))
            )
        center_in2DPlane = torch.tensor([-1.0, -1.0])
        x_radius = torch.tensor(13.0)
        y_radius = torch.tensor(3.0)
        transformation_matrix_to2DPlane_fromEllipse = \
            torch.vstack(
                (
                    torch.hstack(
                        (rotation_to2DPlane_fromEllipse,
                         center_in2DPlane.view(2, 1))
                    ),
                    torch.tensor([0.0, 0.0, 1.0])
                )
            )

        transformation_matrix_toEllipse_from2DPlane = \
            torch.linalg.pinv(transformation_matrix_to2DPlane_fromEllipse)

        C_inPlane = \
            transformation_matrix_toEllipse_from2DPlane.T @ \
            torch.diag(
                torch.stack((1 / x_radius**2, 1 / y_radius**2, -core.T1))
            ) @ \
            transformation_matrix_toEllipse_from2DPlane

        ellipse_node = \
            primitives.Ellipse.create_from_homogeneous_matrix_inPlane(
                plane_node, C_inPlane
            )

        return \
            plane_node, \
            ellipse_node, \
            x_radius, y_radius, center_in2DPlane, angle_deg, C_inPlane

    def test_create_from_origin_angle_and_axes_inPlane(self):
        """test_create_from_origin_angle_and_axes_inPlane.

        Test creation of ellipse node from origin, angle, and lengths of
        semi-axes in the 2D coordinate system of a plane core.
        """

        plane_node, \
            ellipse_node, \
            x_radius, \
            _, \
            center_in2DPlane, \
            angle_deg, \
            _ = \
            self.create_data_for_testing_homogeneous_matrix()

        center_inRoot, _ = \
            ellipse_node.get_center_and_normal_inOther(self.root)

        # To manually get the origin and normal in the root coordinate system,
        # we first map the origin in the 2D plane to 3D.
        origin_plane_inRoot, _ = \
            plane_node.get_origin_and_normal_inOther(self.root)
        orthonormal_inRoot = plane_node.get_orthonormal_inOther(self.root)
        center_inRoot_ = \
            origin_plane_inRoot + orthonormal_inRoot @ center_in2DPlane

        self.assertTrue(torch.allclose(center_inRoot, center_inRoot_))

        #                                                   |  plane 2D
        #                                                   | coodinate
        #                                                   |   system____ .
        #                                                   |  _____/    x apex
        #                                                _____/
        #                                          _____/   *-------------
        #                                    _____/
        #             . y apex         _____/
        #             \          _____/
        #              \   _____/     \
        #               \ /            | angle
        # ellipse center *---------------------
        #
        # The coordinates of x apex in the 2D coordinate system of the plane
        # are (x_radius * cos(angle), x_radius * sin(angle)) - ellipse_center.

        x_extreme_inEllipse = torch.tensor([1.0, 0.0, 0.0]) * x_radius
        x_extreme_inPlane = \
            ellipse_node.get_transform_toOther_fromSelf(
                plane_node
            ).transform(x_extreme_inEllipse)

        angle_rad = core.deg_to_rad(angle_deg)
        x_extreme_in2DPlane_ = \
            x_radius * \
            torch.stack(
                (torch.cos(angle_rad), torch.sin(angle_rad))
            ) + center_in2DPlane

        self.assertTrue(
            torch.allclose(x_extreme_inPlane[:2], x_extreme_in2DPlane_)
        )

    def test_get_homogeneous_matrix_inPlane(self):
        """test_get_homogeneous_matrix_inPlane.

        Test extraction of homogeneous matrix representing ellipse in parent
        plane node.
        """

        # The outputs are:
        #
        # plane_node, \
        #   ellipse_node, \
        #   x_radius, \
        #   y_radius, \
        #   center_in2DPlane, \
        #   angle_deg, \
        #   C_inPlane (homogeneous matrix)
        _, \
            ellipse_node, \
            _, \
            _, \
            _, \
            _, C_inPlane = self.create_data_for_testing_homogeneous_matrix()

        C_inPlane_ = ellipse_node.get_homogeneous_matrix_inPlane()

        # Homogeneous matrices must be the same up to a scale factor.
        scale = torch.argmax(torch.abs(C_inPlane))
        C_inPlane = C_inPlane / scale
        scale_ = torch.argmax(torch.abs(C_inPlane_))
        C_inPlane_ = C_inPlane_ / scale_

        self.assertTrue(torch.allclose(C_inPlane, C_inPlane_))

    def test_create_from_homogeneous_matrix_inPlane(self):
        """test_create_from_homogeneous_matrix_inPlane.

        Test the creation of an ellipse from its homogenous matrix in a
        parent's plane coordinate system.
        """

        plane_node, \
            ellipse_node, \
            x_radius, \
            y_radius, \
            center_in2DPlane, \
            _, \
            _ = self.create_data_for_testing_homogeneous_matrix()

        # Test lengths of semi-axes.
        x_radius_ = ellipse_node.x_radius
        y_radius_ = ellipse_node.y_radius

        self.assertTrue(torch.allclose(x_radius, x_radius_))
        self.assertTrue(torch.allclose(y_radius, y_radius_))

        # Test center and normal directions of ellipse in root's coordinate
        # system.
        origin_inRoot, normal_inRoot = \
            ellipse_node.get_center_and_normal_inOther(self.root)
        #   root | \
        #    |   \
        #  T |     \ T4
        #    |       \
        #    |         \
        #  child 0    child 1 | \           \
        #    |   \           \ T5, T6, T7 T2 | T3  \           \
        #    |       \       (children 0, 1, 2 of child 1) |         \
        #    \
        #  child 0   child 1                 \ Transform of child 0  of child 0
        #  \
        #                                     plane
        #                                       |
        #                                       | origin, normal
        #                                       |
        #                                    ellipse
        origin_inPlane = torch.hstack((center_in2DPlane, core.T0))
        normal_inPlane = torch.tensor([0.0, 0.0, 1.0])

        transform_toPlaneParent_fromPlane = \
            plane_node.transform_toParent_fromSelf
        origin_inPlaneParent = \
            transform_toPlaneParent_fromPlane.transform(origin_inPlane)
        normal_inPlaneParent = \
            transform_toPlaneParent_fromPlane.rotation.transform(
                normal_inPlane)

        # We now concatenate ten transformations to get to the root's
        # coordinate system.
        origin_inRoot_ = origin_inPlaneParent  # Temporarily.
        normal_inRoot_ = normal_inPlaneParent
        for _ in range(10):
            origin_inRoot_ = self.transform.transform(origin_inRoot_)
            normal_inRoot_ = self.transform.rotation.transform(normal_inRoot_)

        self.assertTrue(torch.allclose(origin_inRoot, origin_inRoot_))
        self.assertTrue(torch.allclose(normal_inRoot, normal_inRoot_))

    def test_sample_points_inEllipse(self):
        """test_sample_points_inEllipse.

        Test method that samples points along ellipse.
        """

        # Take an arbitrary ellipse
        parent_node = self.root.children[0].children[0]
        ellipse_node = parent_node.children[-1]

        # Sum of distances from a point on the ellipse to the foci is equal to
        # the length of the major axis.
        points_inParent, _ = ellipse_node.get_points_inParent()
        distance_toCenter_fromFocus = \
            torch.sqrt(
                torch.abs(
                    ellipse_node.x_radius**2
                    - ellipse_node.y_radius**2
                )
            )
        # Foci are along the major axis.
        if ellipse_node.x_radius > ellipse_node.y_radius:
            length_of_major_axis = 2 * ellipse_node.x_radius
            x = distance_toCenter_fromFocus
            y = core.T0.clone()  # Do not create references to "constants"!
        else:
            length_of_major_axis = 2 * ellipse_node.y_radius
            x = core.T0.clone()  # Do not create references to "constants"!
            y = distance_toCenter_fromFocus

        focus_1_inEllipse = torch.stack((x, y, core.T0))
        focus_2_inEllipse = torch.stack((-x, -y, core.T0))

        focus_1_inParent = \
            ellipse_node.transform_toParent_fromSelf.transform(
                focus_1_inEllipse
            )
        focus_2_inParent = \
            ellipse_node.transform_toParent_fromSelf.transform(
                focus_2_inEllipse
            )

        # Compute the distances from points on the ellipse to each focus.
        delta_1_inParent = points_inParent - focus_1_inParent.view(3, 1)
        delta_2_inParent = points_inParent - focus_2_inParent.view(3, 1)
        distance_1 = torch.linalg.norm(delta_1_inParent, dim=0)
        distance_2 = torch.linalg.norm(delta_2_inParent, dim=0)

        sum_of_distances = distance_1 + distance_2

        for d in sum_of_distances:
            self.assertTrue(torch.allclose(d, length_of_major_axis))

        # If we double the number of points, the every other new point must be
        # the same as each old point.
        new_points_inParent, _ = \
            ellipse_node.get_points_inParent(num_points=60)
        self.assertTrue(
            torch.allclose(new_points_inParent[:, ::2], points_inParent)
        )

    def test_set_radii(self):
        """test_set_radii.

        Test the updating of the lengths of the semi-axes of the ellipse.
        """
        # If we double the radii, the sampled points must double in distance to
        # the center of the ellipse.
        parent_node = self.root.children[0].children[0]
        ellipse_node = parent_node.children[-1]
        center_inParent, _ = ellipse_node.get_center_and_normal_inParent()
        points_inParent, _ = ellipse_node.get_points_inParent()

        ellipse_node.set_radii(
            2 * ellipse_node.x_radius, 2 * ellipse_node.y_radius
        )
        new_points_inParent, _ = ellipse_node.get_points_inParent()

        # For each original point create a ray and check that the corresponding
        # new point lies on it.
        for i in range(new_points_inParent.shape[1]):
            ray_node = \
                primitives.Ray.create_from_origin_and_dir_inParent(
                    parent_node,
                    center_inParent,
                    points_inParent[:, i] - center_inParent
                )
            self.assertTrue(
                ray_node.is_point_on_ray_inParent(new_points_inParent[:, i])
            )

    def test_update_radii(self):
        """test_update_radii.

        Test the method that updates the radii of the ellipse. Test both
        additive and multiplicative updates.
        """
        # If we double the radii, the sampled points must double in distance to
        # the center of the ellipse.
        parent_node = self.root.children[0].children[0]
        ellipse_node = parent_node.children[-1]
        center_inParent, _ = ellipse_node.get_center_and_normal_inParent()
        points_inParent, _ = ellipse_node.get_points_inParent()

        # Additive radii update.
        ellipse_node.update_radii(
            ellipse_node.x_radius, ellipse_node.y_radius
        )
        new_points_inParent, _ = ellipse_node.get_points_inParent()

        # For each original point create a ray and check that the corresponding
        # new point lies on it.
        for i in range(new_points_inParent.shape[1]):
            ray_node = \
                primitives.Ray.create_from_origin_and_dir_inParent(
                    parent_node,
                    center_inParent,
                    points_inParent[:, i] - center_inParent
                )
            self.assertTrue(
                ray_node.is_point_on_ray_inParent(new_points_inParent[:, i])
            )

        # Multiplicative shape update.
        ellipse_node.update_radii(
            torch.log(torch.tensor(0.5)),
            torch.log(torch.tensor(0.5)),
            update_mode="multiplicative"
        )
        points_inParent_, _ = ellipse_node.get_points_inParent()

        # Points should be the same as before.
        self.assertTrue(torch.allclose(points_inParent, points_inParent_))

    def test_get_center_and_normal_inParent(self):
        """test_get_center_and_normal_inParent.

        Test extracting the center and normal direction of the ellipse in
        the parent's coordinate system.
        """
        # Get an arbitrary ellipse object.
        parent_node = self.root.children[0]
        ellipse_node = parent_node.children[-1]
        center_inParent, normal_inParent = \
            ellipse_node.get_center_and_normal_inParent()

        # Update the transformation between the ellipse and its parent.
        ellipse_node.update_transform_toParent_fromSelf(self.transform)
        new_center_inParent, new_normal_inParent = \
            ellipse_node.get_center_and_normal_inParent()

        # Transform the old center and normal using the same update.
        new_center_inParent_ = self.transform.transform(center_inParent)
        new_normal_inParent_ = \
            self.transform.rotation.transform(normal_inParent)

        self.assertTrue(
            torch.allclose(new_center_inParent, new_center_inParent_)
        )

        self.assertTrue(
            torch.allclose(new_normal_inParent, new_normal_inParent_)
        )

    def test_get_points_inParent(self):
        """test_get_points_inParent.

        Test getting the sampled ellipse points in the parent's coordinate
        system.
        """
        # Create a trivial ellipse.
        parent_node = core.Node()
        ellipse_node = \
            primitives.Ellipse(
                parent_node,
                core.SE3.create_identity(),
                self.x_radius,
                self.y_radius
            )
        points_inParent, _ = ellipse_node.get_points_inParent(num_points=4)

        # X coordinates are self.x_radius, 0, -self.x_radius, 0.
        self.assertTrue(
            torch.allclose(
                points_inParent[0, :],
                torch.stack(
                    (self.x_radius, core.T0, -self.x_radius, core.T0)
                ),
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

        # Y coordinates are 0, self.y_radius, 0, -self.y_radius.
        self.assertTrue(
            torch.allclose(
                points_inParent[1, :],
                torch.stack(
                    (core.T0, self.y_radius, core.T0, -self.y_radius)
                ),
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

        # Z coordinates are zero.
        self.assertTrue(
            torch.allclose(
                points_inParent[2, :],
                torch.zeros(points_inParent.shape[1]),
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

    def test_get_angle_of_closest_point(self):
        """test_get_angle_of_closest_point.

        Test computation of angle of point on ellipse closest to given
        point.
        """

        angle_deg = torch.tensor(15.0, requires_grad=True)
        angle_rad = core.deg_to_rad(angle_deg)

        ellipse_node = \
            primitives.Ellipse(
                self.root,
                self.transform,
                self.x_radius,
                self.y_radius,
            )

        point_on_ellipse_inEllipse = \
            ellipse_node.get_points_at_angles_inEllipse(angle_rad)

        # Point on ellipse.
        angle_rad_ = \
            ellipse_node.compute_angle_of_closest_point(
                point_on_ellipse_inEllipse
            )
        self.assertAlmostEqual(
            angle_rad.clone().detach().item(),
            angle_rad_.item()  # type: ignore
        )

        # Point out of ellipse.
        normal_at_point_on_ellipse_inEllipse = \
            ellipse_node.get_normal_at_angles_inEllipse(angle_rad)
        point_out_of_ellipse_inEllipse = \
            point_on_ellipse_inEllipse + \
            10.0 * normal_at_point_on_ellipse_inEllipse
        # Make it a leaf node
        point_out_of_ellipse_inEllipse = \
            point_out_of_ellipse_inEllipse.clone().detach()

        point_out_of_ellipse_inEllipse.requires_grad = True
        angle_rad_ = \
            ellipse_node.compute_angle_of_closest_point(
                point_out_of_ellipse_inEllipse
            )
        self.assertAlmostEqual(
            angle_rad.clone().detach().item(),
            angle_rad_.item()  # type: ignore
        )

        # Test gradient with respect to input point.
        # With autograd.
        d_angle_rad_d_point = \
            core.compute_auto_jacobian_from_tensors(
                angle_rad_, point_out_of_ellipse_inEllipse
            )

        # Numerical.
        def fun_angle_given_point(point):
            return ellipse_node.compute_angle_of_closest_point(point)

        d_angle_rad_d_point_ = \
            core.compute_numeric_jacobian_from_tensors(
                point_out_of_ellipse_inEllipse, fun_angle_given_point
            )

        self.assertTrue(
            torch.allclose(
                d_angle_rad_d_point,
                d_angle_rad_d_point_,
                rtol=1e-2,  # This needs to be well relaxed, numerics are bad.
                atol=1e-2   # This needs to be well relaxed, numerics are bad.
            )
        )

        # Test gradient with respect to semi-axes.
        # With autograd.
        d_angle_rad_d_x_radius = \
            core.compute_auto_jacobian_from_tensors(
                angle_rad_, ellipse_node.x_radius
            )

        # Numerical.
        def fun_angle_given_x_radius(x_radius):
            # Store the old x radius.
            old_x_radius = ellipse_node.x_radius.clone().detach()

            # Re-create the ellipse with a new x radius.
            ellipse_node.set_radii(x_radius, self.y_radius)

            # Compute the angle with the new ellipse
            angle = \
                ellipse_node.compute_angle_of_closest_point(
                    point_out_of_ellipse_inEllipse
                )

            # Set back to old radius.
            old_x_radius.requires_grad = True
            ellipse_node.set_radii(old_x_radius, self.y_radius)

            return angle

        d_angle_rad_d_x_radius_ = \
            core.compute_numeric_jacobian_from_tensors(
                ellipse_node.x_radius, fun_angle_given_x_radius
            )

        self.assertTrue(
            torch.allclose(
                d_angle_rad_d_x_radius,
                d_angle_rad_d_x_radius_,
                rtol=1e-2,  # This needs to be well relaxed, numerics are bad.
                atol=1e-3   # This needs to be well relaxed, numerics are bad.
            )
        )


if __name__ == "__main__":
    unittest.main()
