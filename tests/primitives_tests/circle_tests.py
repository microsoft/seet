"""circle_tests.py

Unit tests for Circle class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.primitives as primitives
from tests import tests_utils
import torch
import unittest


class TestCircle(tests_utils.TestUtils):
    """TestCircle.

    Test case for circle is the same as for ellipse.
    """
    radius = torch.tensor(2.0)

    def setUp(self):
        """setUp.

        Initialize test parameters.
        """
        super().setUp()

        nodes = self.root.traverse()  # Depth first
        for parent_node in nodes:
            primitives.Circle(
                parent_node,
                self.transform,
                self.radius,
                name=f"circle in {parent_node.name}"
            )

    def test_create_from_origin_and_radius_inPlane(self):
        """test_create_from_origin_and_radius_inPlane.

        Test class method for creation of circle from an origin and a radius
        in a parent plane.
        """

        plane_node = primitives.Plane(None, core.SE3.create_identity())
        center_in3DPlane = torch.tensor([1.0, 2.0, 0.0])
        center_in2DPlane = center_in3DPlane[:2]
        normal_inPlane = torch.tensor([0.0, 0.0, 1.0])
        radius = torch.tensor(3.0)
        circle_node = \
            primitives.Circle.create_from_origin_and_radius_inPlane(
                plane_node, center_in2DPlane, radius
            )
        center_in3DPlane_, \
            normal_inPlane_ = circle_node.get_center_and_normal_inParent()

        self.assertTrue(torch.allclose(center_in3DPlane, center_in3DPlane_))
        self.assertTrue(torch.allclose(normal_inPlane_, normal_inPlane))
        self.assertTrue(
            torch.allclose(circle_node.radius, radius)  # type: ignore
        )

    def test_get_radius(self):
        """test_get_radius.

        Test getter method for circle radius.
        """
        # Get the radius of some circle.
        parent_node = self.root.children[0]
        circle_node = parent_node.children[-1]
        radius = circle_node.get_radius()

        self.assertTrue(radius == self.radius)
        self.assertTrue(radius == circle_node.x_radius)
        self.assertTrue(radius == circle_node.y_radius)

    def test_set_radius(self):
        """test_set_radius.

        Test setter method for radius of circle.
        """
        # Get the radius of some circle.
        parent_node = self.root.children[0].children[0]
        circle_node = parent_node.children[-1]
        radius = circle_node.get_radius()

        # Double the radius and check that the distance from points on the
        # circle to the center of the circle has doubled.
        circle_node.set_radius(2 * radius)
        center_inParent, _ = circle_node.get_center_and_normal_inParent()
        points_inParent, _ = circle_node.get_points_inParent()
        delta_inParent = points_inParent - center_inParent.view(3, 1)
        distances = torch.linalg.norm(delta_inParent, dim=0)

        self.assertTrue(torch.allclose(distances, 2 * radius))

    def test_update_radius(self):
        """test_update_radius.

        Test update radius of circle.
        """
        # If we double the radii, the sampled points must double in distance to
        # the center of the ellipse.
        parent_node = self.root.children[0].children[0]
        circle_node = parent_node.children[-1]
        center_inParent, _ = circle_node.get_center_and_normal_inParent()
        points_inParent, _ = circle_node.get_points_inParent()

        # Additive radius update.
        circle_node.update_radius(circle_node.radius)
        new_points_inParent, _ = circle_node.get_points_inParent()

        # For each original point create a ray and check that the corresponding
        # new point lies on it.
        for i in range(points_inParent.shape[1]):
            ray_node = \
                primitives.Ray.create_from_origin_and_dir_inParent(
                    parent_node,
                    center_inParent,
                    points_inParent[:, i] - center_inParent
                )
            self.assertTrue(
                ray_node.is_point_on_ray_inParent(new_points_inParent[:, i])
            )

        # Multiplicative radius update.
        circle_node.update_radius(
            torch.log(torch.tensor(0.5)), update_mode="multiplicative"
        )
        points_inParent_, _ = circle_node.get_points_inParent()

        # Points should be the same as before.
        self.assertTrue(torch.allclose(points_inParent, points_inParent_))


if __name__ == "__main__":
    unittest.main()
