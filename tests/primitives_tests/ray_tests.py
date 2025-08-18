"""Unit-tests for methods and classes in ray.py.

Unit-tests for methods and classes in ray.py
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.primitives as primitives
from tests import tests_utils
import torch
import unittest


class TestRay(tests_utils.TestUtils):
    def setUp(self):
        """setUp.

        Setup method run before every test.
        """
        super().setUp()

        nodes = self.root.traverse()  # Depth first.
        for node in nodes:
            primitives.Ray(
                node,
                self.transform,
                name=f"ray in {node.name}"
            )

    def test_creation(self):
        """test_creation.

        Test the creation of rays from origin and direction and from origin
        and angles.
        """
        origin_inParent = torch.tensor([1.0, 2., 3])
        direction_inParent = torch.tensor([-0.1, 0.2, -0.3])
        normalized_direction_inParent = core.normalize(direction_inParent)
        ray_node = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self.root,
                origin_inParent,
                normalized_direction_inParent,
                name=f"test ray in {self.root.name}"
            )

        params_inParent = ray_node.get_params_inParent()
        ray_node_ = \
            primitives.Ray.create_from_params_inParent(
                self.root,
                params_inParent,
                name=f"test ray_ in {self.root.name}"
            )

        origin_inParent_, normalized_direction_inParent_ = \
            ray_node_.get_origin_and_direction_inParent()

        self.assertTrue(torch.allclose(origin_inParent, origin_inParent_))
        self.assertTrue(
            torch.allclose(
                normalized_direction_inParent, normalized_direction_inParent_
            )
        )

    def test_intersect_rays_inOther(self):
        """Test static method for intersection of two rays.
        """
        origin_inParent = torch.tensor([1.0, 2., 3])
        direction_inParent = torch.tensor([-0.1, 0.2, -0.3])
        normalized_direction_inParent = core.normalize(direction_inParent)
        ray_node = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self.root,
                origin_inParent,
                normalized_direction_inParent,
                name=f"test ray in {self.root.name}"
            )

        # Same origin different direction, not parallel to previous direction.
        new_direction_inParent = torch.tensor([0.2, -0.1, 3.0])
        new_ray_node = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self.root, origin_inParent, new_direction_inParent
            )

        intersection_inRoot = \
            primitives.Ray.intersect_rays_inOther(
                self.root, ray_node, new_ray_node
            )
        self.assertTrue(torch.allclose(origin_inParent, intersection_inRoot))

    def test_parameter_getters(self):
        """test_parameter_getters.

        Test the get_params_inX methods, where X is Parent, World, or Other.
        """
        # The last child added to each node was a ray.
        ray_node = self.root.children[-1]

        # The last child added to each node was a ray.
        ray_node = self.root.children[0].children[-1]

        origin, direction = ray_node.get_origin_and_direction_inParent()
        origin_, direction_ = \
            ray_node.get_origin_and_direction_inOther(ray_node.parent)
        # Because other is parent, the origins and directions above should be
        # same.
        self.assertTrue(torch.allclose(origin, origin_))
        self.assertTrue(torch.allclose(direction, direction_))

    def test_project_to_ray_inParent(self):
        """test_project_to_ray_inParent.

        Test the projection of a point into the ray.
        """
        parent_node = self.root.children[1].children[1]
        ray_node = parent_node.children[-1]

        # Projection of a point in the root's coordinate system is just
        # the z-axis component.
        point_inRay = torch.tensor([1.0, -2.0, 5.0])
        projection_inRay = torch.tensor([0.0, 0.0, 5.0])
        toParent_fromRay = ray_node.get_transform_toParent_fromSelf()
        point_inParent = toParent_fromRay.transform(point_inRay)
        projection_inParent = toParent_fromRay.transform(projection_inRay)

        projection_inParent_ = \
            ray_node.project_to_ray_inParent(point_inParent)
        self.assertTrue(
            torch.allclose(projection_inParent, projection_inParent_)
        )

        # Create a point along a ray and verify that its projection onto the
        # ray is the same point. (Also test the method is_point_on_ray)
        ray_node = self.root.children[0].children[0].children[-1]
        origin_inParent, direction_inParent = \
            ray_node.get_origin_and_direction_inParent()
        point_along_ray_inParent = origin_inParent + direction_inParent * 10

        projection_inParent = \
            ray_node.project_to_ray_inParent(point_along_ray_inParent)
        self.assertTrue(
            torch.allclose(
                projection_inParent,
                point_along_ray_inParent,
                atol=100 * core.EPS
            )
        )

        # Move a point orthogonally to the ray's direction and verify that
        # its projection does not change.
        point_inParent = torch.tensor([-3.0, 2.0, 15.0])
        projection_inParent = \
            ray_node.project_to_ray_inParent(point_inParent)

        _, direction_inParent = \
            ray_node.get_origin_and_direction_inParent()
        orthogonal_direction_inParent = \
            core.normalize(
                torch.stack(
                    (-direction_inParent[1], direction_inParent[0], core.T0)
                )
            )
        new_point_inParent = \
            point_inParent + orthogonal_direction_inParent * 10
        projection_inParent_ = \
            ray_node.project_to_ray_inParent(new_point_inParent)

        self.assertTrue(
            torch.allclose(projection_inParent, projection_inParent_)
        )


if __name__ == "__main__":
    unittest.main()
