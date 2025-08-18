"""Unit tests from methods and classes in plane.py.

Unit tests from methods and classes in plane.py
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.primitives as primitives
from tests import tests_utils
import torch
import unittest


class TestPlane(tests_utils.TestUtils):
    def setUp(self):
        super().setUp()

        nodes = self.root.traverse()  # Depth first.
        for parent_node in nodes:
            # Create the planes, assign them as children of the nodes of the,
            # tree.
            primitives.Plane(
                parent_node,
                self.transform,
                name=f"plane in {parent_node.name}"
            )

    def test_create_from_origin_and_normal_inParent(self):
        """test_create_from_origin_and_normal_inParent.

        Test creation of plane.
        """
        # Create a trivial plane
        plane_primitives = primitives.Plane(
            self.root,
            core.SE3.create_identity(),
            name=f"plane in {self.root.name}"
        )

        # Create the same trivial plane from trivial origin and normal.
        origin_inParent = torch.zeros(3)
        normal_inParent = torch.tensor([0.0, 0.0, 1.0])
        plane_primitives_ = \
            primitives.Plane.create_from_origin_and_normal_inParent(
                self.root,
                origin_inParent,
                normal_inParent
            )

        self.assertTrue(
            torch.allclose(plane_primitives.origin, plane_primitives_.origin)
        )
        self.assertTrue(
            torch.allclose(plane_primitives.normal, plane_primitives_.normal)
        )

    def test_get_orthonormal_inOther(self):
        """test_get_orthonormal_inOther.

        Test computation of plane's internal orthonormal coordinate system
        expressed in some other coordinate system.
        """
        # Get an arbitrary plane.
        parent_node = self.root.children[1].children[2]
        plane_node = parent_node.children[-1]
        orthonormal_inRoot = plane_node.get_orthonormal_inOther(self.root)

        #  The tree looks like this:
        #
        #   root
        #    | \
        #    |   \
        #  T |     \ T4
        #    |       \
        #    |         \
        #  child 0    child 1
        #    | \           \
        #    |   \           \ T5, T6, T7
        # T2 | T3  \           \
        #    |       \       (children 0, 1, 2 of child 1)
        #    |         \                      \
        #  child 0   child 1                 T \
        #  of child 0  of child 0               \
        #                                     plane

        # The orthonormal coordinate system of the plane is rotated twelve
        # times.
        orthonormal_inRoot_ = plane_node.orthonormal  # Temporarily
        rotation = self.transform.rotation
        for _ in range(12):
            orthonormal_inRoot_ = rotation.transform(orthonormal_inRoot_)

        self.assertTrue(
            torch.allclose(orthonormal_inRoot, orthonormal_inRoot_))

    def test_normalize_homogeneous_coordinate(self):
        """test_normalize_homogeneous_coordinate.

        Test normalization of homogeneous coordinates of plane.
        """
        # Get an arbitrary plane.
        parent_node = self.root.children[1].children[2]
        plane_node = parent_node.children[-1]
        homogeneous_coordinates_inRoot = \
            plane_node.get_homogeneous_coordinates_inOther(self.root)
        normalized_homogeneous_coordinates_inRoot = \
            primitives.Plane.normalize_homogeneous_coordinates(
                homogeneous_coordinates_inRoot
            )

        # Last entry must not be non-negative.
        self.assertTrue(
            normalized_homogeneous_coordinates_inRoot[-1] <= core.T0
        )

        # First three entries must be a unit vector.
        norm = torch.linalg.norm(
            normalized_homogeneous_coordinates_inRoot[:3]
        )
        self.assertTrue(torch.allclose(norm, core.T1))

    def test_create_from_homogeneous_coordinate_inParent(self):
        """test_create_from_homogeneous_coordinate_inParent.

        Test creation of plane from homogeneous coordinates.
        """
        sqrt_3 = torch.sqrt(torch.tensor(3.0))
        plane_homogeneous_inParent = \
            torch.stack(
                (
                    core.T1 / sqrt_3,
                    core.T1 / sqrt_3,
                    core.T1 / sqrt_3,
                    -sqrt_3
                )
            )
        plane_primitives = \
            primitives.Plane.create_from_homogeneous_coordinates_inParent(
                self.root,
                plane_homogeneous_inParent
            )
        origin_inParent_, normal_inParent_ = \
            plane_primitives.get_origin_and_normal_inParent()

        origin_inParent = torch.ones(3)
        normal_inParent = torch.ones(3) / torch.sqrt(torch.tensor(3.0))

        self.assertTrue(torch.allclose(origin_inParent, origin_inParent_))
        self.assertTrue(torch.allclose(normal_inParent, normal_inParent_))

    def test_get_origin_and_normal_inOther(self):
        """test_get_origin_and_normal_inOther.

        Test getter method for obtaining origin and normal direction of
        plane in an arbitrary coordinate system.
        """
        # Get an arbitrary plane.
        parent_node = self.root.children[0].children[0]
        plane_primitives = parent_node.children[-1]

        origin_inRoot, normal_inRoot = \
            plane_primitives.get_origin_and_normal_inOther(self.root)

        #  The tree looks like this:
        #
        #   root
        #    | \
        #    |   \
        #  T |     \ T4
        #    |       \
        #    |         \
        #  child 0    child 1
        #    | \           \
        #    |   \           \ T5, T6, T7
        # T2 | T3  \           \
        #    |       \       (children 0, 1, 2 of child 1)
        #    |         \
        #  child 0   child 1
        #  of child 0  of child 0
        #    |
        #  T |
        #    |
        #  plane

        # The origin and normal of the plane are (0, 0, 0) and (0, 0, 1)
        # appropriately transformed by T four times.

        origin_inRoot_, normal_inRoot_ = \
            torch.zeros(3), torch.tensor([0.0, 0.0, 1.0])  # Temporarily.
        for _ in range(4):
            origin_inRoot_ = self.transform.transform(origin_inRoot_)
            normal_inRoot_ = \
                self.transform.rotation.transform(normal_inRoot_)

        self.assertTrue(torch.allclose(origin_inRoot, origin_inRoot_))
        self.assertTrue(torch.allclose(normal_inRoot, normal_inRoot_))

    def test_get_origin_and_normal_inParent(self):
        """test_get_origin_and_normal_inParent.

        Test getter method for obtaining origin and normal direction of
        plane in plane's parent coordinate system.
        """
        origin_inParent = torch.tensor([1.0, 2.0, 3.0])
        direction_inParent = torch.tensor([-3.0, 2.0, -1.0])
        normal_inParent = core.normalize(direction_inParent)

        plane_primitives = \
            primitives.Plane.create_from_origin_and_normal_inParent(
                self.root, origin_inParent, normal_inParent
            )

        origin_inParent_, normal_inParent_ = \
            plane_primitives.get_origin_and_normal_inParent()

        self.assertTrue(torch.allclose(origin_inParent, origin_inParent_))
        self.assertTrue(torch.allclose(normal_inParent, normal_inParent_))

    def test_get_homogeneous_coordinates_inOther(self):
        """test_get_homogeneous_coordinates_inOther.

        Test representation of plane in homogeneous coordinates in the
        coordinate system of another node.
        """
        # Get an arbitrary plane.
        parent_node = self.root.children[0].children[0]
        plane_primitives = parent_node.children[-1]

        # Get the coordinates of the plane in the coordinate system of root.
        homogeneous_plane_inRoot = \
            plane_primitives.get_homogeneous_coordinates_inOther(self.root)

        #  The tree looks like this:
        #
        #   root
        #    | \
        #    |   \
        #  T |     \ T4
        #    |       \
        #    |         \
        #  child 0    child 1
        #    | \           \
        #    |   \           \ T5, T6, T7
        # T2 | T3  \           \
        #    |       \       (children 0, 1, 2 of child 1)
        #    |         \
        #  child 0   child 1
        #  of child 0  of child 0
        #    |
        #  T |
        #    |
        #  plane

        # The plane in covariant homogeneous coordinates, so it transforms
        # according with the inverse of the direct transform. In its own
        # coordinate system the homogeneous coordinates of the plane are (0, 0,
        # 1, 0).
        homogeneous_plane_inRoot_ = \
            torch.tensor([0.0, 0.0, 1.0, 0.0])  # Temporarily.
        covariant_transform_matrix = self.transform.inverse_transform_matrix.T
        for _ in range(4):
            homogeneous_plane_inRoot_ = \
                covariant_transform_matrix @ homogeneous_plane_inRoot_

        normalized_plane_inRoot = \
            homogeneous_plane_inRoot / homogeneous_plane_inRoot[-1]
        normalized_plane_inRoot_ = \
            homogeneous_plane_inRoot_ / homogeneous_plane_inRoot_[-1]

        self.assertTrue(
            torch.allclose(
                normalized_plane_inRoot, normalized_plane_inRoot_
            )
        )

    def test_get_homogeneous_coordinates_inParent(self):
        """test_get_homogeneous_coordinates_inParent.

        Test extraction of homogeneous coordinates of plane in coordinate
        system of plane's parent.
        """
        # Get an arbitrary plane.
        parent_node = self.root.children[0].children[0]
        plane_primitives = parent_node.children[-1]

        # Get an arbitrary point on the plane.
        origin_inParent, normal_inParent = \
            plane_primitives.get_origin_and_normal_inParent()

        # Take an orthogonal but otherwise arbitrary direction to the plane's
        # normal.
        direction_inParent = torch.tensor([1.0, -2.0, 3.0])
        orthogonal_to_normal_inParent = \
            torch.cross(direction_inParent, normal_inParent)

        arbitrary_inParent = \
            origin_inParent + 10 * orthogonal_to_normal_inParent

        # In homogeneous coordinates, plane p_h and point x_h lying on the
        # plane satisfy the equation p_h.T * x_h = 0.
        homogeneous_plane_inParent = \
            plane_primitives.get_homogeneous_coordinates_inParent()
        homogeneous_point_inParent = core.homogenize(arbitrary_inParent)

        self.assertTrue(
            torch.allclose(
                homogeneous_plane_inParent @ homogeneous_point_inParent,
                core.T0,
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

    def test_is_point_on_plane(self):
        # Get an arbitrary plane.
        parent_node = self.root.children[0].children[0]
        plane_primitives = parent_node.children[-1]
        # Take an arbitrary point on the plane.
        point_inPlane = torch.tensor([1.0, -2.0, 0.0])
        point_inPlaneParent = \
            plane_primitives.transform_toParent_fromSelf.transform(
                point_inPlane
            )

        self.assertTrue(
            plane_primitives.is_point_on_plane_inParent(point_inPlaneParent)
        )

    def test_compute_signed_distance_to_point(self):
        """test_compute_signed_distance_to_point.

        Test computation of signed distances of points to plane.
        """
        # Take an arbitrary node in the tree.
        parent_node = self.root.children[0].children[0]
        # Take a plane whose parent is the arbitrary node.
        plane_primitives = parent_node.children[-1]

        # Create a point with signed distance equal to -1.
        point_inPlane = torch.tensor([100.0, -1000.0, -1])
        distance = \
            plane_primitives.compute_signed_distance_to_point_inPlane(
                point_inPlane
            )

        self.assertTrue(torch.allclose(distance, -core.T1))

        # Create a point with signed distance equal to 3 and which is not a
        # child of plane.
        point_inPlane = -3 * point_inPlane
        point_inParent = \
            plane_primitives.transform_toParent_fromSelf.transform(
                point_inPlane
            )
        distance = \
            plane_primitives.compute_signed_distance_to_point_inParent(
                point_inParent
            )

        self.assertTrue(
            torch.allclose(distance, 3 * core.T1, atol=100 * core.EPS)
        )

    def test_intersect_from_origin_and_direction(self):
        """test_intersect_from_origin_and_direction.

        Test intersection of ray with plane, from origin and direction in
        the plane's coordinate system.
        """
        # Take an arbitrary plane.
        parent_node = self.root.children[0].children[0]
        plane_primitives = parent_node.children[-1]

        # Create arbitrary origin and direction in plane's coordinate system.
        origin_inPlane = torch.tensor([1.1, -2.2, 3.3])
        direction_inPlane = torch.tensor([-0.3, 0.2, -0.1])

        intersection_inPlane = \
            plane_primitives.intersect_from_origin_and_direction_inPlane(
                origin_inPlane, direction_inPlane
            )
        # Intersection is on plane.
        self.assertTrue(
            plane_primitives.is_point_on_plane_inPlane(intersection_inPlane)
        )

        # Intersection is on ray.
        ray = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                plane_primitives, origin_inPlane, direction_inPlane
            )
        self.assertTrue(
            ray.is_point_on_ray_inParent(intersection_inPlane)
        )
        plane_primitives.remove_child(ray)

        # Create an arbitrary origin and direction in the coordinate system of
        # the plane's parent.
        # Create arbitrary origin and direction in plane's coordinate system.
        origin_inParent = torch.tensor([1.1, -2.2, 3.3])
        direction_inParent = torch.tensor([-0.3, 0.2, -0.1])

        intersection_inParent = \
            plane_primitives.intersect_from_origin_and_direction_inParent(
                origin_inParent, direction_inParent
            )

        # Intersection is on plane.
        self.assertTrue(
            plane_primitives.is_point_on_plane_inParent(intersection_inParent)
        )

        # Intersection is on ray.
        ray = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                parent_node, origin_inParent, direction_inParent
            )
        self.assertTrue(
            ray.is_point_on_ray_inParent(intersection_inParent)
        )

    def test_intersect_ray_inParent(self):
        """test_intersect_ray_inParent.

        Test intersection of ray with plane.
        """

        # Create trivial plane and ray.
        origin_inParent = torch.zeros(3)
        direction_inParent = torch.tensor([0.0, 0.0, 1.0])
        normal_inParent = core.normalize(direction_inParent)
        plane_primitives = \
            primitives.Plane.create_from_origin_and_normal_inParent(
                self.root, origin_inParent, normal_inParent
            )
        ray_node = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self.root, origin_inParent, direction_inParent
            )
        intersection_inParent = \
            plane_primitives.intersect_ray_inParent(ray_node)

        # Intersection should be (0, 0, 0).
        if intersection_inParent is not None:
            self.assertTrue(
                torch.allclose(intersection_inParent, torch.zeros(3))
            )

        # Take an arbitrary node in the tree.
        parent_node = self.root.children[0].children[0]
        # Take a plane whose parent is the arbitrary node.
        plane_primitives = parent_node.children[-1]
        # Take an arbitrary point on the plane.
        point_inPlane = \
            torch.tensor([1.0, 0.1, 0.0])  # Last coordinate is zero
        origin_inParent = \
            plane_primitives.transform_toParent_fromSelf.transform(
                point_inPlane
            )
        # Take an arbitrary direction.
        direction_inParent = torch.tensor([1.0, -2.0, 7.0])
        # Create a ray from the point and direction with same parent as plane.
        ray_node = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                parent_node,
                origin_inParent,
                direction_inParent
            )
        # Intersection of ray with plane must be original point in plane.
        intersection_inParent = \
            plane_primitives.intersect_ray_inParent(ray_node)

        if intersection_inParent is not None:
            self.assertTrue(
                torch.allclose(origin_inParent, intersection_inParent)
            )

        # Take an arbitrary plane and an arbitrary ray in an arbitrary parent.
        origin_of_plane_inParent = torch.tensor([1.0, -2.0, 3.0])
        direction_of_plane_inParent = torch.tensor([-3.0, -1.0, -2.0])
        normal_of_plane_inParent = \
            core.normalize(direction_of_plane_inParent)
        plane_primitives = \
            primitives.Plane.create_from_origin_and_normal_inParent(
                self.root, origin_of_plane_inParent, normal_of_plane_inParent
            )
        origin_of_ray_inParent = torch.tensor([1.2, -3.4, -5.6])
        direction_of_ray_inParent = torch.tensor([-7.8, 9.1, -1.1])
        ray_node = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self.root, origin_of_ray_inParent, direction_of_ray_inParent
            )
        intersection_inParent = \
            plane_primitives.intersect_ray_inParent(ray_node)
        # Intersection must lie on plane.
        self.assertTrue(
            plane_primitives.is_point_on_plane_inParent(intersection_inParent)
        )

        # Intersection must lie on ray.
        self.assertTrue(
            ray_node.is_point_on_ray_inParent(intersection_inParent)
        )

    def test_intersect_ray_inOther(self):
        """test_intersect_ray_inOther.

        Test intersection of ray with plane.
        """

        # Create trivial plane and ray.
        origin_inParent = torch.zeros(3)
        direction_inParent = torch.tensor([0.0, 0.0, 1.0])
        normal_inParent = core.normalize(direction_inParent)
        plane_primitives = \
            primitives.Plane.create_from_origin_and_normal_inParent(
                self.root, origin_inParent, normal_inParent
            )
        ray_node = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self.root, origin_inParent, direction_inParent
            )

        # Other is the parent, but that does not matter.
        intersection_inOther = \
            plane_primitives.intersect_from_origin_and_direction_inOther(
                plane_primitives.parent,  # Other is parent.
                origin_inParent,
                direction_inParent
            )

        # Intersection should be (0, 0, 0).
        if intersection_inOther is not None:
            self.assertTrue(
                torch.allclose(intersection_inOther, torch.zeros(3))
            )

        # Take an arbitrary node in the tree.
        parent_node = self.root.children[0].children[0]
        # Take a plane whose parent is the arbitrary node.
        plane_primitives = parent_node.children[-1]
        # Take an arbitrary point on the plane.
        point_inPlane = \
            torch.tensor([1.0, 0.1, 0.0])  # Last coordinate is zero
        origin_inParent = \
            plane_primitives.transform_toParent_fromSelf.transform(
                point_inPlane
            )
        # Take an arbitrary direction.
        direction_inParent = torch.tensor([1.0, -2.0, 7.0])
        # Create a ray from the point and direction with same parent as plane.
        ray_node = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                parent_node,
                origin_inParent,
                direction_inParent
            )

        # Again, parent is other, but that, again, does not matter.
        #
        # Intersection of ray with plane must be original point in plane.
        intersection_inOther = \
            plane_primitives.intersect_from_origin_and_direction_inOther(
                plane_primitives.parent,
                origin_inParent,
                direction_inParent
            )

        if intersection_inOther is not None:
            # Other and parent are the same.
            self.assertTrue(
                torch.allclose(origin_inParent, intersection_inOther)
            )

        # Take an arbitrary plane and an arbitrary ray in an arbitrary parent.
        origin_of_plane_inParent = torch.tensor([1.0, -2.0, 3.0])
        direction_of_plane_inParent = torch.tensor([-3.0, -1.0, -2.0])
        normal_of_plane_inParent = \
            core.normalize(direction_of_plane_inParent)
        plane_primitives = \
            primitives.Plane.create_from_origin_and_normal_inParent(
                self.root, origin_of_plane_inParent, normal_of_plane_inParent
            )
        origin_of_ray_inParent = torch.tensor([1.2, -3.4, -5.6])
        direction_of_ray_inParent = torch.tensor([-7.8, 9.1, -1.1])
        ray_node = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self.root, origin_of_ray_inParent, direction_of_ray_inParent
            )
        intersection_inOther = \
            plane_primitives.intersect_from_origin_and_direction_inOther(
                plane_primitives.parent,
                origin_of_ray_inParent,
                direction_of_ray_inParent
            )
        # Intersection must lie on plane.
        self.assertTrue(
            plane_primitives.is_point_on_plane_inOther(
                intersection_inOther, plane_primitives.parent
            )
        )

        # Intersection must lie on ray.
        self.assertTrue(
            ray_node.is_point_on_ray_inOther(
                intersection_inOther, plane_primitives.parent
            )
        )


if __name__ == "__main__":
    unittest.main()
