"""point_tests.py

Unit tests for Point class
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.primitives as primitives
from tests import tests_utils
import torch
import unittest


class TestPoint(tests_utils.TestUtils):
    def setUp(self):
        super().setUp()

        nodes = self.root.traverse()  # Depth first
        for parent_node in nodes:
            # Create and attach a point to each node in the pose graph.
            primitives.Point(
                parent_node,
                self.transform,
                name=f"point in {parent_node.name}"
            )

    def test_create_from_coordinates_inParent(self):
        """test_create_from_coordinates_inParent.

        Test creation of a point from its coordinates in its parent
        coordinate system.
        """
        # Get some arbitrary leaf node.
        leaf = self.root.children[1].children[2]

        # Create some arbitrary coordinates in the coordinate system of the
        # leaf node.
        coordinates_inLeaf = torch.tensor([1.0, -2.0, 3.0])

        point_node = primitives.Point.create_from_coordinates_inParent(
            leaf, coordinates_inLeaf
        )
        coordinates_inRoot = point_node.get_coordinates_inOther(self.root)

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
        #  child 0   child 1                   \ Translation by (1, 2, 3)
        #  of child 0  of child 0               \
        #                                   point_node

        # Coordinates of point in root node should be (1, 2, 3) transformed by
        # T eleven times.
        coordinates_inRoot_ = coordinates_inLeaf  # Temporarily.
        for _ in range(11):
            coordinates_inRoot_ = self.transform.transform(coordinates_inRoot_)

        self.assertTrue(
            torch.allclose(coordinates_inRoot, coordinates_inRoot_)
        )

    def test_get_coordinates_inOther(self):
        """test_get_coordinates_inOther.

        Test manipulation of coordinate system.
        """
        # Get an arbitrary point on a leaf node.
        leaf = self.root.children[0].children[0]
        point_inLeaf = leaf.children[-1]

        # Get its coordinates in the coordinate system of root.
        coordinates_inRoot = point_inLeaf.get_coordinates_inOther(self.root)

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
        # T  |
        #    |
        #  point_inLeaf

        # Coordinates of point in root node should be (0, 0, 0) transformed by
        # T four times.

        coordinates_inRoot_ = torch.zeros(3)
        for _ in range(4):
            coordinates_inRoot_ = self.transform.transform(coordinates_inRoot_)

        self.assertTrue(
            torch.allclose(coordinates_inRoot, coordinates_inRoot_)
        )

    def test_get_coordinates_inParent(self):
        # Create a point with arbitrary coordinates on a leaf node.
        leaf = self.root.children[0].children[0]
        point_node = leaf.children[-1]
        point_inLeaf = point_node.get_coordinates_inParent()

        # Coordinates of point in root node should be point_inLeaf
        # transformed by T three times.
        point_inRoot = point_node.get_coordinates_inOther(self.root)

        point_inRoot_ = point_inLeaf  # Temporarily.
        for _ in range(3):
            point_inRoot_ = self.transform.transform(point_inRoot_)

        self.assertTrue(
            torch.allclose(point_inRoot, point_inRoot_)
        )


if __name__ == "__main__":
    unittest.main()
