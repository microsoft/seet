"""direction_tests.py

Unit tests for Direction class
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.primitives as primitives
from tests import tests_utils
import torch
import unittest


class TestDirection(tests_utils.TestUtils):
    def setUp(self):
        super().setUp()

        nodes = self.root.traverse()
        for parent_node in nodes:
            # Create and attach a direction to each node in the pose graph.
            primitives.Direction(
                parent_node,
                self.transform,
                name=f"direction in {parent_node.name}"
            )

    def test_create_from_components_inParent(self):
        # Get some arbitrary leaf node.
        leaf = self.root.children[1].children[2]

        # Create some arbitrary components in the coordinate system of the
        # leaf node.
        arbitrary_vector = torch.tensor([1.0, -2.0, 3.0])
        components_inLeaf = core.normalize(arbitrary_vector)

        direction_node = \
            primitives.Direction.create_from_components_inParent(
                leaf, components_inLeaf
            )
        components_inRoot = \
            direction_node.get_components_inOther(self.root)

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
        #  child 0   child 1                  | rot from (0, 0, 1) to (1, 2, 3)
        #  of child 0  of child 0              \
        #                                 direction_node

        # Coordinates of point in root node should be (1, 2, 3) transformed by
        # T eleven times.
        components_inRoot_ = components_inLeaf  # Temporarily.
        for _ in range(11):
            components_inRoot_ = \
                self.transform.rotation.transform(components_inRoot_)

        self.assertTrue(
            torch.allclose(components_inRoot, components_inRoot_)
        )

    def test_get_components_inOther(self):
        """test_get_components_inOther.

        Test manipulation of coodrinate system.
        """
        # Get an arbitrary direction on a leaf node.
        leaf = self.root.children[1].children[0]
        direction_inLeaf = leaf.children[-1]

        # Get its components in the coordinate system of root.
        components_inRoot = direction_inLeaf.get_components_inOther(self.root)

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
        #    |         \            \
        #  child 0   child 1          \ T
        #  of child 0  of child 0       \
        #                             direction_inLeaf

        # Coordinates of point in root node should be (0, 0, 0) transformed by
        # T ten times.

        components_inRoot_ = torch.tensor([0.0, 0.0, 1.0])
        for _ in range(10):
            components_inRoot_ = \
                self.transform.rotation.transform(components_inRoot_)

        self.assertTrue(
            torch.allclose(components_inRoot, components_inRoot_)
        )


if __name__ == "__main__":
    unittest.main()
