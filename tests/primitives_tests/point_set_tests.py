"""point_set_tests.py

Unit tests for PointSet class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.primitives as primitives
from tests import tests_utils
import torch
import unittest


class TestPointSet(tests_utils.TestUtils):
    def setUp(self):
        super().setUp()

        nodes = self.root.traverse()  # Depth first
        for parent_node in nodes:
            # Create and attach points sets to each node in the pose graph.
            coordinates = torch.randn((3, 10))
            primitives.PointSet(
                parent_node,
                self.transform,
                coordinates,
                name=f"point set in {parent_node.name}"
            )

        # The pose graph looks like this:
        #
        #              root--->PointSet
        #               | \
        #               |   \
        #             T |     \ T4
        #               |       \
        #               |         \
        # PointSet<---child 0    child 1--->PointSet | \           \
        #               |   \           \ T5, T6, T7
        #            T2 | T3  \           \
        #               |       \       (children 0, 1, 2 of child 1)
        #               |         \                \   \   \
        #             child 0     child 1       PtSt  PtSt  PtSt
        #             of child 0  of child 0
        #               |             |
        #               V             V
        #            PointSet      PointSet

    def test_create_from_coordinates_inParent(self):
        """test_create_from_coordinates_inParent.

        Test class method to create point set from coordinates in parent.
        """
        parent_node = self.root.children[1].children[-1]  # Itself a PointSet
        point_set = \
            primitives.PointSet.create_from_coordinates_inParent(
                parent_node, torch.randn((3, 10)), name="new point set"
            )

        coordinates_inChild1PointSet = point_set.get_coordinates_inParent()
        N = coordinates_inChild1PointSet.shape[1]
        distance_graph = torch.empty((N, N))
        for i in range(N):
            pt_i = coordinates_inChild1PointSet[:, i]
            for j in range(N):
                pt_j = coordinates_inChild1PointSet[:, j]
                distance_graph[i, j] = torch.linalg.norm(pt_i - pt_j)

        # Get coordinates in a different coordinate system.
        coordinates_inChild0 = \
            point_set.get_coordinates_inOther(self.root.children[0])
        distance_graph_ = torch.empty((N, N))
        for i in range(N):
            pt_i = coordinates_inChild0[:, i]
            for j in range(N):
                pt_j = coordinates_inChild0[:, j]
                distance_graph_[i, j] = torch.linalg.norm(pt_i - pt_j)

        self.assertTrue(
            torch.allclose(
                distance_graph,
                distance_graph_,
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )

        # Attach the point set to a different node.
        self.root.children[1].children[2].add_child(point_set)
        coordinates_inNewParent = point_set.get_coordinates_inParent()

        # Coordinates have changed.
        for i in range(N):
            self.assertFalse(
                torch.allclose(
                    coordinates_inNewParent[:, i],
                    coordinates_inChild1PointSet[:, i]
                )
            )

        # Distance graph remains.
        distance_graph_ = torch.empty((N, N))
        for i in range(N):
            pt_i = coordinates_inNewParent[:, i]
            for j in range(N):
                pt_j = coordinates_inNewParent[:, j]
                distance_graph_[i, j] = torch.linalg.norm(pt_i - pt_j)

        self.assertTrue(
            torch.allclose(
                distance_graph,
                distance_graph_,
                rtol=core.EPS * 100,
                atol=core.EPS * 100
            )
        )


if __name__ == "__main__":
    unittest.main()
