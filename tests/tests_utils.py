"""tests_utils.py

Auxiliary classes and methods for unit tests.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import torch
import unittest


class TestUtils(unittest.TestCase):
    def setUp(self):
        # Create a pose graph, depth first.
        self.root = core.Node(name="root", )

        # "Small" perturbation in SE3.
        axis = torch.tensor([1.0, 2.0, 3.0])
        n_axis = core.normalize(axis)
        axis_angle_rad = n_axis * core.TPI / 180 * 10
        Rotation = core.rotation_matrix(axis_angle_rad)
        Translation = torch.tensor([-3.0, -2.0, -1.0]).view(3, 1) * 10
        self.transform = core.SE3(torch.hstack((Rotation, Translation)))

        # Root has two children.
        transform = core.SE3.create_identity()
        for i in range(2):
            child_name = "child {0:d}".format(i)
            transform = \
                core.SE3.compose_transforms(transform, self.transform)
            # Create and assign so that we can use later.
            child = \
                core.Node(
                    parent=self.root,
                    transform_toParent_fromSelf=transform,
                    name=child_name
                )

            # Each child has a few children.
            for j in range(i + 2):
                grandchild_name = \
                    "child {0:d} of {1:s}".format(j, child_name)
                transform = \
                    core.SE3.compose_transforms(transform, self.transform)
                # Create but no need to assign, as the created node is assigned
                # to the parent.
                core.Node(
                    parent=child,  # From outer loop.
                    transform_toParent_fromSelf=transform,
                    name=grandchild_name
                )

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

        super().setUp()
