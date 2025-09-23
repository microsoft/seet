"""node_tests.py.

Unit tests for methods and classes in node.py
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from tests import tests_utils
import torch
import unittest


class TestNode(tests_utils.TestUtils):
    def setUp(self):
        super().setUp()

    def test_traverse(self):
        # From root, depth first should return
        #
        # (root, child 0, child 0 of child 0, child 1 of child 0, child 1,
        #   child 0 of child 1, child 1 of child 1, child 2 of child 1)
        traversal = (
            "root",
            "child 0",
            "child 0 of child 0", "child 1 of child 0",
            "child 1",
            "child 0 of child 1", "child 1 of child 1", "child 2 of child 1"
        )
        traversal_ = self.root.traverse()

        for name, parent_node in zip(traversal, traversal_):
            self.assertTrue(name == parent_node.name)

        # Traversal with unknown mode defaults to depth.
        traversal_ = self.root.traverse(mode="garbage")

        for name, parent_node in zip(traversal, traversal_):
            self.assertTrue(name == parent_node.name)

        # From root, breadth first should return
        #
        # (root, child 0, child 1, child 0 of child 0, child 1 of child 0,
        #   child 0 of child 1, child 1 of child 1, child 2 of child 1)
        traversal = (
            "root",
            "child 0", "child 1",
            "child 0 of child 0", "child 1 of child 0",
            "child 0 of child 1", "child 1 of child 1", "child 2 of child 1"
        )
        traversal_ = self.root.traverse(mode="breadth")

        for name, parent_node in zip(traversal, traversal_):
            self.assertTrue(name == parent_node.name)

    def test_sample_points(self):
        # Base class does not have points, just the method to sample them.
        pass

    def test_get_root(self):
        # Use the last child of the last child to access the root.
        last_last = self.root.children[-1].children[-1]

        root = last_last.get_root()

        self.assertTrue(root.name == self.root.name)

    def test_update_transform_toParent_fromSelf(self):
        # The transform from the first child of the last child of the root
        # concatenates 9 transformations to the world. Let's make that 10:
        first_last = self.root.children[-1].children[0]
        first_last.update_transform_toParent_fromSelf(self.transform)

        # Check:
        transform = core.SE3.create_identity()
        for _ in range(10):
            transform = \
                core.SE3.compose_transforms(transform, self.transform)

        transform_ = first_last.get_transform_toWorld_fromSelf()

        transform_matrix = transform.transform_matrix
        transform_matrix_ = transform_.transform_matrix

        self.assertTrue(torch.allclose(transform_matrix, transform_matrix_))

    def test_translate_inParent(self):
        # Apply a translation then remove it, to check that that doesn't do
        # anything.
        translation_vector = torch.tensor([1.0, 2.0, 3.0])
        self.root.translate_inParent(translation_vector)
        self.root.translate_inParent(-translation_vector)

        transform_matrix = torch.eye(4)
        transform_matrix_ = \
            self.root.transform_toParent_fromSelf.transform_matrix

        self.assertTrue(torch.allclose(transform_matrix, transform_matrix_))

        # Apply a translation via a transform, and remove it directly.
        transform_matrix = \
            core.stack_tensors(
                (
                    (torch.eye(3), translation_vector.view(3, 1)),
                    (torch.zeros(1, 3), core.T1.view(1, 1))
                )
            )
        transform = core.SE3(transform_matrix)
        self.root.update_transform_toParent_fromSelf(transform)
        self.root.translate_inParent(-translation_vector)

        transform_matrix = torch.eye(4)
        transform_matrix_ = \
            self.root.transform_toParent_fromSelf.transform_matrix

        self.assertTrue(torch.allclose(transform_matrix, transform_matrix_))

    def test_rotate_inParent(self):
        # Apply a rotation then remove it, to check that that doesn't do
        # anything.
        axis_angle_rad = torch.tensor([0.1, 0.2, 0.3])
        rotation_matrix = core.rotation_matrix(axis_angle_rad)
        self.root.rotate_inParent(rotation_matrix)
        self.root.rotate_inParent(rotation_matrix.T)

        transform_matrix = torch.eye(4)
        transform_matrix_ = \
            self.root.transform_toParent_fromSelf.transform_matrix

        self.assertTrue(torch.allclose(transform_matrix, transform_matrix_, rtol=core.EPS, atol=core.EPS))

        # Apply a rotation via a transform, and remove it directly.
        transform_matrix = \
            core.stack_tensors(
                (
                    (rotation_matrix, torch.zeros(3, 1)),
                    (torch.zeros(1, 3), core.T1.view(1, 1))
                )
            )
        transform = core.SE3(transform_matrix)
        self.root.update_transform_toParent_fromSelf(transform)
        self.root.rotate_inParent(rotation_matrix.T)

        transform_matrix = torch.eye(4)
        transform_matrix_ = \
            self.root.transform_toParent_fromSelf.transform_matrix

        self.assertTrue(
            torch.allclose(
                transform_matrix,
                transform_matrix_,
                rtol=core.EPS,  # Need some wiggle room.
                atol=core.EPS
            )
        )

    def test_list_ancestry_fromSelf_toRoot(self):
        # From child 0 of child 1 we should get (child 0 of child 1, child 1,
        # root)
        child_0_of_child_1 = self.root.children[-1].children[0]
        ancestry_list = child_0_of_child_1.list_ancestry_fromSelf_toRoot()
        self.assertTrue(ancestry_list[0].name == "child 0 of child 1")
        self.assertTrue(ancestry_list[1].name == "child 1")
        self.assertTrue(ancestry_list[2].name == "root")

    def test_is_ancestor_of(self):
        # Root is an ancestor to everybody other than itself.
        for child in self.root.children:
            self.assertTrue(self.root.is_ancestor_of(child))
            for grandchild in child.children:
                self.assertTrue(self.root.is_ancestor_of(grandchild))
                self.assertTrue(child.is_ancestor_of(grandchild))

        # Children are not ancestors of other children or other children's
        # grandchildren.
        first_child = self.root.children[0]
        for i in range(1, len(self.root.children)):
            other_child = self.root.children[i]
            self.assertFalse(first_child.is_ancestor_of(other_child))
            for grandchild in other_child.children:
                self.assertFalse(first_child.is_ancestor_of(grandchild))

        # Children are not ancestors of their parents or grandparents.
        for child in self.root.children:
            self.assertFalse(child.is_ancestor_of(self.root))
            for grandchild in child.children:
                self.assertFalse(grandchild.is_ancestor_of(self.root))
                self.assertFalse(grandchild.is_ancestor_of(child))

    def test_find_closest_common_ancestor(self):
        # Common ancestor of cousins is grandparent (root).
        cousin_1 = self.root.children[0].children[0]
        cousin_2 = self.root.children[1].children[0]

        common_ancestor = cousin_1.find_closest_common_ancestor(cousin_2)
        self.assertTrue(common_ancestor == self.root)

        common_ancestor = cousin_2.find_closest_common_ancestor(cousin_1)
        self.assertTrue(common_ancestor == self.root)

        # Common ancestor of siblings is parent.
        sibling_1 = self.root.children[0]
        sibling_2 = self.root.children[1]

        common_ancestor = sibling_1.find_closest_common_ancestor(sibling_2)
        self.assertTrue(common_ancestor == self.root)

        common_ancestor = sibling_2.find_closest_common_ancestor(sibling_1)
        self.assertTrue(common_ancestor == self.root)

        parent = self.root.children[1]
        sibling_1 = parent.children[0]
        sibling_2 = parent.children[-1]

        common_ancestor = sibling_1.find_closest_common_ancestor(sibling_2)
        self.assertTrue(common_ancestor == parent)

        common_ancestor = sibling_2.find_closest_common_ancestor(sibling_1)
        self.assertTrue(common_ancestor == parent)

        # Common ancestor of child and parent is parent.
        common_ancestor = parent.find_closest_common_ancestor(sibling_1)
        self.assertTrue(common_ancestor == parent)

        common_ancestor = sibling_1.find_closest_common_ancestor(parent)
        self.assertTrue(common_ancestor == parent)

    def test_get_transform_toOther_fromSelf(self):
        # The transform from the last child of the last child to the first last
        # child of the first child concatenates T^-3 o T^-1 o T^4 o T^7
        # transforms, for a total of T^7.
        last_child_of_last_child = self.root.children[-1].children[-1]
        last_child_of_first_child = self.root.children[0].children[-1]

        transform = \
            last_child_of_last_child.get_transform_toOther_fromSelf(
                last_child_of_first_child
            )

        transform_ = core.SE3.create_identity()
        for _ in range(7):  # There must be 7 transformations!
            transform_ = \
                core.SE3.compose_transforms(transform_, self.transform)

        transform_matrix = transform.transform_matrix
        transform_matrix_ = transform_.transform_matrix

        self.assertTrue(
            torch.allclose(
                transform_matrix, transform_matrix_, atol=100 * core.EPS
            )
        )

    def test_get_transform_toWorld_fromSelf(self):
        # Transform to world from root is identity.
        identity_ = self.root.transform_toParent_fromSelf.transform_matrix
        device = identity_.device
        identity = torch.eye(4, device=device)
        self.assertTrue(torch.allclose(identity, identity_))

        # Last child of last child (last grandchild) of root concatenates 11
        # transforms to root (world) from self:
        transform = core.SE3.create_identity()
        for _ in range(11):
            transform = core.SE3.compose_transforms(transform, self.transform)

        child_2_of_child_1 = self.root.children[-1].children[-1]
        transform_ = child_2_of_child_1.get_transform_toWorld_fromSelf()

        transform_matrix = transform.transform_matrix
        transform_matrix_ = transform_.transform_matrix

        # Ensure both matrices are on the same device
        transform_matrix = transform_matrix.to(transform_matrix_.device)
        self.assertTrue(torch.allclose(transform_matrix, transform_matrix_))

    def test_remove_child(self):
        num_children = len(self.root.children)
        child_to_be_removed = self.root.children[0]
        self.root.remove_child(child_to_be_removed)

        new_num_children = len(self.root.children)
        self.assertTrue(new_num_children == num_children - 1)

        # Removed child is now parentless.
        self.assertTrue(child_to_be_removed.parent is None)

        # The last child of the removed child concatenates 3 transforms to
        # world from self.
        transform = core.SE3.create_identity()
        for _ in range(3):
            transform = \
                core.SE3.compose_transforms(transform, self.transform)

        child_2_of_removed_child = child_to_be_removed.children[-1]
        transform_ = child_2_of_removed_child.get_transform_toWorld_fromSelf()

        transform_matrix = transform.transform_matrix
        transform_matrix_ = transform_.transform_matrix

        self.assertTrue(torch.allclose(transform_matrix, transform_matrix_))

    def test_add_child(self):
        # Let's rearrange the tree by attaching the last child of the first
        # child directly to the root node. We use the first child because after
        # the attachment the last child changes, but the first child remains
        # the same.
        num_children_root = len(self.root.children)
        num_children_first_child = len(self.root.children[0].children)
        child_to_be_rearranged = self.root.children[0].children[-1]
        self.root.add_child(child_to_be_rearranged)

        # The number of children of the first child must have dropped by 1.
        new_num_children_first_child = len(self.root.children[0].children)

        self.assertTrue(
            new_num_children_first_child == num_children_first_child - 1
        )

        # The number of children of the root node must have increased by 1.
        new_num_children_root = len(self.root.children)

        self.assertTrue(new_num_children_root == num_children_root + 1)

        # The node that was reattached concatenated 4 transform to the world.
        transform = core.SE3.create_identity()
        for _ in range(4):
            transform = \
                core.SE3.compose_transforms(transform, self.transform)

        # Since the parent of the reattached node is now root, its transform to
        # parent from self must be the same as the concatenated transform.
        transform_ = child_to_be_rearranged.transform_toParent_fromSelf

        transform_matrix = transform.transform_matrix
        transform_matrix_ = transform_.transform_matrix

        self.assertTrue(torch.allclose(transform_matrix, transform_matrix_))

    def test_branchcopy(self):
        """test_branchcopy.

        Test branchcopying of a node and its descendants.
        """

        child_0 = self.root.children[0]
        child_0_copy = child_0.branchcopy(child_0)

        # Root is the same object, not different copies of the same object.
        self.assertTrue(child_0.get_root() == child_0_copy.get_root())

        # But the nodes themselves are different copies of the same object.
        self.assertFalse(child_0 == child_0_copy)
        self.assertTrue(len(child_0.children) == len(child_0_copy.children))

        # The spatial relationship between the last descendant of each copy and
        # the root node is the same.
        last_0 = child_0.children[-1]
        last_0_copy = child_0_copy.children[-1]
        self.assertFalse(last_0 == last_0_copy)

        T = last_0.get_transform_toOther_fromSelf(self.root)
        T_copy = last_0_copy.get_transform_toOther_fromSelf(self.root)
        self.assertTrue(
            torch.allclose(T.transform_matrix, T_copy.transform_matrix)
        )


if __name__ == "__main__":
    unittest.main()
