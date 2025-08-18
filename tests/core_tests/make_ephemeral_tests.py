"""make_ephemeral_tests.py.

Unit tests for methods and classes in make_ephemera.py.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from tests.core_tests import node_tests


class Test_make_ephemeral(node_tests.TestNode):
    def setUp(self):
        super().setUp()

    def test_make_ephemeral_new_node(self):
        """test_make_ephemeral_new_node.

        Test the creation and automatic deletion of an ephemeral node.
        """

        # Create an ephemeral node that is a (new) child of child 1 of child 1.
        child_1_of_child_0 = self.root.children[1].children[1]
        with core.make_ephemeral(
            core.Node(
                parent=child_1_of_child_0,
                name="child 0 of child 1 of child 1"
            )
        ):  # type: ignore
            traversal = (
                "root",
                "child 0",
                "child 0 of child 0",
                "child 1 of child 0",
                "child 1",
                "child 0 of child 1",
                "child 1 of child 1",
                "child 0 of child 1 of child 1",
                "child 2 of child 1"
            )

            traversal_ = self.root.traverse(mode="depth (not breadth)")

            # Inside the context, "child 0 of child 1 of child 1" is present.
            for name, node_obj_ in zip(traversal, traversal_):
                self.assertTrue(name == node_obj_.name)

        # Outside context, we perform the regular test.
        super().test_traverse()

    def test_make_ephemeral_current_branch(self):
        """test_make_ephemeral_current_branch.

        Make an already existing branch of the three ephemeral within a context
        manager. Outside the context the branch will be deleted.
        """

        with core.make_ephemeral(self.root.children[1]):  # type: ignore
            traversal = (
                "root",
                "child 0",
                "child 0 of child 0",
                "child 1 of child 0",
                "child 1",
                "child 0 of child 1",
                "child 1 of child 1",
                "child 2 of child 1"
            )

            traversal_ = self.root.traverse(mode="depth (not breadth)")

            # Branch still exists inside context.
            for name, node_obj_ in zip(traversal, traversal_):
                self.assertTrue(name == node_obj_.name)

        # Branch is gone outside context:
        traversal = (
                "root",
                "child 0",
                "child 0 of child 0",
                "child 1 of child 0"
            )

        traversal_ = self.root.traverse(mode="depth (not breadth)")
        self.assertTrue(len(traversal_) == len(traversal))

        for name, node_obj_ in zip(traversal, traversal_):
            self.assertTrue(name == node_obj_.name)
