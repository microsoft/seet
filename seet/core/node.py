"""node.py.

User-defined package for model Nodes
"""


__author__ = "Chris Aholt (chaholt@microsoft.com)"


from collections import deque
import io
import json
from seet.core import groups
import torch


class Node():
    """Node.

    Abstract class for representing nodes in a pose graph. This is a new-style
    class as it derives from object.
    """

    @staticmethod
    def open(file_name_dict_or_stream):
        """open.

        A general function to read parameters of Node objects either from
        configuration files, dictionaries, or from a file stream. This allows
        us to create configuration files on the fly.

        Args:
            file_name_dict_or_stream (str, dict, or stream): if a str, this is
            the name of the file with configuration parameters for the Node
            object. If a dictionary, it directly contains the parameters of a
            Node object. If a stream, this is the stream produced by opening a
            configuration file for the object.
        """
        if isinstance(file_name_dict_or_stream, str):
            return open(file_name_dict_or_stream)
        else:
            assert isinstance(file_name_dict_or_stream, dict)
            return Node.generate_stream(file_name_dict_or_stream)

    @staticmethod
    def generate_stream(json_dict):
        """generate_stream.

        Generate a text stream from a dictionary formatted as a json file. This
        means keys are strings, and values are strings, bools, None, ints,
        floats, lists of these types, or nested dictionaries also formatted as
        json files.

        Args:
            json_dict (dict): a dictionary of parameters formatted as a json
            file.
        """
        str_dict = json.dumps(json_dict)

        # Output as a file stream.
        return io.StringIO(str_dict)

    def __init__(
        self,
        parent=None,
        transform_toParent_fromSelf=None,
        name="",
        requires_grad=False
    ):
        """Initialize node.

        Args:
            parent (_type_, optional): _description_. Defaults to None.

            transform_toParent_fromSelf (_type_, optional): _description_.
            Defaults to None.

            name (str, optional): _description_. Defaults to "".

            requires_grad (bool, optional): _description_. Defaults to False.
        """
        if transform_toParent_fromSelf is None:
            transform_toParent_fromSelf = \
                groups.SE3.create_identity(requires_grad=requires_grad)

        self.transform_toParent_fromSelf = transform_toParent_fromSelf

        self.parent = parent
        if self.parent is not None:
            self.parent.children.append(self)
        self.children = []
        self.name = name
        self.requires_grad = requires_grad
        self.points = None

    def get_args(self):
        """Get list of arguments required to initialize object.

        Returns:
            list: list with parameters used in initialization, in order
            required by the self.__init__ method.
        """
        return (self.parent, self.transform_toParent_fromSelf)

    def get_kwargs(self):
        """Get list of keyword arguments used to initialize object.

        Returns:
            dict: dict with keywords and values corresponding to keywords and
            values of optional arguments for object initialization.
        """
        return {"name": self.name, "requires_grad": self.requires_grad}

    @classmethod
    def shallowcopy(cls, this):
        """Create a shallow copy of a node.

        A shallow copy will always have empty children, but the other
        attributes will the same as that of the node which is being copied.

        Args:
            this (Node): node to be copied.

        Returns:
            Node: copy of input node.
        """
        return cls(*this.get_args(), **this.get_kwargs())

    @classmethod
    def branchcopy(cls, this):
        """Create a copy of a node and its children and attach it to the tree.

        Regular Python copying does not play nicely with the computational
        graph of PyTorch. For example, we cannot do a deep copy of a non-leaf
        PyTorch tensor. So, if we have a class with code as in the example
        below,

        class A():
            def __init__(self, data: torch.Tensor):
                self.data = data
                self.more_data = 2 * data

        and we simply copy things, the fact that self.more_data depends on
        self.data in the computational graph will be lost. What we want is
        something like

        @classmethod def copy(cls, this):
            new = "cls.__init__(this.initialization_data)" for child in
            this.children:
                new.add_child(child.copy(child))

        Args:
            this_node (Node): node to be copied.

        Returns:
            Node: copy of this_node and its children, attached to this node's
            parent.
        """
        new = this.shallowcopy(this)
        # Primitive nodes are shallow copied without their children, but
        # higher-level nodes may gain children during construction. If the
        # children are already there, i.e., in case the node this is a
        # high-level node we don't need to do anything. Otherwise, we need to
        # add the children.
        if len(new.children) == 0:
            for child in this.children:
                new.add_child(
                    child.branchcopy(child),
                    transform_toSelf_fromChild=child.
                    transform_toParent_fromSelf
                )

        return new

    def in_place_mirror(self):
        """in_place_mirror.

        Changes self to a mirror version of itself with respect to the yz plane
        defined by the coordinate system of the node's parent.

        Args:
            other_node (Node, optional): node in which the mirroring yz plane
            is defined. Defaults to None, in which case the parent node of self
            is used.
        """
        # If the name has "left", change it to "right", and vice-versa.
        if self.name.find("left") >= 0:
            self.name = self.name.replace("left", "right")
        elif self.name.find("Left") >= 0:
            self.name = self.name.replace("Left", "Right")
        elif self.name.find("right") >= 0:
            self.name = self.name.replace("right", "left")
        elif self.name.find("Right") >= 0:
            self.name = self.name.replace("Right", "Left")

        self.name = self.name + " mirrored"

        # Recursively mirror children.
        for child in self.children:
            child.in_place_mirror()

        mirror_inYZPlane = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))

        T_toParent_fromSelf = \
            self.get_transform_toParent_fromSelf().transform_matrix

        new_T_toParent_fromSelf = \
            mirror_inYZPlane @ \
            T_toParent_fromSelf @ \
            mirror_inYZPlane  # This is the tricky bit, to keep handedness.

        # Create transformation update.
        tmp = self.transform_toParent_fromSelf
        T_toSelf_fromParent = tmp.inverse_transform_matrix

        delta_T_toParent_fromParent = \
            new_T_toParent_fromSelf @ T_toSelf_fromParent

        self.update_transform_toParent_fromSelf(
            groups.SE3(delta_T_toParent_fromParent)
        )

    @classmethod
    def mirror(cls, this_node):
        """mirror.

        Create a node that is a mirrored version of the original node with
        respect to the yz plane in the coordinate system of the node's parent.
        Note that, contrary to physical mirroring transformation, we preserve
        handedness by inverting the x axis of the resulting node. This is meant
        to model lateral (left-right) symmetry while keeping the coordinate
        systems sane (SE3 *must* be right-handed).

        Args:
            this_node (Node): input node to be mirrored.
        """
        new_node = this_node.branchcopy(this_node)
        new_node.in_place_mirror()

        return new_node

    def traverse(self, mode="depth") -> list:
        """traverse.

        Traverse the tree from self down to all of its descendants. If mode ==
        "depth", do traverse depth first; otherwise, traverse breadth first.

        Returns:
            list: list of descendant nodes.
        """
        descendants = [self, ]

        if mode == "depth":
            for child in self.children:
                descendants = descendants + child.traverse()
        elif mode == "breadth":
            stack = deque([self, ])
            while len(stack) > 0:
                current = stack.popleft()
                for child in current.children:
                    descendants = descendants + [child, ]
                    stack.append(child)
        else:
            print("Traversal mode unknown. Using depth first.")
            descendants = self.traverse(mode="depth")

        return descendants

    def sample_points(self):
        """sample_points.

        Sample points of node and of node's children and so on.

        Returns:
            torch.Tensor: (3,...) points in 3D, in node's coordinate system.
        """
        self_points_inSelf = self.points if self.points is not None else []
        child_points_inParent = \
            [
                child.transform_toParent_fromSelf.transform(
                    child.sample_points()
                )
                for child in self.children
            ]

        all_points_inSelf = self_points_inSelf + child_points_inParent

        return torch.cat(all_points_inSelf, dim=1)

    def get_root(self):
        """get_root.

        Traverse the graph to find root node, i.e., a node without a parent.

        Returns:
            node: current node's most immediate ancestor without a parent.
        """
        current = self
        prev = current.parent
        while prev is not None:
            current = prev
            prev = current.parent

        return current

    def update_transform_toParent_fromSelf(
        self,
        delta_transform_toParent_fromParent
    ):
        """Apply a delta transform to the transform to parent from self.

        This means that the original transform_toParent_fromSelf will become
        delta_transform_toParent_fromParent * transform_toParent_fromSelf.


        Args:
            delta_transform_toParent_fromParent (groups.SE3): perturbation to
            be applied to current toParent_fromSelf transform.
        """
        self.transform_toParent_fromSelf = \
            groups.SE3.compose_transforms(
                delta_transform_toParent_fromParent,
                self.transform_toParent_fromSelf)

    def translate_inParent(self, translation_toParent_fromParent):
        """Translate node to new position in coordinate system of its parent.

        Args:
            translation_toParent_fromParent (tensor.Torch): (3,) tensor
            corresponding to a translation in the parent's coordinate system.
        """
        transform_toParent_fromParent = \
            groups.SE3(
                torch.hstack(
                    (
                        torch.eye(3),
                        translation_toParent_fromParent.view(3, 1)
                    )
                )
            )
        self.update_transform_toParent_fromSelf(transform_toParent_fromParent)

    def rotate_inParent(self, rotation_toParent_fromParent):
        """rotate_inParent.

        Rotate node to new position in the coordinate system of its parent.

        Args:
            rotation_toParent_fromParent (groups.SO3 or torch.Tensor): element
            SO3 or torch.Tensor representing a (3, 3) rotation matrix.
        """
        if torch.is_tensor(rotation_toParent_fromParent):
            transform_toParent_fromParent = \
                groups.SO3(rotation_toParent_fromParent)
        else:
            transform_toParent_fromParent = rotation_toParent_fromParent
        self.update_transform_toParent_fromSelf(transform_toParent_fromParent)

    def list_ancestry_fromSelf_toRoot(self):
        ancestry_list = [self, ]
        if self.parent is None:
            return ancestry_list
        else:
            return \
                ancestry_list + self.parent.list_ancestry_fromSelf_toRoot()

    def is_ancestor_of(self, other) -> bool:
        """is_ancestor_of.

        Check if self is ancestor of other. Note that in our definition a node
        is an ancestor of itself.

        Args:
            other (Node): node whose ancestry is to be checked.

        Returns:
            bool: True if self is an ancestor of other, false otherwise.
        """
        if self == other:
            return True

        testing = other
        while testing not in self.children:
            if testing.parent is None:
                return False
            else:
                testing = testing.parent

        return True

    def find_closest_common_ancestor(self, other):
        """find_closest_common_ancestor.

        Find the closest common ancestor between self and other.

        If other (resp., self) is a direct ancestor of self (resp., other),
        return other (resp., self).

        Args:
            other (Node): Node whose common ancestry with self is sought.
        """
        self_ancestry_list = self.list_ancestry_fromSelf_toRoot()
        other_ancestry_list = other.list_ancestry_fromSelf_toRoot()

        # We could have disjoint trees, with no common ancestor...
        if self_ancestry_list[-1] != other_ancestry_list[-1]:
            return None

        # The trees are not disjoint, the root is a common ancestor.
        common_ancestor = self_ancestry_list[-1]

        self_length = len(self_ancestry_list)
        other_length = len(other_ancestry_list)
        min_length = min(self_length, other_length)
        for i in range(2, min_length + 1):
            # Go backwards along the ancestry lists.
            if self_ancestry_list[-i] == other_ancestry_list[-i]:
                common_ancestor = self_ancestry_list[-i]

        return common_ancestor

    def get_transform_toOther_fromSelf(self, other) -> groups.SE3:
        """get_transform_toOther_fromSelf.

        Computes the transformation from the coordinate system of self to the
        coordinate system of other.

        Args:
            other (Node): Node to which we wish to compute the transform from
            self.

        Returns:
            groups.SE3: SE(3) transformation from coordinate system of self to
            coordinate system of other.
        """
        # Rather than go all the way back to the root, we go only up to the
        # nearest common ancestor.
        if other is None:
            other = self.get_root()

        common = self.find_closest_common_ancestor(other)

        # Go up to common from self.
        transform = groups.SE3.create_identity()
        current = self
        while current != common and current is not None:
            transform_toCurrentParent_fromCurrent = \
                current.transform_toParent_fromSelf
            transform = \
                groups.SE3.compose_transforms(
                    transform_toCurrentParent_fromCurrent, transform
                )

            current = current.parent

        transform_toCommon_fromSelf = transform

        # Go up to common from other.
        transform = groups.SE3.create_identity()
        current = other
        while current != common and current is not None:
            transform_toCurrentParent_fromCurrent = \
                current.transform_toParent_fromSelf
            transform = \
                groups.SE3.compose_transforms(
                    transform_toCurrentParent_fromCurrent, transform
                )

            current = current.parent

        transform_toCommon_fromOther = transform

        return \
            groups.SE3.compose_transforms(
                transform_toCommon_fromOther.create_inverse(),
                transform_toCommon_fromSelf
            )

    def get_transform_toParent_fromSelf(self):
        """get_transform_toParent_fromSelf.

        Get the SE3 transformation that maps from the coordinate system of the
        node to that of its parent.
        """
        return self.get_transform_toOther_fromSelf(self.parent)

    def get_transform_toWorld_fromSelf(self):
        """get_transform_toWorld_fromSelf.

        Creates the direct transform to world from node, bypassing the
        intermediary nodes of the pose graph.
        """
        return self.get_transform_toOther_fromSelf(self.get_root())

    def remove_child(self, child):
        if child in self.children:
            self.children.remove(child)
            child.parent = None
            child.transform_toParent_fromSelf = groups.SE3.create_identity()

    def add_child(self, child, transform_toSelf_fromChild=None):
        """add_child.

        Add child to pose graph as a node of self.

        If transform_toSelf_fromChild is None, and the child is already in the
        pose graph, we detach the child from its current parent and reattach it
        to self, maintaining its original spatial relationship with the tree.

        Note that if child is already in the pose graph, it could in principle
        be an ancestor of self. In this case, adding child as a child of self
        would break the tree structure, and so the add_child method does
        nothing.

        If transform_toSelf_fromChild is None but the child is not in the pose
        graph, we assume that the transform_toParent_fromSelf from the child's
        point of view is identity. (This transformation could also be referred
        to as transform_toSelf_fromChild, from the point of view of self.)

        If transform_toSelf_fromChild is present, we use it as the new spatial
        relation between child and self after attachment.

        Args:
            child (Node): Node that will become a child of self.

            transform_toSelf_fromChild (groups.SE3, optional): Prescribed
            transformation in SE(3) between child and self. Defaults to None,
            in which case the transformation is inferred.
        """
        # If child is ancestor of self, do nothing.
        if child.is_ancestor_of(self):
            return

        # Check whether child is already in the pose graph of self.
        common = self.find_closest_common_ancestor(child)
        if transform_toSelf_fromChild is None:
            if common is not None:
                # There is a common ancestor, and the transform between child
                # and self is not provided. We must therefore infer it.
                transform_toCommon_fromChild = \
                    child.get_transform_toOther_fromSelf(common)
                transform_toCommon_fromSelf = \
                    self.get_transform_toOther_fromSelf(common)
                transform_toSelf_fromChild = \
                    groups.SE3.compose_transforms(
                        transform_toCommon_fromSelf.create_inverse(),
                        transform_toCommon_fromChild
                    )
            else:
                # No common ancestor, and the transform between child and self
                # is not provided. We assume it is identity.
                transform_toSelf_fromChild = groups.SE3.create_identity()

        # Remove child from old parent, any.
        if child.parent is not None:
            child.parent.remove_child(child)

        # Reset the child to parent transform.
        child.transform_toParent_fromSelf = transform_toSelf_fromChild

        # Establish that the child is child of self and self is parent of
        # child.
        self.children.append(child)
        child.parent = self
