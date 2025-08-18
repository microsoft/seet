"""point.py

Point as a type of node in a pose graph
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"

import seet.core as core
import torch


class Point(core.Node):
    """Point.

    Represents a point in a coordinate system.

    Args:
        Node: Parent class of Point.
    """

    def __init__(
        self,
        parent,
        transform_toParent_fromSelf,
        name="",
        requires_grad=False
    ):
        """__init__.

        In its own coordinate system, the point is always (0, 0, 0).

        Args:
            parent (Node): parent node of the Point object in the pose graph.

            transform_toParent_fromSelf (SE3): SE3 object representing
            transformation from coordinate system of point to the coordinate
            system of the point's parent.

            name (string, optional): name of point object. Defaults to "".
        """

        self.coordinates = torch.zeros(3)

        super().__init__(
            parent,
            transform_toParent_fromSelf,
            name=name,
            requires_grad=requires_grad
        )

    @classmethod
    def create_from_coordinates_inParent(
        cls, parent, coordinates_inParent, name=""
    ):
        """
        Create a Point object from its coordinates in its parent coordinate
        system.

        Args:
            parent (Node): parent coordinate system.

            coordinates_inParent (torch.Tensor): coordinates of point in parent
            coordinate system, represented as a (3,) or (3, 1) torch.Tensor.

            name (string, optional): name of Point object. Defaults to "".

        Returns:
            Point: newly created Point object.
        """
        transform_matrix_toParent_fromSelf = \
            torch.hstack(
                (
                    torch.eye(4, 3),
                    core.homogenize(coordinates_inParent).view(4, 1)
                )
            )
        transform_toParent_fromSelf = \
            core.SE3(transform_matrix_toParent_fromSelf)

        return cls(parent, transform_toParent_fromSelf, name=name)

    def get_coordinates_inOther(self, other):
        """get_coordinates_inOther.

        Get the coordinates of self in the coordinate system of other.

        Args:
            other (Node): node in which to obtain the coordinates of point.
        """
        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)

        return transform_toOther_fromSelf.transform(self.coordinates)

    def get_coordinates_inParent(self):
        """get_coordinates_inParent.

        Get the coordinates of self in the coordinate system of its parent.
        """
        return self.get_coordinates_inOther(self.parent)
