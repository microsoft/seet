"""point_set.py

Point set as a type of node in a pose graph. This is less expensive than
representing each individual point as a node.
"""


__author__ = "Paulo R. S. Mendonca(padossa@microsoft.com)"


import seet.core as core
import torch


class PointSet(core.Node):
    """PointSet.

    Represents a point set in a coordinate system.
    """

    def __init__(
        self,
        parent,
        transform_toParent_fromSelf,
        coordinates_inSelf,
        name="",
        requires_grad=False
    ):
        """
        Represents a point set in a coordinate system. A point set is
        initialized with the coordinates of points in the point set's own
        coordinate system.

        Args:
            parent (Node): parent node of point set in the pose graph.

            transform_toParent_fromSelf (SE3): SE3 object representing
            transformation from coordinate system of the point set to the
            coordinate system of the point set's parent.

            coordinates_inSelf (torch.Tensor): (3, N) tensor representing
            coordinates of N points in the point set's coordinate system.

            name (str, optional): name of point set object. Defaults to "".
        """

        self.coordinates = coordinates_inSelf
        self.num = self.coordinates.shape[1]

        super().__init__(
            parent,
            transform_toParent_fromSelf,
            name=name,
            requires_grad=requires_grad
        )

    def get_args(self):
        """Augment base-class method.

        Returns:
            _type_: _description_
        """

        return super().get_args() + (self.coordinates, )

    @classmethod
    def create_from_coordinates_inParent(
        cls, parent, coordinates_inParent, name=""
    ):
        return \
            cls(
                parent,
                core.SE3.create_identity(),
                coordinates_inParent,
                name=name
            )

    # We need to overwrite the in_place_mirror method for point sets because,
    # to keep things lightweight, the coordinates of the points are not
    # children.
    def in_place_mirror(self):
        super().in_place_mirror()

        self.coordinates = \
            torch.diag(torch.tensor([-1.0, 1.0, 1.0])) @ self.coordinates

    def get_coordinates_inParent(self):
        """get_coordinates_inParent.

        Return coordinates of points in point set.
        """
        return \
            self.transform_toParent_fromSelf.transform(self.coordinates)

    def get_coordinates_inOther(self, other):
        """get_coordinates_inOther.

        Return coordinates of points in point set in the coordinate system
        of Node other.

        Args:
            other (Node): node in which to represent the coordinates of the
            points in the point set.
        """
        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)

        return transform_toOther_fromSelf.transform(self.coordinates)
