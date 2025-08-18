"""direction.py

Direction as a type of node in a pose graph.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"

import seet.core as core
import torch


class Direction(core.Node):
    """Direction.

    Represents a scaled direction in a coordinate system.

    Args:
        Node: Parent class of Direction
    """

    def __init__(
        self,
        parent,
        transform_toParent_fromSelf,
        scale=core.T1.clone(),
        name="",
        requires_grad=False
    ):
        """
        In its own coordinate system, the direction is always (0, 0, 1).

        Args:
            parent (Node): parent node of the Direction object in the pose
            graph.

            transform_toParent_fromSelf (SE3): SE3 object representing
            transformation from coordinate system of the Direction object to
            the coordinate system of the point's parent.

            scale (float): scale of direction vector. Defaults to 1.0.

            name (string, optional): name of Direction object. Defaults to
            None.
        """
        super().__init__(
            parent,
            transform_toParent_fromSelf,
            name=name,
            requires_grad=requires_grad
        )

        self.components = torch.tensor((0.0, 0.0, 1.0))
        self.scale = scale

    def get_kwargs(self):
        """Augment base-class method.

        Returns:
            dict: dictionary with keyword values.
        """

        base_kwargs = super().get_kwargs()
        this_kwargs = {"scale": self.scale}

        return {**base_kwargs, **this_kwargs}

    @classmethod
    def create_from_components_inParent(
        cls, parent, components_inParent, name=""
    ):
        """
        Create a Direction object from its components in its parent
        coordinate system.

        Args:
            parent (Node): parent coordinate system.

            components_inParent (torch.Tensor): components of direction in
            parent coordinate system, represented as a (3,) or (3, 1)
            torch.Tensor.

            name (string, optional): name of Direction object. Defaults to
            None.
        """
        scale = torch.linalg.norm(components_inParent)
        transform_matrix_toParent_fromSelf = \
            core.rotation_matrix_from_u_to_v(
                torch.tensor([0.0, 0.0, 1.0]),
                components_inParent
            )

        transform_toParent_fromSelf = \
            core.SE3(transform_matrix_toParent_fromSelf)

        return cls(parent, transform_toParent_fromSelf, scale=scale, name=name)

    def get_components_inOther(self, other):
        """get_components_inOther.

        Get the components of the direction in the coordinate system of
        another node in the pose graph.

        Args:
            other (Node): node in which to obtain the components of the
            direction.
        """
        transform_toOther_fromSelf = self.get_transform_toOther_fromSelf(other)

        return transform_toOther_fromSelf.rotation.transform(self.components)
