"""circle.py

User-defined package defining a circle primitives.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from seet.primitives import ellipse


class Circle(ellipse.Ellipse):
    """Circle.

    Class defining a circle object. A circle is an ellipse for which both
    semi-axes have the same value, corresponding to the radius of the circle.

    Args:
        Ellipse (Ellipse): Ellipse class from which Circle is derived.
    """

    def __init__(
        self,
        parent,
        transform_toParent_fromSelf,
        radius,
        name="",
        requires_grad=False
    ):
        """
        Construct a Circle object.

        Args:
            radius (torch.Tensor): x_radius and y_radius values in parent class
            Ellipse.

            parent (Node): parent node in pose graph.

            transform_toParent_fromSelf (SE3): SE3 object representing
            transformation from coordinate system of Circle object to
            coordinate system of its parent node.

            name (string, optional): Name of object. Defaults to "".
        """
        super().__init__(
            parent,
            transform_toParent_fromSelf,
            radius,
            radius,
            name=name,
            requires_grad=requires_grad
        )

        self.radius = radius

    def get_args(self):
        """Augment base-class method.

        Returns:
            list: list of required arguments.
        """

        return super().get_args() + (self.radius, )

    @classmethod
    def create_from_origin_and_radius_inPlane(
        cls,
        plane_node,
        center_in2DPlane,
        radius,
        name=""
    ):
        """
        Create a circle from its origin and radius in the coordinate system
        of the plane node.

        Args:
            plane_node (Plane): parent node of ellipse. Establishes the 2D
            coordinate system in which other input parameters are represented.

            center_in2DPlane (torch.Tensor): (2,) torch Tensor representing the
            coordinates of the circle center in the plane's internal 2D
            coordinate system. In the plane's own 3D coordinate system this
            point is given by plane_node.orthonormal * origin_in2DPlane.

            radius (float or torch.Tensor): radius of circle.

            name (str, optional): name of circle node. Defaults to "".

        Returns:
            Circle: Circle object as a child of the input plane.
        """

        base = \
            ellipse.Ellipse.create_from_origin_angle_and_axes_inPlane(
                plane_node,
                center_in2DPlane,
                core.T0,
                radius,
                radius,
                name=name
            )

        # Promote to circle and set remaining attribute, i.e., radius.
        base.__class__ = Circle
        base.radius = radius  # type: ignore

        return base

    def get_radius(self):
        """get_radius.

        Getter method for the radius of the circle.

        Returns:
            float or torch.Tensor: radius of circle object.
        """
        return self.x_radius

    def set_radius(self, radius):
        """set_radius.

        Set radius of circle to new value.

        Args:
            radius (float or torch.Tensor): new value of radius.
        """
        super().set_radii(radius, radius)

        # Superclass does not know about self.radius, so we update it here.
        self.radius = self.get_radius()

    def update_radius(self, update_radius, update_mode="additive"):
        """update_radius.

        Apply an additive or multiplicative update to the radius of the
        circle.

        An additive update means that the new value of the radius will be
        self.radius += radius_update.

        A multiplicative update means that the new values of the radius will be
        self.radius *= exp(update_radius).

        Args:
            radius (float or torch.Tensor): Additive or multiplicative update
            to be applied to the radius of the circle.
        """
        super().update_radii(
            update_radius, update_radius, update_mode=update_mode
        )

        # Superclass does not know about self.radius, so we update it here.
        self.radius = self.get_radius()
