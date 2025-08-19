"""occluder.py.

Class representing an occluder, i.e., the boundary of a surface that
    occludes light either from the LEDs to the eye or from the eye to the
    cameras.

    Boundaries are always topologically equivalent to circles (no funny knots),
    but occlusion can happen in the interior or exterior of the boundary.
    """


__author__ = "Paulo R. S. Mendonca"


import json
import seet.core as core
from seet.device import device_configs
import seet.primitives as primitives
import os
import torch


class Occluder(primitives.PointSet):
    """Occluder.

    Class for an occluder object. The occluder is defined by its
    boundary, which is topologically equivalent to a circle and defined by
    a set of points.
    """

    def __init__(
        self,
        subsystem_model,
        transform_toSubsystemModel_fromOccluder,
        name="",
        parameter_file_name=os.path.join(
            device_configs.DEVICE_DIR,
            r"default_device/default_left_occluder.json"
        ),
        requires_grad=None
    ):
        """
        Initialize Occluder object.

        Args:
            subsystem_model (SubsystemModel): SubsystemModel object of
            which LEDs is a child node.

            transform_toSubsystemModel_fromOccluder (groups.SE3): element
            of SE(3) corresponding to the transformation from the
            coordinate system of the LEDs to that of the eye-tracking
            subsystem to which the LEDs attached. Typically, this will be
            the identity transformation.

            name (str, optional): name of object.

            parameter_file_name (str, alt): path to parameter file of LEDs.
            Defaults to "default_left_occluder.json". It may be overwritten
            by values in parameters_dict.

            requires_grad (bool, optional): if true, the coordinates of the
            points in the occluder boundary are assumed to be
            differentiable. Defaults to False.

        Returns:
            Occluder: occluder object.
        """

        self.parameter_file_name = parameter_file_name

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            occluder_parameters = json.load(parameter_file_stream)

        # Name the occluder.
        if "name" in occluder_parameters.keys():
            name = occluder_parameters["name"]
        else:
            name = name

        # Occluder type. Default is 'window'.
        if "type" in occluder_parameters.keys():
            self.type = occluder_parameters["type"]
        else:
            self.type = "window"

        if requires_grad is None:
            if "requires grad" in occluder_parameters.keys() and \
                    occluder_parameters["requires grad"]:
                requires_grad = True
            else:
                requires_grad = False

        coordinates_inSelf = \
            torch.tensor(
                occluder_parameters["coordinates"],
                requires_grad=requires_grad
            ).T

        super().__init__(
            subsystem_model,
            transform_toSubsystemModel_fromOccluder,
            coordinates_inSelf,
            name=name,
            requires_grad=requires_grad
        )

    def is_ray_occluded_inParent(self, origin_inParent, direction_inParent):
        """is_ray_occluded_inParent.

        Determine whether a ray defined by an origin and a direction in the
        occluder parent is occluded.

        Args:
            origin_inParent (torch.Tensor): (3,) tensor corresponding to
            origin of ray in occluder parent coordinate system.

            direction_inParent (torch.Tensor): (3,) tensor corresponding to
            direction of ray in occluder parent coordinate system.
        """

        # We start by defining a plane orthogonal to the ray. The plane will be
        # a node, so we have to remember to keep the pose graph lean by
        # removing the plane after we use it.
        plane_node = \
            primitives.Plane.create_from_origin_and_normal_inParent(
                self.parent, origin_inParent, direction_inParent
            )

        # We then project rays from the boundary of the occluder onto the plane
        # and determine if (0, 0) (the intersection of the input ray with the
        # plane in the plane's coordinate system) is interior to the polygon
        # defined by the intersection points we just computed. We use the
        # winding-number algorithm
        # (https://en.wikipedia.org/wiki/Point_in_polygon).
        transform_toPlane_fromSelf = \
            self.get_transform_toOther_fromSelf(plane_node)
        points_inPlane = transform_toPlane_fromSelf.transform(self.coordinates)
        # In the plane's coordinate system projection is simply zeroing of
        # third coordinate.
        first_point_non_normalized = points_inPlane[:2, -1]
        first_point = core.normalize(first_point_non_normalized)

        # A nasty bug happened when I did this:
        #
        # angle_sum_rad = core.T0
        #
        # It created are reference to core.T0, which was updated in the loop.
        #
        angle_sum_rad = core.T0.clone()

        for single_point_inPlane in points_inPlane.T:
            intersection_inPlane = single_point_inPlane[:2]
            second_point = core.normalize(intersection_inPlane)
            z_coordinate_of_cross_product = \
                first_point @ \
                torch.tensor([[0.0, 1.0], [-1.0, 0.0]]) @ \
                second_point
            angle_sum_rad += torch.asin(z_coordinate_of_cross_product)
            first_point = second_point

        # We are done with the plane. The check is unnecessary, but otherwise
        # pylance complains.
        if self.parent is not None:
            self.parent.remove_child(plane_node)

        # Value will be zero if point is outside polygon, or +/- 2*pi if point
        # is outside, with sign depending on whether points are clockwise or
        # counterclockwise (which we neither know or care).
        is_interior = torch.abs(angle_sum_rad) > core.TPI

        if self.type == "blocker":
            # For an occluder or type 'blocker', point is occluded if ray
            # intersects the surface of the blocker.
            return is_interior
        else:
            # For an occluder of type 'window', point is occluded if ray
            # intersects exterior of the blocker.
            return not is_interior

    def is_ray_occluded(self, ray_node):
        """is_ray_occluded.

        Check whether occluder occludes ray in an arbitrary coordinate
        system.

        Args:
            ray_node (Ray): ray node is same pose graph as occluder.
        """

        origin_inParent, destination_inParent = \
            ray_node.get_origin_and_destination_inOther(self.parent)

        return self.is_ray_occluded_inParent(
            origin_inParent, destination_inParent
        )

    def get_args(self):
        """Overwrite base-class method.

        Returns:
            tuple: tuple with required arguments.
        """

        return (self.parent, self.transform_toParent_fromSelf)

    def get_kwargs(self):
        """Augment base-class method.

        Returns:
            dict: dictionary with keyword arguments.
        """

        base_kwargs = super().get_kwargs()
        this_kwargs = {"parameter_file_name": self.parameter_file_name}

        return {**base_kwargs, **this_kwargs}
