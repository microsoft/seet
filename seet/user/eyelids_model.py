"""eyelids_model.py

Eyelids model.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import json
import seet.core as core
import seet.primitives as primitives
from seet.user import user_configs
import os
import torch


class EyelidsModel(core.Node):
    """EyelidsModel.

    Object representing eyelids in 3D. Eyelids consist of a rotation point
    and planes that intersect the cornea on ellipsoidal contours. The parent
    node of the eyelids is the eye and the parent not of the rotation point is
    the eyelids node. But the parent of the planes is the rotating point. This
    setup allows for more easily rotating the eyelids to simulate their opening
    and closing.
    """

    def __init__(
        self,
        eye_node,
        transform_toEye_fromSelf,
        name="",
        parameter_file_name=os.path.join(
            user_configs.USER_DIR, r"default_user/default_left_eye.json"
        ),
        requires_grad=False
    ):
        """
        Initialize EyelidsModel object.

        Args:
            eye_rotation_center (Node): node of which EyelidsModel object is a
            child. Should be the eye.

            transform_toEye_fromSelf (SE3): transformation from coordinate
            system of EyelidsModel object to that of its parent EyeModel.

            name (str, optional): name of object.

            parameter_file_name (str, optional): json file containing
            parameters of ellipse model. Defaults to
            os.path.join(user_configs.USER_DIR,
            "default_user/default_left_eye.json").

            requires_grad (bool, optional): if True, parameters of model are
            differentiable. Defaults to None.
        """

        self.parameter_file_name = parameter_file_name

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            eye_parameters = json.load(parameter_file_stream)

        eyelids_parameters = eye_parameters["eyelids"]

        if "name" in eyelids_parameters.keys():
            name = eyelids_parameters["name"]
        else:
            name = ""

        super().__init__(
            eye_node,
            transform_toEye_fromSelf,
            name=name,
            requires_grad=requires_grad
        )

        if requires_grad is None or requires_grad is False:
            if "requires grad" in eyelids_parameters.keys() and \
                    eyelids_parameters["requires grad"]:
                requires_grad = True
            else:
                requires_grad = False

        # Upper eyelid.
        upper_parameters = eyelids_parameters["upper"]
        upper_vertical_angle_deg = \
            torch.tensor(
                upper_parameters["vertical angle"],
                requires_grad=requires_grad
            )
        self.upper_eyelid_to_eye_rotation_ratio = \
            -(
                torch.tensor(
                    upper_parameters["vertical angle at 20 deg upward gaze"],
                    requires_grad=requires_grad
                ) - upper_vertical_angle_deg
            ) / 20  # This is 20 deg.

        # We rotate like this, so that points "in front" of the plane are
        # visible:
        #
        #  ^ y           ___ upper-eyelid plane |         ___/ |     ___/_ |
        #  ___/      \  positive rotation is is up (x axis into screen) |/
        #  | +---------------> z \_ \_ _\| plane normal
        #
        upper_eyelids_plane_normal_inRotationPoint = \
            core.rotation_around_x(upper_vertical_angle_deg) @ \
            torch.tensor([0.0, -1.0, 0.0])

        if "name" in upper_parameters.keys():
            name = upper_parameters["name"]
        else:
            name = ""

        self.upper_eyelid = \
            primitives.Plane.create_from_origin_and_normal_inParent(
                self,
                torch.zeros(3),
                upper_eyelids_plane_normal_inRotationPoint,
                name=name
            )

        # Lower eyelid
        lower_parameters = eyelids_parameters["lower"]
        lower_vertical_angle_deg = \
            torch.tensor(
                lower_parameters["vertical angle"],
                requires_grad=requires_grad
            )

        # For the lower eyelid we rotate the plane so that a normal pointing up
        # now points up and forward. Again, we want that point "in front" of
        # the plane be visible.
        lower_eyelids_plane_normal_inRotationPoint = \
            core.rotation_around_x(lower_vertical_angle_deg) @ \
            torch.tensor([0.0, 1.0, 0.0])

        if "name" in lower_parameters.keys():
            name = lower_parameters["name"]
        else:
            name = ""

        self.lower_eyelid = \
            primitives.Plane.create_from_origin_and_normal_inParent(
                self,
                torch.zeros(3),
                lower_eyelids_plane_normal_inRotationPoint,
                name=name
            )

        # We need to know what it means to be "in between" the eyelids, but it
        # is hard to define sidedness for planes that go through the origin. We
        # do a manual check.
        in_between_angle_deg = \
            upper_vertical_angle_deg + lower_vertical_angle_deg
        in_between_angle_rad = core.deg_to_rad(in_between_angle_deg)

        # The parent of the upper and eyelid planes in the pose graph is the
        # eyelid. "Parent" below means "Eyelid".
        in_between_point_inParent = \
            torch.hstack(
                (
                    core.T0,
                    torch.sin(in_between_angle_rad),
                    torch.cos(in_between_angle_rad)
                )
            )
        self.upper_in_between_sign = \
            torch.sign(
                self.upper_eyelid.compute_signed_distance_to_point_inParent(
                    in_between_point_inParent
                )
            )
        self.lower_in_between_sign = \
            torch.sign(
                self.lower_eyelid.compute_signed_distance_to_point_inParent(
                    in_between_point_inParent
                )
            )

    def get_kwargs(self):
        """Augment base-class method.

        Returns:
            dict: dictionary with keyword arguments.
        """

        base_kwargs = super().get_kwargs()
        this_kwargs = {"parameter_file_name": self.parameter_file_name}

        return {**base_kwargs, **this_kwargs}

    def rotate_eyelid(self, angle_deg, upper=True):
        """rotate_eyelid.

        Rotate plane of upper or lower eyelid. Negative means up. To shut
        the eye, rotate the upper eyelid (upper=True) by a positive angle and
        the lower eyelid (upper=False) by a negative angle.

        Args:
            angle_deg (float or torch.float): rotation angle in degrees.

            upper (bool, optional): if True, rotate the upper eyelid.
            Otherwise, rotate the lower eyelid. Defaults to True.
        """
        rotation_matrix_toParent_fromParent = core.rotation_around_x(angle_deg)
        rotation_toParent_fromParent = \
            core.SO3(rotation_matrix_toParent_fromParent)

        if upper:
            self.upper_eyelid.rotate_inParent(rotation_toParent_fromParent)
        else:
            self.lower_eyelid.rotate_inParent(rotation_toParent_fromParent)

    def is_between_inEyelids(self, point_inEyelid):
        """is_between_inEyelids.

        Determine whether input point in parent coordinate system is between
        the eyelids.

        Args:
            point_inParent (torch.Tensor): (3,) tensor with coordinates of
            point in coordinate system of eyelids parent.

        Returns:
            bool: True if point is between eyelids, False otherwise.
        """
        upper_sign = \
            torch.sign(
                # For the eyelid plane, parent means "Eyelid".
                self.upper_eyelid.compute_signed_distance_to_point_inParent(
                    point_inEyelid
                )
            )
        if upper_sign != self.upper_in_between_sign:
            return False

        lower_sign = \
            torch.sign(
                # For the eyelid plane, parent means "Eyelid".
                self.lower_eyelid.compute_signed_distance_to_point_inParent(
                    point_inEyelid
                )
            )
        if lower_sign != self.lower_in_between_sign:
            return False

        return True
