"""limbus_model.py

Limbus model class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com) and Chris Aholt"


import json
import seet.core as core
import seet.primitives as primitives
from seet.user import user_configs
import os
import torch


class LimbusModel(primitives.Circle):
    """LimbusModel.

    Class representing a limbus in an EyeModel object. In the pose graph,
    a LimbusModel object is a child of the an EyeModel object, and it is a
    subclass of Circle.

    The origin of the limbus coordinate system is the center of the circular
    limbus, but the origin of its parent node, i.e., the eye, is the eye
    rotation center.
    """

    def __init__(
        self,
        eye_node,
        transform_toEye_fromSelf,
        name="",
        parameter_file_name=os.path.join(
            user_configs.USER_DIR, r"default_user/default_left_eye.json"
        ),
        requires_grad=None
    ):
        """
        Initialize LimbusModel object.

        Args:
            eye_node (Point): node object that is a parent node of the
            LimbusModel object. Should be the rotation center of the eye.

            transform_toEye_fromSelf (SE3): transformation from coordinate
            system of LimbusModel to coordinate system of EyeModel parent node.

            name (str, optional): name of object.

            parameter_file_name (str, alt): json file containing parameters of
            limbus model. Defaults to "default_user/default_left_eye.json".

            requires_grad (bool, optional): if True, radius of limbus is a
            differentiable parameter. Defaults to True.
        """

        self.parameter_file_name = parameter_file_name

        with core.Node.open(self.parameter_file_name) as parameter_file_stream:
            eye_parameters = json.load(parameter_file_stream)

        # Limbus parameters.
        limbus_parameters = eye_parameters["limbus"]

        if requires_grad is None:
            if "requires grad" in limbus_parameters.keys() and \
                    limbus_parameters["requires grad"]:
                requires_grad = True
            else:
                requires_grad = False

        # Name the eye model.
        if "name" in limbus_parameters.keys():
            name = limbus_parameters["name"]
        else:
            name = ""

        radius = \
            torch.tensor(
                limbus_parameters["radius"], requires_grad=requires_grad
            )

        super().__init__(
            eye_node,
            transform_toEye_fromSelf,
            radius,
            name=name,
            requires_grad=requires_grad
        )

    def get_args(self):
        """Overwrite base-class method.
        """

        return (self.parent, self.transform_toParent_fromSelf)

    def get_kwargs(self):
        """Augment base-class method.
        """

        base_kwargs = super().get_kwargs()
        this_kwargs = {"parameter_file_name": self.parameter_file_name}

        return {**base_kwargs, **this_kwargs}
