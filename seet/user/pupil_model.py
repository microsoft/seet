"""pupil_model.py

Pupil class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com) and Chris Aholt"


import json
import seet.core as core
import seet.primitives as primitives
from seet.user import user_configs
import os
import torch


class PupilModel(primitives.Circle):
    """PupilModel.

    Class defining a pupil in an EyeModel object. The EyeModel object is
    the pupil's parent. A pupil is a circle centered and orthogonal to the
    eye's visual axis.
    """

    def __init__(
        self,
        eye_node,
        transform_toEye_fromSelf,
        name="",
        parameter_file_name=os.path.join(
            user_configs.USER_DIR,
            r"default_user/default_left_eye.json"
        ),
        requires_grad=None
    ):
        """
        Initialize PupilModel object.

        Args:
            eye_node (EyeModel): EyeModel object/node of which the PupilModel
            object is a child.

            transform_toEye_fromSelf (SE3): transformation from coordinate
            system of PupilModel to coordinate system of parent EyeModel.

            name (str, optional): name of object.

            parameter_file_name (str, alt): json file containing parameters of
            pupil model. Defaults to "default_user/default_left_eye.json".

            requires_grad (bool, optional): if True, radius of pupil is a
            differentiable parameter. Defaults to True.

        Returns:
            PupilModel: PupilModel object as a child node of EyeModel and a
            subclass of Circle.
        """

        self.parameter_file_name = parameter_file_name

        with core.Node.open(self.parameter_file_name) as parameter_file_stream:
            eye_parameters = json.load(parameter_file_stream)

        # Pupil parameters.
        pupil_parameters = eye_parameters["pupil"]

        if requires_grad is None:
            if "requires grad" in pupil_parameters.keys() and \
                    pupil_parameters["requires grad"]:
                requires_grad = True
            else:
                requires_grad = False

        radius = \
            torch.tensor(
                pupil_parameters["radius"], requires_grad=requires_grad
            )

        super().__init__(
            eye_node,
            transform_toEye_fromSelf,
            radius,
            name=pupil_parameters["name"],
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
