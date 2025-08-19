"""cornea_model.py

Cornea model class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com) and Chris Aholt"


import json
import seet.core as core
import seet.primitives as primitives
from seet.user import user_configs
import os
import torch


class CorneaModel(primitives.Ellipsoid):
    """CorneaModel.

    Object representing a cornea in 3D. A cornea is an ellipsoid whose
    parent is an Eye node.

    The origin of the cornea coordinate system is the center of the ellipsoid,
    but the origin of the eye is at the eye rotation center.
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
        Initialize CorneaModel object.

        Args:
            eye_node (EyeModel): node of which CorneaModel object is a child.
            Should be they eye's rotation center.

            transform_toEye_fromSelf (SE3): transformation from coordinate
            system of CorneaModel object to that of parent EyeModel.

            name (str, optional): name of object.

            parameter_file_name (str, alt): json file containing parameters of
            cornea model. Defaults to "default_user/default_left_eye.json".

            requires_grad (bool, optional): if True, parameters of model are
            differentiable. Defaults to False.

        Returns:
            CorneaModel: CorneaModel object as a child node of EyeModel and a
            subclass of Ellipsoid.
        """

        self.parameter_file_name = parameter_file_name

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            data = json.load(parameter_file_stream)

        # Cornea parameters.
        cornea_parameters = data["cornea"]
        if requires_grad is None:
            if "requires grad" in cornea_parameters.keys() and \
                    cornea_parameters["requires grad"]:
                requires_grad = True
            else:
                requires_grad = False

        radius = \
            torch.tensor(
                cornea_parameters["radius of curvature"],
                requires_grad=requires_grad
            )
        radii_factors = \
            torch.tensor(
                cornea_parameters["radii factors"],
                requires_grad=requires_grad
            )
        shape_parameters = radius * radii_factors

        eta = cornea_parameters.get("refractive index", core.TETA.item())
        self.refractive_index = torch.tensor(eta, requires_grad=requires_grad)

        # Name the eye model.
        if "name" in cornea_parameters.keys():
            name = cornea_parameters["name"]
        else:
            name = ""

        # Superclass is Ellipsoid.
        super().__init__(
            eye_node,
            transform_toEye_fromSelf,
            shape_parameters,
            name=name,
            requires_grad=requires_grad
        )

    def get_args(self):
        """Overwrite base-class method.

        Returns:
            list: list of required arguments.
        """

        return (self.parent, self.transform_toParent_fromSelf)

    def get_kwargs(self):
        """Augment base-class method.

        Returns:
            dict: dictionary of keyword arguments.
        """

        base_kwargs = super().get_kwargs()
        this_kwargs = {"parameter_file_name": self.parameter_file_name}

        return {**base_kwargs, **this_kwargs}

    def get_apex_inEye(self):
        """get_apex_inEye.

        Get the coordinates of the cornea apex in the coordinate system of
        the parent EyeModel.
        """
        apex_inSelf = core.translation_in_z(self.shape_parameters[-1])

        # Parent is eye's rotation center.
        return self.transform_toParent_fromSelf.transform(apex_inSelf)
