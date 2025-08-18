"""leds.py.

Class representing a set of LEDs. LEDs are a subclass of point set.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from seet.device import device_configs
import seet.primitives as primitives
import json
import os
import torch


class LEDs(primitives.PointSet):
    """LEDs.

    Class for a set of LEDs.
    """

    def __init__(
        self,
        subsystem_model,
        transform_toSubsystemModel_fromLEDs,
        name="",
        parameter_file_name=os.path.join(
            device_configs.DEVICE_DIR,
            r"default_device/default_left_leds.json"
        ),
        requires_grad=None
    ):
        """
        Initialize LEDs object.

        Args:
            subsystem_model (SubsystemModel): SubsystemModel object of which
            LEDs is a child node.

            transform_toSubsystemModel_fromLEDs (groups.SE3): element of SE(3)
            corresponding to the transformation from the coordinate system of
            the LEDs to that of the eye-tracking subsystem to which the LEDs
            attached. Typically, this will be the identity transformation.

            name (str, optional): name of object.

            parameter_file_name (str, alt): path to parameter file of LEDs.
            Defaults to "default_left_leds.json". It may be overwritten by
            values in parameters_dict.

            requires_grad (bool, optional): if true, the coordinates of the
            LEDs are assumed to be differentiable. Defaults to False.

        Returns:
            LEDs: LEDs object.
        """

        self.parameter_file_name = parameter_file_name

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            led_parameters = json.load(parameter_file_stream)

        # Name the LEDs
        if "name" in led_parameters.keys():
            name = led_parameters["name"]
        else:
            name = name

        if requires_grad is None:
            if "requires grad" in led_parameters.keys() and \
                    led_parameters["requires grad"]:
                requires_grad = True
            else:
                requires_grad = False

        coordinates_inSelf = \
            torch.tensor(
                led_parameters["coordinates"],
                requires_grad=requires_grad
            ).T

        super().__init__(
            subsystem_model,
            transform_toSubsystemModel_fromLEDs,
            coordinates_inSelf,
            name=name,
            requires_grad=requires_grad
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
