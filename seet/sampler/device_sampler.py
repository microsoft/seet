"""device_sampler.py

Class for sampling device parameters.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import json
import seet.core as core
from seet.sampler import sampler_configs
from seet.sampler import sampler_utils
import numpy
import os
import random
import torch


class DeviceSampler:
    """
    Class for sampling user parameters. Devices have nominal locations based
    on their expected fit.
    """

    def __init__(
        self,
        device,
        num_samples,
        parameter_file_name=os.path.join(
            sampler_configs.SAMPLER_DIR,
            r"default_sampler/default_device_sampler.json"
        )
    ):
        """
        Create a device sampler. Sampling is performed by perturbing the
        parameters of the canonical device provided as input.

        Args:
            device (DeviceModel): canonical device; the mode of the device
            distribution.

            num_samples (int): number of samples to be generated.

            parameter_file_name (str or dict, optional): If a str, this is the
            name of a json file with parameters for device sampling. If a
            dictionary, the keys and values are the parameters for device
            sampling. Defaults to os.path.join( sampler_configs.SAMPLER_DIR,
            r"default_sampler/default_device_sampler.json" ).
        """

        # Canonical device. It is the mode of the device distribution.
        self.device = device

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            self.sampler_parameters = json.load(parameter_file_stream)

        self.num_samples = num_samples

        seed = self.sampler_parameters.get("seed")
        if seed is not None:
            numpy.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        # If device does not have occluders, this does nothing.
        self._set_occluder_sampling_parameters()

    def _read_sampling_parameters(self, motion):
        """_read_sampling_parameters.

        Read in sampling parameters for rotation or translation.

        Args:
            motion (str): if "rotation", read in the sampling parameters for
            the rotation of the occluders. If "translation", read in the
            parameters for the translation of the occluders.
        """

        if motion == "rotation":
            key = "occluder rotational perturbation"
        else:
            assert (motion == "translation"), "Unknown motion type."
            key = "occluder translational perturbation"

        return \
            sampler_utils.Sampler.read_SE3_parameters(
                self.sampler_parameters, key
            )

    def _set_occluder_sampling_parameters(self):
        """_set_occluder_sampling_parameters.

        Set parameters for sampling or the orientation of the occluders.
        """
        # We assume rigidity, so that subsystems are a rigid body in the scene.
        # However, occluders are typically separate bodies, such as glasses,
        # and are therefore allowed to move independently.

        if self.device.subsystems[0].occluder is not None or \
                self.device.subsystems[1].occluder is not None:
            self.occluder_rotation_samplers = \
                self._read_sampling_parameters("rotation")
            self.occluder_translation_samplers = \
                self._read_sampling_parameters("translation")
        else:
            self.occluder_rotation_samplers = []
            self.occluder_translation_samplers = []

    def generate_samples(self):
        for _ in range(self.num_samples):
            # Create a copy of the device, which is later thrown away. This
            # avoids having to undo whichever transformation we apply to the
            # original device.
            device = self.device.branchcopy(self.device)

            if len(self.occluder_rotation_samplers) > 0:
                angle_axis_deg = \
                    [
                        self.occluder_rotation_samplers[i].generate_sample()
                        for i in range(3)
                    ]

                angle_axis_rad = core.deg_to_rad(torch.tensor(angle_axis_deg))
                rotation_matrix_toDevice_fromDevice = \
                    core.rotation_matrix(angle_axis_rad)
                transform_toDevice_fromDevice = \
                    core.groups.SO3(rotation_matrix_toDevice_fromDevice)

                apply = True
            else:
                transform_toDevice_fromDevice = \
                    core.groups.SE3.create_identity()
                apply = False

            if len(self.occluder_translation_samplers) > 0:
                translation = \
                    [
                        self.occluder_translation_samplers[i].generate_sample()
                        for i in range(3)
                    ]

                transform_toDevice_fromDevice = \
                    core.groups.SE3.compose_transforms(
                        core.groups.SE3(torch.tensor(translation)),
                        transform_toDevice_fromDevice
                    )

                apply = True

            if apply:
                # We transform the occluders in each subsystem separately.
                for subsystem in device.subsystems:
                    occluder = subsystem.occluder
                    if occluder is not None:
                        # We want to update the pose of the occluder by
                        # applying update_transform_toParent_fromSelf where
                        # parent is the subsystem, and self is the occluder.
                        # Note that the input to this update is an updated
                        # transform to parent from parent, and the parent of
                        # the occluder is the subsystem. We have
                        #
                        # update_transform_toSubsystem_fromSubsystem = \
                        #   transform_toSubsystem_fromDevice *
                        #   transform_toDevice_fromDevice *
                        #   transform_toDevice_fromSubsystem

                        transform_toDevice_fromSubsystem = \
                            subsystem.get_transform_toParent_fromSelf()
                        transform_toSubsystem_fromDevice = \
                            transform_toDevice_fromSubsystem.create_inverse()

                        update_transform_toSubsystem_fromSubsystem = \
                            core.groups.SE3.compose_transforms(
                                transform_toSubsystem_fromDevice,
                                core.groups.SE3.compose_transforms(
                                    transform_toDevice_fromDevice,
                                    transform_toDevice_fromSubsystem
                                )
                            )

                        occluder.update_transform_toParent_fromSelf(
                            update_transform_toSubsystem_fromSubsystem
                        )

            yield device

            root = device.get_root()
            del root
