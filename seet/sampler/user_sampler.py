"""user_sampler.py

Class for sampling user parameters.
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


class UserSampler:
    """
    Class for sampling of user parameters. Users has nominal locations for
    the apex of their corneas, and nominal gaze angles.
    """

    def __init__(
        self,
        user,
        num_samples,
        parameter_file_name=os.path.join(
            sampler_configs.SAMPLER_DIR,
            r"default_sampler/default_user_sampler.json"
        )
    ):
        """
        Create a user sampler. Sampling is performed by perturbing the
        parameters of the canonical user provided as input.

        Args:
            user (UserModel): canonical user; the mode of the user
            distribution.

            num_samples (int): number of samples to be generated. It may be
            overwritten by the contents of the parameter file.

            parameter_file_name (str or dict, optional): If a str, this is the
            name of a json file with parameters for user sampling. If a
            dictionary, the keys and values are the parameters for user
            sampling. Defaults to os.path.join( sampler_configs.SAMPLER_DIR,
            r"default_sampler/default_user_sampler.json" ).
        """

        # Canonical user. It is the mode of the user distribution.
        self.user = user

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            self.sampler_parameters = json.load(parameter_file_stream)

        self.num_samples = num_samples

        seed = self.sampler_parameters.get("seed")
        if seed is not None:
            numpy.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        self._set_gaze_angle_sampling_parameters()
        self._set_eye_position_sampling_parameters()

    def _set_gaze_angle_sampling_parameters(self):
        """_set_gaze_angle_sampling_parameters.

        Set samplers for horizontal and vertical rotation of the eye's
        nominal gaze directions.
        """

        # Create sampler for eye gaze and position.
        nominal_gaze_sampling_parameters = \
            self.sampler_parameters["nominal gaze angle perturbation"]

        self.sample_nominal_gaze = \
            nominal_gaze_sampling_parameters.get("apply", False)

        if not self.sample_nominal_gaze:
            return

        vert_rotation_sampling_parameters = \
            nominal_gaze_sampling_parameters["vertical rotation"]
        dist_type = vert_rotation_sampling_parameters["distribution type"]
        self.vert_rotation_sampler = \
            sampler_utils.Sampler(
                dist_type=dist_type,
                **vert_rotation_sampling_parameters
            )

        vert_rotation_disparity_sampling_parameters = \
            nominal_gaze_sampling_parameters["vertical rotation disparity"]
        dist_type = \
            vert_rotation_disparity_sampling_parameters[
                "distribution type"
            ]
        self.vert_rotation_disparity_sampler = \
            sampler_utils.Sampler(
                dist_type=dist_type,
                **vert_rotation_disparity_sampling_parameters
            )

        horiz_rotation_sampling_parameters = \
            nominal_gaze_sampling_parameters["horizontal rotation"]
        dist_type = \
            horiz_rotation_sampling_parameters["distribution type"]
        self.horiz_rotation_sampler = \
            sampler_utils.Sampler(
                dist_type=dist_type,
                **horiz_rotation_sampling_parameters
            )

        horiz_rotation_disparity_sampling_parameters = \
            nominal_gaze_sampling_parameters["horizontal rotation disparity"]
        dist_type = \
            horiz_rotation_disparity_sampling_parameters[
                "distribution type"
            ]
        self.horiz_rotation_disparity_sampler = \
            sampler_utils.Sampler(
                dist_type=dist_type,
                **horiz_rotation_disparity_sampling_parameters
            )

    def _set_eye_position_sampling_parameters(self):
        """_set_eye_position_sampling_parameters.

        _summary_
        """
        eye_position_sampling_parameters = \
            self.sampler_parameters["extrinsics perturbation"]

        self.sample_extrinsics = \
            eye_position_sampling_parameters.get("apply", False)

        if not self.sample_extrinsics:
            return

        IPD_sampling_parameters = \
            eye_position_sampling_parameters["IPD"]
        dist_type = IPD_sampling_parameters["distribution type"]
        self.IPD_sampler = \
            sampler_utils.Sampler(
                dist_type=dist_type,
                **IPD_sampling_parameters
            )

        vertical_disparity_sampling_parameters = \
            eye_position_sampling_parameters["vertical disparity"]
        dist_type = vertical_disparity_sampling_parameters["distribution type"]
        self.vertical_disparity_sampler = \
            sampler_utils.Sampler(
                dist_type=dist_type,
                **vertical_disparity_sampling_parameters
            )

        eye_relief_disparity_sampling_parameters = \
            eye_position_sampling_parameters["eye relief disparity"]
        dist_type = \
            eye_relief_disparity_sampling_parameters["distribution type"]
        self.eye_relief_disparity_sampler = \
            sampler_utils.Sampler(
                dist_type=dist_type,
                **eye_relief_disparity_sampling_parameters
            )

    def generate_samples(self):
        for _ in range(self.num_samples):
            # Create a copy of the user, which is later thrown away. This
            # avoids having to undo whichever transformation we apply to the
            # original user.
            user = self.user.branchcopy(self.user)

            if self.sample_extrinsics:
                x_times_2 = self.IPD_sampler.generate_sample()
                y_times_2 = self.vertical_disparity_sampler.generate_sample()
                z_times_2 = self.eye_relief_disparity_sampler.generate_sample()

                translation_inEye = \
                    torch.tensor((x_times_2, y_times_2, z_times_2)) / 2

                for eye in user.eyes:
                    # We don't translate the eye directly. We translate the
                    # eye's rotation center. And we translate it in the
                    # coordinate system of the eye at nominal gaze. Therefore,
                    # the translation in z corresponds to translation along the
                    # nominal gaze direction.
                    transform_toUser_fromEye = \
                        eye.get_transform_toOther_fromSelf(self.user)
                    translation_inUser = \
                        transform_toUser_fromEye.rotation.transform(
                            translation_inEye
                        )
                    eye.rotation_center.translate_inParent(translation_inUser)
                    translation_inEye = -translation_inEye

            if self.sample_nominal_gaze:
                # New nominal gaze. Common to both eyes.
                horiz_angle = self.horiz_rotation_sampler.generate_sample()
                vert_angle = self.vert_rotation_sampler.generate_sample()
                gaze_angles = torch.tensor((horiz_angle, vert_angle))

                for eye in user.eyes:
                    eye.rotate_from_gaze_angles_inParent(
                        gaze_angles, move_eyelids=False
                    )

                    # Gaze disparities. Individual to each eye.
                    horiz_angle = \
                        self.horiz_rotation_disparity_sampler.generate_sample()
                    vert_angle = \
                        self.vert_rotation_disparity_sampler.generate_sample()
                    gaze_angles = torch.tensor((horiz_angle, vert_angle))

                    eye.rotate_from_gaze_angles_inParent(
                        gaze_angles, move_eyelids=False
                    )

            yield user

            # Avoid bloating of the pose graph by removing the additional node.
            root = user.get_root()
            del root
