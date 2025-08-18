"""eyelids_model_tests.py

Unit tests for eyelids model. EyelidsModel is a pair of planes centered
at a node. Planes rotate around this node to indicate eyelids opening and
closing.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import json
import seet.core as core
import seet.user as user
from tests.user_tests import user_tests_configs
import os
import torch
import unittest


class TestEyelidsModel(unittest.TestCase):
    """TestEyelidsModel.

    Unit tests for eyelids model.
    """

    def setUp(self):
        """setUp.

        Create basic data for tests.
        """

        super().setUp()

        parameter_file_name = \
            os.path.join(
                user_tests_configs.USER_MODELS_TEST_DIR,
                r"user_tests_data/test_left_eye.json"
            )

        self.eyelids = \
            user.EyelidsModel(
                core.Node(),
                core.SE3.create_identity(),
                parameter_file_name=parameter_file_name
            )

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            eye_parameters = json.load(parameter_file_stream)

        eyelids_parameters = eye_parameters["eyelids"]

        upper_parameters = eyelids_parameters["upper"]
        self.upper_vertical_angle_deg = \
            torch.tensor(upper_parameters["vertical angle"])

        lower_parameters = eyelids_parameters["lower"]
        self.lower_vertical_angle_deg = \
            torch.tensor(lower_parameters["vertical angle"])

        in_between_angle_deg = \
            self.upper_vertical_angle_deg + self.lower_vertical_angle_deg
        in_between_angle_rad = core.deg_to_rad(in_between_angle_deg)

        # The parent of the upper and eyelid planes in the pose graph is the
        # eyelid. "Parent" below means "Eyelid".
        self.in_between_point_inParent = \
            torch.hstack(
                (
                    core.T0,
                    torch.sin(in_between_angle_rad),
                    torch.cos(in_between_angle_rad)
                )
            )

    def test_is_between_inEyelids(self):
        """test_is_between_inEyelids.

        Test verification of occlusion due to eyelids.
        """

        # in_between_point_inParent was designed to be between the eyelids.
        self.assertTrue(
            self.eyelids.is_between_inEyelids(self.in_between_point_inParent)
        )

        # Rotate the point up so it no longer is in between the eyelids.
        too_high_inParent = \
            core.rotation_around_x(-35) @ self.in_between_point_inParent
        self.assertFalse(self.eyelids.is_between_inEyelids(too_high_inParent))

        # Rotate down
        too_low_inParent = \
            core.rotation_around_x(35) @ self.in_between_point_inParent
        self.assertFalse(self.eyelids.is_between_inEyelids(too_low_inParent))

        # Shut the eyelids.
        # Original upper vertical angle is negative. We want to shut the eye,
        # so we need a positive (downwards) rotation.
        self.eyelids.rotate_eyelid(-self.upper_vertical_angle_deg, upper=True)
        # Original lower vertical angle is positive. We want to shut the eye,
        # so we need a negative (upwards) rotation.
        self.eyelids.rotate_eyelid(self.lower_vertical_angle_deg, upper=False)

        # Point was in between before, but not anymore.
        self.assertFalse(
            self.eyelids.is_between_inEyelids(self.in_between_point_inParent)
        )
