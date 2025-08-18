"""eye_model_tests.py

Unit tests for eye mode. EyeModel is cornea, plus pupil, plus limbus, plus
eyelids.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import json
import seet.core as core
from tests.user_tests import user_tests_configs
import seet.user as user
import os
import torch
import unittest


class TestEyeModel(unittest.TestCase):
    """TestEyeModel.

    Unit tests for eye model.
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

        self.eye = \
            user.EyeModel(
                core.Node(),
                core.SE3.create_identity(),
                parameter_file_name=parameter_file_name
            )

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            self.eye_parameters = json.load(parameter_file_stream)

    def test_get_cornea_apex(self):
        """test_get_cornea_apex.

        Test getting cornea apex in the some arbitrary coordinate system.
        """

        # Get location of cornea apex in cornea's coordinate system. That's
        # just the length of the cornea's z semi-axis. Traditionally, the
        # length of the semi-axes are denoted by a, b, and c.
        c = self.eye.cornea.shape_parameters[-1].item()
        cornea_apex_inCornea = torch.tensor([0.0, 0.0, c])

        # We say "arbitrary coordinate system" because the method
        # get_cornea_apex_inOther does not care that we are asking for the
        # coordinates in the cornea coordinate system.
        cornea_apex_inCornea_ = \
            self.eye.get_cornea_apex_inOther(self.eye.cornea)

        self.assertTrue(
            torch.allclose(cornea_apex_inCornea, cornea_apex_inCornea_)
        )

    def test_rotate_unrotate_inParent(self):
        """test_rotate_unrotate_inParent.

        Test rotating and unrotating the eye in the coordinate system of the
        eye's parent.
        """

        # Cornea apex prior to rotation.
        cornea_apex_inEyeParent = self.eye.get_cornea_apex_inOther(self.eye.parent)

        # Rotate eye. Horizontal rotation followed by vertical rotation
        angles_deg = torch.tensor([13.0, 10.0])  # Arbitrary values.
        self.eye.rotate_from_gaze_angles_inParent(angles_deg)

        # Get new cornea apex.
        new_cornea_apex_inEyeParent = \
            self.eye.get_cornea_apex_inOther(self.eye.parent)

        # Check that new cornea apex is rotated version of original. Note
        # that rotation around x is vertical, and rotation around y is
        # horizontal. We apply a horizontal rotation followed by a vertical
        # rotation, so the rightmost rotation, which is applied first is
        # horizontal, i.e., around y.
        rotation = \
            core.rotation_around_x(angles_deg[1]) @ \
            core.rotation_around_y(angles_deg[0])
        new_cornea_apex_inEyeParent_ = rotation @ cornea_apex_inEyeParent

        self.assertTrue(
            torch.allclose(
                new_cornea_apex_inEyeParent, new_cornea_apex_inEyeParent_
            )
        )

        # Unrotate eye.
        self.eye.unrotate_from_gaze_angles_inParent(angles_deg)

        # Check that rotation followed by unrotation preserve the apex.
        cornea_apex_inEyeParent_ = self.eye.get_cornea_apex_inOther(self.eye.parent)

        # Give the test some leeway due to floating-point error.
        self.assertTrue(
            torch.allclose(
                cornea_apex_inEyeParent,
                cornea_apex_inEyeParent_,
                rtol=core.EPS * 100,
                atol=core.EPS * 100)
        )

    def test_direct_at_point_inParent(self):
        """Test directing eye to a point in the eye's parent coordinate system.
        """

        # Arbitrary point in coordinate system of eye's parent
        point_inParent = torch.tensor([1.0, -2.0, 5.0])
        self.eye.direct_at_point_inParent(point_inParent)

        gaze_inParent = self.eye.get_gaze_direction_inParent()

        # In the eye's parent coordinate system, the origin is rotation center.
        normalized_point_inParent = \
            point_inParent / torch.linalg.norm(point_inParent)

        self.assertTrue(
            torch.allclose(gaze_inParent, normalized_point_inParent)
        )

    def test_point_is_beyond_limbus_inOther(self):
        """test_point_is_beyond_limbus_inOther.

        Test check of whether point is beyond the limbus or not.
        """

        # The rotation center of the eye should be beyond the limbus.
        rotation_center_inEye = torch.zeros(3)
        self.assertTrue(
            self.eye.point_is_beyond_limbus_inOther(rotation_center_inEye, self.eye)
        )

        # The cornea apex cannot be beyond the limbus. Let's make it weird
        # and get the cornea apex in the coordinate system of the eyelids.
        # The test shouldn't care.
        cornea_apex_inEyelids = self.eye.get_cornea_apex_inOther(self.eye.eyelids)
        self.assertFalse(
            self.eye.point_is_beyond_limbus_inOther(
                cornea_apex_inEyelids, self.eye.eyelids
            )
        )

    def test_generate_glint_inCornea(self):
        """test_generate_glint_inCornea.

        Test generation of glints in coordinate system of cornea.
        """

        # This is not too different from the method in the Ellipsoid
        # primitives. The only interesting thing is the application of
        # occlusions by the eyelids and the removal of points beyond the
        # limbus.

        # Try a reflection from behind the cornea.
        origin_inCornea = torch.tensor([-10.0, 0.0, -10.0])
        destination_inCornea = torch.tensor([10.0, 0.0, -10.0])

        # Without application of occlusions.
        glint_inCornea = \
            self.eye.generate_glint_inCornea(
                origin_inCornea,
                destination_inCornea,
                apply_eyelids_occlusion=False,
                remove_points_beyond_limbus=False)

        self.assertTrue(glint_inCornea is not None)

        # Apply occlusions by eyelids.
        glint_inCornea = \
            self.eye.generate_glint_inCornea(
                origin_inCornea,
                destination_inCornea,
                apply_eyelids_occlusion=True,
                remove_points_beyond_limbus=False)

        self.assertTrue(glint_inCornea is None)

        # Apply occlusions by limbus.
        glint_inCornea = \
            self.eye.generate_glint_inCornea(
                origin_inCornea,
                destination_inCornea,
                apply_eyelids_occlusion=False,
                remove_points_beyond_limbus=True)

        # Now, a reflection in front of the cornea.

        # Without the application of occlusions.
        glint_inCornea = \
            self.eye.generate_glint_inCornea(
                -origin_inCornea,
                -destination_inCornea,
                apply_eyelids_occlusion=False,
                remove_points_beyond_limbus=False)

        self.assertTrue(glint_inCornea is not None)

        # Apply occlusion by limbus and eyelids. Glint should still be
        # visible.
        glint_inCornea = \
            self.eye.generate_glint_inCornea(
                -origin_inCornea,
                -destination_inCornea,
                apply_eyelids_occlusion=True,
                remove_points_beyond_limbus=True)

        self.assertTrue(glint_inCornea is not None)

        # Shut the eyelids. Glint should no longer be visible.
        self.eye.eyelids.rotate_eyelid(22.11, upper=True)
        self.eye.eyelids.rotate_eyelid(-28.41, upper=False)

        # Apply occlusion by limbus only. Should still be visible.
        glint_inCornea = \
            self.eye.generate_glint_inCornea(
                -origin_inCornea,
                -destination_inCornea,
                apply_eyelids_occlusion=False,
                remove_points_beyond_limbus=True)

        self.assertTrue(glint_inCornea is not None)

        # Apply occlusion by eyelids only. Should no longer be visible.
        glint_inCornea = \
            self.eye.generate_glint_inCornea(
                -origin_inCornea,
                -destination_inCornea,
                apply_eyelids_occlusion=True,
                remove_points_beyond_limbus=False)

        self.assertTrue(glint_inCornea is None)

    def test_generate_pupil_or_occluding_contour_inCornea(self):
        """test_generate_pupil_or_occluding_contour_inCornea.

        Test generation of refracted pupil.
        """

        # Point to the side of cornea.
        viewpoint_inCornea = torch.tensor([10.0, 0.0, 15.0])

        pupil_points_inCornea, _ = \
            self.eye.generate_refracted_pupil_inCornea(viewpoint_inCornea)
        occluding_points_inCornea = \
            self.eye.generate_occluding_contour_inCornea(viewpoint_inCornea)

        for points in (pupil_points_inCornea, occluding_points_inCornea):
            # The points lie on the cornea.
            for single_point_inCornea in points:
                # Skip occluded points
                if single_point_inCornea is None:
                    continue

                d = \
                    self.eye.cornea.compute_algebraic_distance_inEllipsoid(
                        single_point_inCornea
                    )
                self.assertTrue(
                    torch.allclose(
                        d,
                        core.T0,
                        rtol=core.EPS * 100,
                        atol=core.EPS * 100
                    )
                )

        # Shut the eyelids. Points should no longer be visible.
        self.eye.eyelids.rotate_eyelid(22.11, upper=True)
        self.eye.eyelids.rotate_eyelid(-28.41, upper=False)

        # But we don't check, so we don't know that it is not visible!
        # But we can have more values if we want.
        points, _ = \
            self.eye.generate_refracted_pupil_inCornea(
                viewpoint_inCornea,
                num_points=100,
                apply_eyelids_occlusion=False
            )

        # Points still there.
        self.assertTrue(len(points) > 0)

        # Now we check.
        points, _ = \
            self.eye.generate_refracted_pupil_inCornea(
                viewpoint_inCornea,
                num_points=100,
                apply_eyelids_occlusion=True
            )

        # It is a list, with the default 30 values.
        points = \
            [
                single_point_inCornea for single_point_inCornea
                in points if single_point_inCornea is not None
            ]

        # All points gone.
        self.assertTrue(len(points) == 0)

    def test_generate_eyelids_inCornea(self):
        """test_generate_eyelids_inCornea.

        Test generation of eyelid points.
        """

        # Previous tests have shown that eyelids can do occlusion. Are they
        # on the cornea,though?
        eyelids_inCornea = self.eye.generate_eyelids_inCornea()

        for single_eyelid_inCornea in (eyelids_inCornea):
            for single_point_inCornea in single_eyelid_inCornea:
                d = \
                    self.eye.cornea.compute_algebraic_distance_inEllipsoid(
                        single_point_inCornea
                    )
                self.assertTrue(
                    torch.allclose(
                        d,
                        core.T0,
                        rtol=core.EPS * 100,
                        atol=core.EPS * 100
                    )
                )
