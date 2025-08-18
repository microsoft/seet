"""user_model_tests.py

Unit test for user models. A user-model consists of Point objects
representing rotation centers of eyes and to which EyeModel objects are
attached.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.user as user
from tests.user_tests import user_tests_configs
import os
import torch
import unittest


class TestUserModel(unittest.TestCase):
    """TestUserModel.

    Unit tests for user model.
    """

    def setUp(self):
        """setUp.

        Create basic data for tests.
        """

        super().setUp()

        self.user = \
            user.UserModel(
                core.Node(),
                core.SE3.create_identity(),
                parameter_file_name=os.path.join(
                    user_tests_configs.USER_MODELS_TEST_DIR,
                    r"user_tests_data/test_user.json"
                )
            )

        self.another_user = \
            user.UserModel(
                core.Node(),
                core.SE3.create_identity(),
                requires_grad=True
            )

    def test_init(self):
        """test_init.

        Test initialization of user parameters.
        """

        num = 2
        for a_user in (self.user, self.another_user):
            # Users have two eyes.
            self.assertTrue(num == len(a_user.eyes))

            # The default setting has symmetric positions for the cornea apexes of the eyes.
            apex_inOther = a_user.eyes[0].get_cornea_apex_inOther(a_user)
            apex_inOther_ = a_user.eyes[1].get_cornea_apex_inOther(a_user)
            apex_inOther_[0] *= -1
            self.assertTrue(torch.allclose(apex_inOther, apex_inOther_))

    def test_get_vergence_point_inSelf(self):
        """Test computation of vergence point.
        """

        # True vergence point about one meter away.
        true_vergence_point_inUser = torch.tensor((1.0, 2.0, 100.0))
        self.user.verge_at_point_inSelf(true_vergence_point_inUser)
        estimated_vergence_point_inUser = self.user.get_vergence_point_inSelf()

        self.assertTrue(
            torch.allclose(
                true_vergence_point_inUser,
                estimated_vergence_point_inUser,
                rtol=core.EPS * 10000,
                atol=core.EPS * 10000
            )
        )

    def test_get_vertical_disparity_mrad(self):
        """Test computation of vertical disparity.
        """

        # Force vergence. Vertical disparity should be zero.
        true_vergence_point_inUser = torch.tensor((1.0, 2.0, 100.0))
        self.user.verge_at_point_inSelf(true_vergence_point_inUser)

        disparity_angle_mrad = self.user.get_disparity_mrad()

        # We have to be forgiving as forcing vergence and computing disparity
        # is sensitive to round-off errors.
        self.assertTrue(
            torch.allclose(
                disparity_angle_mrad,
                core.T0,
                rtol=core.EPS * 100000,
                atol=core.EPS * 100000
            )
        )


if __name__ == "__main__":
    unittest.main()
