"""scene_model_tests.py

Unit tests for SEET scenes. A scene consists of a user and a device.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.device as device
import seet.scene as scene
from tests.device_tests import device_tests_configs
from tests.scene_tests import scene_tests_configs
from tests.user_tests import user_tests_configs
import seet.user as user
import os
import unittest


class TestSceneModel(unittest.TestCase):
    """TestSceneModel.

    Unit tests for SEET scene.
    """

    def setUp(self):
        """setUp.

        Create basic data for tests.
        """

        super().setUp()

        self.scene = \
            scene.SceneModel(
                parameter_file_name=os.path.join(
                    scene_tests_configs.SCENE_MODELS_TEST_DIR,
                    r"scene_tests_data/test_scene.json"
                ),
                requires_grad=False
            )

    def test_init(self):
        """test_init.

        Test initialization of scene.
        """

        # Scene has a device.
        self.assertTrue(isinstance(self.scene.device, device.DeviceModel))

        # Scene has a user.
        self.assertTrue(isinstance(self.scene.user, user.UserModel))


if __name__ == "__main__":
    unittest.main()
