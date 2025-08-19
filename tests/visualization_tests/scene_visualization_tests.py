"""scene_visualization_tests.py

Unit tests for scene visualization
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import matplotlib.pyplot as plt
import seet.scene as scene
from tests.device_tests import device_tests_configs
from tests.user_tests import user_tests_configs
import seet.visualization as visualization
import os
import unittest


class TestSceneModelVisualization(unittest.TestCase):
    """TestSceneModelVisualization.

    Test visualization of SEET scene.
    """

    def setUp(self) -> None:
        super().setUp()

        # Create a scene.
        eye_blobs = []
        for eye in ("left", "right"):
            file_name = \
                r"user_tests_data/test_{:s}_eye_calibration.json".format(eye)
            eye_blob = \
                os.path.join(
                    user_tests_configs.USER_MODELS_TEST_DIR, file_name
                )
            eye_blobs = eye_blobs + [eye_blob, ]

        device_blob = \
            os.path.join(
                device_tests_configs.DEVICE_MODELS_TEST_DIR,
                r"device_tests_data/test_device_calibration.json"
            )

        self.scene = scene.SceneModel.create_scene_from_real_data(
            eye_blobs[0], eye_blobs[1], device_blob
        )

    def tearDown(self):
        """tearDown.

        Instantaneously show figures.
        """

        # We can visualize the results if we make block=True, but this will
        # make the test to require interaction.
        plt.show(block=False)
        plt.close()
        self.assertTrue(True)

        super().tearDown()

    def test_scene_model_visualization(self):
        """test_scene_model_visualization.

        Test 3D plotting of scene.
        """

        # Create axes.
        fig = plt.figure()
        axes_3D = fig.add_subplot(projection="3d", proj_type="ortho")
        # Create visualization
        visualizer_3D = \
            visualization.scene_visualization.SceneModelVisualization(
                self.scene, axes_3D, visualization_node=self.scene
            )

        visualizer_3D.visualize()

    def test_image_visualization(self):
        """test_image_visualization.

        Test plotting of 2D camera images.
        """

        _, axs_2D = plt.subplots(nrows=1, ncols=2, figsize=[12, 6])
        visualizer_2D_left = \
            visualization.scene_visualization.ImageVisualization(
                self.scene, axs_2D[1], subsystem_index=0  # type: ignore
            )
        visualizer_2D_right = \
            visualization.scene_visualization.ImageVisualization(
                self.scene, axs_2D[0], subsystem_index=1  # type: ignore
            )

        visualizer_2D_right.visualize()
        visualizer_2D_left.visualize()


if __name__ == "__main__":
    unittest.main()
