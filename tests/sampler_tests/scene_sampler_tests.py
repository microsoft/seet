"""scene_sampler_tests.py

Unit tests of sampling a scene.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet
import os
import unittest


class TestSceneSampler(unittest.TestCase):
    """TestSceneSampler.

    Unit tests for scene sampler.
    """

    def setUp(self):
        """setUp.

        Create basic data for tests.
        """

        super().setUp()

        self.scene_sampler_object = \
            seet.sampler.SceneSampler(num_samples=1)

        # Create a scene with default configuration for testing
        default_scene = seet.scene.SceneModel(
            parameter_file_name=os.path.join(
                seet.scene.scene_configs.SCENE_DIR,
                r"default_scene/default_scene.json"
            )
        )
        self.another_scene_sampler_object = \
            seet.sampler.SceneSampler(
                default_scene,
                num_samples=1,
                parameter_file_name=os.path.join(
                    seet.sampler.sampler_configs.SAMPLER_DIR,
                    r"default_sampler/default_scene_sampler.json"
                )
            )

    def test_generate_data_for_visibility_analysis(self):
        """test_generate_data_for_visibility_analysis.

        Test generation of data for visibility analysis.
        """

        df = self.scene_sampler_object.generate_data_for_visibility_analysis(
            gaze_grid=[2, 2]
        )
        # For now, this is a silly test.
        self.assertTrue(len(df) == 8)  # 1 sample x (2x2) gaze grid x 2 eyes.

        df = \
            self.another_scene_sampler_object.\
            generate_data_for_visibility_analysis(
                gaze_grid=[1, 1]
            )
        # Another silly test.
        self.assertTrue(len(df) == 2)  # 1 sampler x (1x1) gaze grid x 2 eyes.
