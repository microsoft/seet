"""eye_model_visualization_tests.py

Unit tests for eye-model visualization.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import matplotlib.pyplot as plt
import seet.core as core
import seet.primitives as primitives
import seet.user as user
import seet.visualization as visualization
import torch
import unittest


class TestEyeModelVisualization(unittest.TestCase):
    """TestEyeModelVisualization.

    Test visualization of eye model.
    """

    def setUp(self) -> None:
        super().setUp()

        # Create an eye.
        self.eye = user.EyeModel(core.Node(), core.SE3.create_identity())

        # Create origin, destination, and reflection nodes.
        self.origin_inEye = torch.tensor([4.0, -4.0, 14.0])
        self.origin_node = \
            primitives.Point.create_from_coordinates_inParent(
                self.eye, self.origin_inEye
            )

        self.outside_destination_inEye = torch.tensor([1.0, 2.0, 21.0])
        self.outside_node = \
            primitives.Point.create_from_coordinates_inParent(
                self.eye, self.outside_destination_inEye
            )

        self.inside_destination_inEye = torch.tensor([0.0, -1.0, 5.0])
        self.inside_node = \
            primitives.Point.create_from_coordinates_inParent(
                self.eye, self.inside_destination_inEye
            )

        # Create axes.
        fig = plt.figure()
        self.axes = fig.add_subplot(projection="3d", proj_type="ortho")

    def tearDown(self):
        """tearDown.

        Instantaneously show figures.
        """
        plt.show(block=False)
        # We can make visualize the results if we comment out the command
        # below, but then the tests will require interaction.
        plt.close()
        self.assertTrue(True)

        super().tearDown()

    def test_eye_model_display(self):
        """test_eye_model_display.

        Test 3D plotting of eye model.
        """

        # Visualize the eye.
        visualization.user_visualization.EyeModelVisualization(
            self.eye, self.axes
        )
        visualization.primitives_visualization.PointVisualization(
            self.origin_node, self.axes
        )
        visualization.primitives_visualization.PointVisualization(
            self.outside_node, self.axes
        )
        visualization.primitives_visualization.PointVisualization(
            self.inside_node, self.axes
        )

    def test_eye_model_reflection_display(self):
        """test_eye_model_reflection_display.

        Test 3D plotting of eye model and a reflection point.
        """

        visualization.user_visualization.EyeModelVisualization(
            self.eye, self.axes
        )
        visualization.primitives_visualization.PointVisualization(
            self.origin_node, self.axes
        )
        visualization.primitives_visualization.PointVisualization(
            self.outside_node, self.axes
        )

        reflection_inCornea = \
            self.eye.cornea.compute_reflection_point_inEllipsoid(
                self.origin_node.get_coordinates_inOther(self.eye.cornea),
                self.outside_node.get_coordinates_inOther(self.eye.cornea)
            )
        reflection_node = \
            primitives.Point.create_from_coordinates_inParent(
                self.eye.cornea, reflection_inCornea
            )
        visualization.primitives_visualization.PointVisualization(
            reflection_node, self.axes
        )

    def test_eye_model_refraction_display(self):
        """test_eye_model_refraction_display.

        Test 3D plotting of eye model and a refraction point.
        """

        visualization.user_visualization.EyeModelVisualization(
            self.eye, self.axes
        )
        visualization.primitives_visualization.PointVisualization(
            self.origin_node, self.axes
        )
        visualization.primitives_visualization.PointVisualization(
            self.inside_node, self.axes
        )

        refraction_inCornea = \
            self.eye.cornea.compute_refraction_point_inEllipsoid(
                self.origin_node.get_coordinates_inOther(self.eye.cornea),
                self.inside_node.get_coordinates_inOther(self.eye.cornea)
            )
        refraction_node = \
            primitives.Point.create_from_coordinates_inParent(
                self.eye.cornea, refraction_inCornea
            )
        visualization.primitives_visualization.PointVisualization(
            refraction_node, self.axes
        )


if __name__ == "__main__":
    unittest.main()
