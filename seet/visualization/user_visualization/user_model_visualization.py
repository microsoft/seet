"""user_model_visualization.py

Visualize user in 3D.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.visualization import core_visualization, primitives_visualization
from seet.visualization import user_visualization
from seet.visualization.user_visualization import user_visualization_configs
import os


class UserModelVisualization(
    core_visualization.NodeVisualization
):
    def __init__(
        self,
        user,
        axes_3D,
        visualization_node=None,
        style_file_name=os.path.join(
            user_visualization_configs.USER_VISUALIZATION_DIR,
            r"default_user_style/default_user_model_style.json"
        )
    ):
        """
        Add a visualization of a user to existing 3D axes. The coordinate
        system of the visualization is determined by the optional parameter
        visualization_node. If visualization_node is None, the root of the node
        of the User is used instead.

        Args:
            user (UserModel): user model to be visualized.

            axes_3D (Axes3D): 3D axes to which add the visualization.

            num_level_sets (int, optional): number of ellipsoidal level sets
            used to visualize the user's eyes. Defaults to 7.

            num_points_per_ellipse (int, optional): number of points per
            elliptical level set used to describe the user's eyes. Defaults to
            30.

            visualization_node (Node, optional): node in whose coordinate
            system the eye is to be visualized. Defaults to None, in which case
            the coordinate system of the root node of the pose graph is used.

            style_file_name (str, optional): path for style-configuration file.
            Defaults to parameter_file_name=os.path.join(
            user_visualization_configs.USER_VISUALIZATION_DIR,
            "default_user_style/default_user_model_style.json")
        """

        super().__init__(
            user, axes_3D, visualization_node, style_file_name=style_file_name
        )

        eyes_style = self.style_dict["eyes"]

        self.eye_glyphs = []
        self.rotation_center_glyphs = []
        num_eye_styles = len(eyes_style)

        USER_VISUALIZATION_DIR = self.style_dict.get("path")
        if USER_VISUALIZATION_DIR is None:
            USER_VISUALIZATION_DIR = \
                user_visualization_configs.USER_VISUALIZATION_DIR
        for i in range(len(user.eyes)):
            # Cycle through styles:
            style = eyes_style[i % num_eye_styles]

            # Visualization of rotation center.
            rotation_center_style = style["rotation center"]

            self.rotation_center_glyphs = \
                self.rotation_center_glyphs + \
                [
                    primitives_visualization.PointVisualization(
                        user.rotation_centers[i],
                        self.axes,
                        visualization_node=self.visualization_node,
                        **rotation_center_style
                    )
                ]

            # Visualization of eye.
            eye_style_file = os.path.join(USER_VISUALIZATION_DIR, style["eye"])
            self.eye_glyphs = \
                self.eye_glyphs + \
                [
                    user_visualization.EyeModelVisualization(
                        user.eyes[i],
                        self.axes,
                        visualization_node=self.visualization_node,
                        style_file_name=eye_style_file
                    )
                ]

    def visualize(self):
        """visualize.

        Actual visualization.
        """

        for single_rotation_center_glyph in self.rotation_center_glyphs:
            single_rotation_center_glyph.visualize()

        for single_eye_glyph in self.eye_glyphs:
            single_eye_glyph.visualize()

        self.adjust_3D(self.axes)
