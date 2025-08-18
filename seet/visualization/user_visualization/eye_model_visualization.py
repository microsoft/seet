"""eye_model_visualization.py

Visualize an eye model in 3D.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.primitives as primitives
from seet.visualization import core_visualization
from seet.visualization import primitives_visualization
import seet.visualization.user_visualization as user_visualization
from seet.visualization.user_visualization import user_visualization_configs
import os
import torch


class EyeModelVisualization(core_visualization.NodeVisualization):
    def __init__(
        self,
        eye,
        axes_3D,
        visualization_node=None,
        style_file_name=os.path.join(
            user_visualization_configs.USER_VISUALIZATION_DIR,
            r"default_user_style/default_left_eye_model_style.json"
        )
    ):
        """
        Add a visualization of an eye-model node to existing 3D axes. The
        coordinate system of the visualization is determined by the optional
        parameter visualization_node. If visualization_node is None, the root
        node of the eye model is used instead.

        Args:
            eye (EyeModel): eye model to be visualized.

            axes_3D (Axes3D): 3D axes to which add the visualization.

            visualization_node (Node, optional): node in whose coordinate
            system the eye is to be visualized. Defaults to None, in which case
            the coordinate system of the root node of the pose graph is used.

            parameter_file (str, optional): configuration file with style
            parameters for visualization. Defaults to os.path.join(
            user_visualization_configs.USER_VISUALIZATION_DIR,
            "default_user_style/default_left_eye_model_style.json")
        """

        super().__init__(
            eye, axes_3D, visualization_node, style_file_name=style_file_name
        )

        if "num level sets" in self.style_dict.keys():
            num_level_sets = self.style_dict["num level sets"]
        else:
            num_level_sets = 7
        if "num points per ellipse" in self.style_dict.keys():
            num_points_per_ellipse = self.style_dict["num points per ellipse"]
        else:
            num_points_per_ellipse = 30
        cornea_style = self.style_dict["cornea"]
        if "cornea apex" in self.style_dict.keys():
            cornea_apex_style = self.style_dict["cornea apex"]
        else:
            cornea_apex_style = {"marker": "o", "c": "c", "alpha": 0.5}
        pupil_style = self.style_dict["pupil"]
        limbus_style = self.style_dict["limbus"]

        # Cornea visualization.
        transform_toOther_fromLimbus = \
            eye.limbus.get_transform_toOther_fromSelf(eye.cornea)
        limbus_center_inCornea = \
            transform_toOther_fromLimbus.transform(torch.zeros(3))
        self.cornea_glyph = \
            primitives_visualization.EllipsoidVisualization(
                eye.cornea,
                self.axes,
                min_level=limbus_center_inCornea.clone().detach().numpy()[-1],
                num_ellipses=num_level_sets,
                num_points_per_ellipse=num_points_per_ellipse,
                visualization_node=self.visualization_node,
                **cornea_style
            )

        # Add the apex as an embellishment.
        apex_node = primitives.Point.create_from_coordinates_inParent(
            eye, eye.cornea.get_apex_inEye()
        )
        self.cornea_apex_glyph = \
            primitives_visualization.PointVisualization(
                apex_node,
                self.axes,
                visualization_node=self.visualization_node,
                **cornea_apex_style
            )

        # Remove the apex from the pose graph.
        eye.remove_child(apex_node)

        # Pupil visualization.
        self.pupil_glyph = primitives_visualization.EllipseVisualization(
            eye.pupil,
            axes_3D,
            num_points=num_points_per_ellipse,
            visualization_node=self.visualization_node,
            **pupil_style
        )

        # Limbus visualization.
        self.limbus_glyph = primitives_visualization.EllipseVisualization(
            eye.limbus,
            axes_3D,
            num_points=num_points_per_ellipse,
            visualization_node=self.visualization_node,
            **limbus_style
        )

        # Eyelids visualization. Eyelids are not primitivess. Parameters such as
        # number of points to plot along the ellipse contour and the style of
        # the curvers are in the parameter file for the eye.
        self.eyelids_glyph = \
            user_visualization.EyelidsModelVisualization(
                eye.eyelids,
                axes_3D,
                visualization_node=self.visualization_node,
                style_file_name=style_file_name
            )

    def visualize(self):
        self.cornea_glyph.visualize()
        self.cornea_apex_glyph.visualize()
        self.eyelids_glyph.visualize()
        self.pupil_glyph.visualize()
        self.limbus_glyph.visualize()

        if "show origin" in self.style_dict.keys():
            if self.style_dict["show origin"]:
                super().visualize()

        self.adjust_3D(self.axes)
