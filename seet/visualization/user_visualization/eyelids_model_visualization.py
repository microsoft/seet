"""eyelids_model_visualization.py

Visualize the eyelids in 3D.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.visualization import core_visualization
from seet.visualization.user_visualization import user_visualization_configs
import os
import torch


class EyelidsModelVisualization(core_visualization.NodeVisualization):
    def __init__(
        self,
        eyelids,
        axes_3D,
        visualization_node=None,
        style_file_name=os.path.join(
            user_visualization_configs.USER_VISUALIZATION_DIR,
            r"default_user_style/default_left_eye_model_style.json"
        )
    ):
        """
        Add a visualization of eyelids to existing 3D axes. The coordinate
        system of the visualization is determined by the optional parameter
        visualization_node. If visualization_node is None, the root node of the
        eyelid object is used instead.

        Args:
            eyelids (EyelidModel): eyelids model to be visualized.

            axes_3D (Axes3D): 3D axes to which add the visualization.

            visualization_node (Node, optional): node in whose coordinate
            system the eye is to be visualized. Defaults to None, in which case
            the coordinate system of the root node of the pose graph is used.

            parameter_file_name (str, optional): configuration file with style
            parameters for visualization. Defaults to os.path.join(
            user_visualization_configs.USER_VISUALIZATION_DIR,
            "default_user_style/default_left_eye_model_style.json")
        """

        super().__init__(
            eyelids,
            axes_3D,
            visualization_node,
            style_file_name=style_file_name
        )

        # Find cornea.
        eye = eyelids.parent
        self.cornea = eye.cornea  # The eye's parent is the rotation point.

        num_points_per_ellipse = self.style_dict["num points per ellipse"]
        self.style_keys = ["upper eyelid", "lower eyelid"]
        self.eyelids_inVisualizationNode = \
            eye.generate_eyelids_inOther(
                other_node=self.visualization_node,
                num_points=num_points_per_ellipse,
                remove_points_beyond_limbus=True
            )

    def visualize(self):
        """visualize.

        Actual visualization.
        """
        for i in range(len(self.eyelids_inVisualizationNode)):
            # Check if its not empty before attempting to plot.
            if len(self.eyelids_inVisualizationNode[i]) > 0:
                self.axes.plot(
                    *torch.stack(
                        self.eyelids_inVisualizationNode[i]
                    ).T.clone().detach().numpy(),
                    **self.style_dict[self.style_keys[i]]
                )

        self.adjust_3D(self.axes)
