"""camera_visualization.py

Visualization of camera geometry in 3D.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.visualization import core_visualization
# We are creating the device_visualization module, so it is not yet ready for
# import.
from seet.visualization.device_visualization \
    import device_visualization_configs
import os
import torch


class CameraVisualization(core_visualization.NodeVisualization):
    def __init__(
        self,
        camera,
        axes_3D,
        visualization_node=None,
        style_file_name=os.path.join(
            device_visualization_configs.DEVICE_VISUALIZATION_DIR,
            r"default_device_style/default_left_camera_style.json"
        )
    ):
        """
        Add a visualization of camera to existing 3D axes.

        Args:
            camera (NormalizedCamera): camera to be visualized.

            axes_3D (Axes3D): 3D axes to which add the visualization.

            visualization_node (Node, optional): node in whose coordinate
            system the camera is to be visualized. If None, defaults to root
            node of pose graphs.

            parameter_file_name (str, optional): path to visualization style
            file. Defaults to
            os.path.join(device_visualization_configs.DEVICE_VISUALIZATION_DIR,
            "default_device_style/default_left_camera_style.json").
        """

        super().__init__(
            camera,
            axes_3D,
            visualization_node,
            style_file_name=style_file_name
        )
        self.scale_mm = self.style_dict["scale mm"]
        self.wireframe_style = self.style_dict["wireframe"]
        self.optical_center_style = self.style_dict["optical center"]

        # Create glyph coordinates.
        glyph_inCamera = \
            torch.tensor(
                [
                    [+0.0, 0.0, 0.0],  # Optical center
                    [+0.0, 0.0, 1.0],  # Image center.
                    [-1.0, -1.0, 1.0],  # Top left (seen from optical center).
                    [+1.0, -1.0, 1.0],  # Top right.
                    [+1.0, 1.0, 1.0],  # Bottom right.
                    [-1.0, 1.0, 1.0],  # Bottom left.
                    [-1.0, -1.0, 1.0],  # Back to top left.
                ]
            ).T * self.scale_mm

        transform_toVisualizationNode_fromCamera = \
            self.node_object.get_transform_toOther_fromSelf(
                self.visualization_node
            )

        self.glyph_inRoot = \
            transform_toVisualizationNode_fromCamera.transform(
                glyph_inCamera
            ).clone().detach().numpy()

    def visualize(self):
        # Add wireframe.
        self.wireframe_glyph = \
            self.axes.plot(*self.glyph_inRoot, **self.wireframe_style)

        # Add optical center.
        self.optical_center_glyph = \
            self.axes.plot(
                *self.glyph_inRoot[:, 0], **self.optical_center_style
            )

        if "show origin" in self.style_dict.keys():
            if self.style_dict["show origin"]:
                super().visualize()

        self.adjust_3D(self.axes)
