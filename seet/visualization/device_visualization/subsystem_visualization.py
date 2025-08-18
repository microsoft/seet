"""subsystem_visualization.py

Visualization of an eye-tracking subsystem in 3D.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.visualization import core_visualization
from seet.visualization import device_visualization
from seet.visualization.device_visualization \
    import device_visualization_configs
import os


class SubsystemModelVisualization(core_visualization.NodeVisualization):
    def __init__(
        self,
        et_subsystem,
        axes_3D,
        visualization_node=None,
        style_file_name=os.path.join(
            device_visualization_configs.DEVICE_VISUALIZATION_DIR,
            r"default_device_style/default_left_subsystem_style.json"
        )
    ):
        """
        Add a visualization of a device subsystem to existing 3D axes.

        Args:
            et_subsystem (SubsystemModel): _description_

            axes_3D (Axes3D): 3D axes to which add the visualization.

            visualization_node (Node, optional): node in whose coordinate
            system the subsystem is to be visualized. Defaults to None, in
            which case the coordinate system of the root node of the pose graph
            is used.
        """

        super().__init__(
            et_subsystem,
            axes_3D,
            visualization_node,
            style_file_name=style_file_name
        )

        #######################################################################
        # Visualize cameras.
        cameras_style = self.style_dict["cameras"]
        self.camera_glyphs = []
        num_camera_styles = len(cameras_style)
        style_counter = 0

        DEVICE_VISUALIZATION_DIR = self.style_dict.get("path")
        if DEVICE_VISUALIZATION_DIR is None:
            DEVICE_VISUALIZATION_DIR = \
                device_visualization_configs.DEVICE_VISUALIZATION_DIR

        for camera in et_subsystem.cameras:
            # Cycle through the camera styles.
            camera_style_file = \
                os.path.join(
                    device_visualization_configs.DEVICE_VISUALIZATION_DIR,
                    cameras_style[style_counter % num_camera_styles]
                )

            single_camera_glyph = device_visualization.CameraVisualization(
                camera,
                self.axes,
                visualization_node=self.visualization_node,
                style_file_name=camera_style_file
            )

            self.camera_glyphs = self.camera_glyphs + [single_camera_glyph, ]
            style_counter += 1

        #######################################################################
        # Visualize LEDs.
        leds_style = self.style_dict["leds"]
        leds_style_file = \
            os.path.join(
                device_visualization_configs.DEVICE_VISUALIZATION_DIR,
                leds_style
            )
        self.leds_glyph = \
            device_visualization.LEDsVisualization(
                et_subsystem.led_set,
                self.axes,
                visualization_node=self.visualization_node,
                style_file_name=leds_style_file
            )

        #######################################################################
        # Visualize Occluder.
        if et_subsystem.occluder is not None:
            if "occluder" in self.style_dict["occluder"]:
                occluder_style = self.style_dict["occluder"]
                occluder_style_file = \
                    os.path.join(
                        device_visualization_configs.DEVICE_VISUALIZATION_DIR,
                        occluder_style
                    )
                self.occluder_glyph = \
                    device_visualization.OccluderVisualization(
                        et_subsystem.occluder,
                        self.axes,
                        visualization_node=self.visualization_node,
                        style_file_name=occluder_style_file
                    )
            else:
                self.occluder_glyph = None
        else:
            self.occluder_glyph = None

    def visualize(self):
        """visualize.

        Actual visualization.
        """
        self.leds_glyph.visualize()
        if self.occluder_glyph is not None:
            self.occluder_glyph.visualize()
        for single_camera_glyph in self.camera_glyphs:
            single_camera_glyph.visualize()

        if "show origin" in self.style_dict.keys():
            if self.style_dict["show origin"]:
                super().visualize()

        self.adjust_3D(self.axes)
