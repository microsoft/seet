"""device_model_visualization.py

Visualization of eye-tracking device in 3D.
"""


__author__ = "Paulo. R. S. Mendonca (padossa@microsoft.com)"


from seet.visualization import core_visualization
from seet.visualization import device_visualization
from seet.visualization.device_visualization \
    import device_visualization_configs
import os


class DeviceModelVisualization(core_visualization.NodeVisualization):
    def __init__(
        self,
        et_device,
        axes_3D,
        visualization_node=None,
        style_file_name=os.path.join(
            device_visualization_configs.DEVICE_VISUALIZATION_DIR,
            r"default_device_style/default_device_model_style.json"
        )
    ):
        """
        Add a visualization of an eye-tracking device to existing 3D axes.

        Args:
            device (DeviceModel): eye-tracking device to be visualized.

            axes_3D (Axes3D): 3D axes to which add the visualization

            visualization_node (Node, optional): node in whose coordinate
            system the device is to be visualized. Defaults to None, in which
            case the coordinate system of the root node of the pose graph is
            used.

            parameter_file_name (str, optional): name of json file with style
            configuration parameters. Defaults to
            os.path.join(device_visualization_configs.DEVICE_VISUALIZATION_DIR,
            "default_device_style/default_device_model_style.json").
        """

        super().__init__(
            et_device,
            axes_3D,
            visualization_node,
            style_file_name=style_file_name
        )
        subsystems_styles = self.style_dict["et subsystems"]

        self.et_subsystem_glyphs = []
        num_subsystem_styles = len(subsystems_styles)
        style_counter = 0

        DEVICE_VISUALIZATION_DIR = self.style_dict.get("path")
        if DEVICE_VISUALIZATION_DIR is None:
            DEVICE_VISUALIZATION_DIR = \
                device_visualization_configs.DEVICE_VISUALIZATION_DIR

        for et_subsystem in self.node_object.subsystems:
            subsystem_style_file = \
                os.path.join(
                    DEVICE_VISUALIZATION_DIR,
                    subsystems_styles[style_counter % num_subsystem_styles]
                )

            self.et_subsystem_glyphs = \
                self.et_subsystem_glyphs + \
                [
                    device_visualization.SubsystemModelVisualization(
                        et_subsystem,
                        self.axes,
                        visualization_node=self.visualization_node,
                        style_file_name=subsystem_style_file
                    ),
                ]

            style_counter += 1

    def visualize(self):
        """visualize.

        Actual visualization.
        """
        for single_et_subsystem_glyph in self.et_subsystem_glyphs:
            single_et_subsystem_glyph.visualize()

        if "show origin" in self.style_dict.keys():
            if self.style_dict["show origin"]:
                super().visualize()

        self.adjust_3D(self.axes)
