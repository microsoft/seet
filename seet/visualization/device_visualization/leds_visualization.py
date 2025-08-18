"""leds_visualization.py

Visualization of LED geometry in 3D.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.visualization import core_visualization
from seet.visualization.device_visualization \
    import device_visualization_configs
import os


class LEDsVisualization(core_visualization.NodeVisualization):
    def __init__(
        self,
        led_set,
        axes_3D,
        visualization_node=None,
        style_file_name=os.path.join(
            device_visualization_configs.DEVICE_VISUALIZATION_DIR,
            r"default_device_style/default_left_leds_style.json"
        )
    ):
        """
        Add a visualization of LEDs to existing 3D axes.

        Args:
            led_set (LEDs): LEDs to be visualized.

            axes_3D (Axes3D): 3D axes to which add the visualization.
        """

        super().__init__(
            led_set,
            axes_3D,
            visualization_node,
            style_file_name=style_file_name
        )
        self.point_set_style = self.style_dict["point set"]

        # Create LED coordinates
        coordinates_inVisualizationNode = \
            self.node_object.get_coordinates_inOther(
                self.visualization_node
            ).clone().detach().numpy()
        _, N = coordinates_inVisualizationNode.shape
        self.coordinates_inVisualizationNode = \
            coordinates_inVisualizationNode[:, list(range(N)) + [0, ]]

    def visualize(self):
        """visualize.

        Actual visualization.
        """
        self.point_set_glyph = \
            self.axes.plot(
                *self.coordinates_inVisualizationNode, **self.point_set_style
            )

        if "show origin" in self.style_dict.keys():
            if self.style_dict["show origin"]:
                super().visualize()

        self.adjust_3D(self.axes)
