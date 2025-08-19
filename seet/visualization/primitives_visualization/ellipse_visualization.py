"""ellipse_visualization.py

Visualization of an ellipse in 3D
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.visualization import core_visualization
import torch


class EllipseVisualization(core_visualization.NodeVisualization):
    def __init__(
        self,
        ellipse_node,
        axes_3D,
        visualization_node=None,
        num_points=30,
        show_origin=False,
        **kwargs
    ):
        """
        Add a visualization of an ellipse Node to existing 3D axes. The
        coordinate system of the visualization is determined by the optional
        parameter visualization_node. If visualization_node is None, the root
        node of the point set in the pose graph is used.

        Args:
            ellipse_node (Ellipse): ellipse node to be visualized.

            axes_3D (Axes3D): 3D axes to which add the visualization.

            num_points (int, optional): number of points to be sampled along
            ellipse. Defaults to 30.

            visualization_node (Node, optional): node in whose coordinate
            system the ellipse is to be visualized. Defaults to None, which
            case the coordinate system of the root node of the pose graph is
            used.

            num_points (int, optional): number of points to be sampled along
            ellipse. Defaults to 30.

            show_origin (bool, optional): if True, add a node glyph to the
            origin of the ellipse's coordinate system.

            kwargs (dict, optional): style parameters for plotting.
        """

        # This defines self.node_object (ellipse_node), self.axes, and
        # self.visualization_node.
        super().__init__(ellipse_node, axes_3D, visualization_node)

        self.show_origin = show_origin
        self.kwargs = kwargs

        coordinates_inVisualizationNode, _ = \
            ellipse_node.get_points_inOther(
                self.visualization_node, num_points=num_points
            )

        _, N = coordinates_inVisualizationNode.shape
        indices = list(range(N)) + [0, ]  # Circular indices, go back to 1s pt.
        self.coordinates_inVisualizationNode = \
            coordinates_inVisualizationNode[:, indices]

    def visualize(self):
        """visualize.

        Actual visualization.
        """

        coordinates = self.coordinates_inVisualizationNode
        self.contour_glyph = \
            self.axes.plot(
                *coordinates.clone().detach().numpy(), **self.kwargs
            )

        if self.show_origin:
            super().visualize()

        self.adjust_3D(self.axes)
