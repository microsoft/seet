"""point_visualization.py

Visualization of a point in 3D.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.visualization import core_visualization


class PointVisualization(core_visualization.NodeVisualization):
    def __init__(
        self,
        point_node,
        axes_3D,
        visualization_node=None,
        show_origin=False,
        **kwargs
    ):
        """
        Add a visualization of a point Node to existing 3D axes. The
        coordinate system of the visualization is determined by the optional
        parameter visualization_node. If visualization_node is None, the root
        node of the pose graph is used to determine the coordinate system of
        the visualization.

        Args:
            point_node (Point): point to be visualized.

            axes_3D (Axes3D): 3D axes to which add the visualization.

            visualization_node (Node, optional): node in whose coordinate
            system the point is to be visualized. Defaults to None, in which
            case the coordinate system of the root node of the pose graph is
            used.

            show_origin (bool, optional): if true, show a coordinate system
            glyph at the location of the point. Defaults to False.

            kwargs (dict, optional): visualization style parameters.
        """

        # This defines self.node_object (point), self.axes, and
        # self.visualization_node.
        super().__init__(
            point_node, axes_3D, visualization_node=visualization_node
        )

        self.show_origin = show_origin
        self.kwargs = kwargs

        self.coordinates_inVisualizationNode = \
            point_node.get_coordinates_inOther(
                self.visualization_node
            ).clone().detach().numpy()

    def visualize(self):
        if self.show_origin:
            super().visualize()
        else:
            self.point_glyph = \
                self.axes.scatter(
                    *self.coordinates_inVisualizationNode, **self.kwargs
                )

        self.adjust_3D(self.axes)
