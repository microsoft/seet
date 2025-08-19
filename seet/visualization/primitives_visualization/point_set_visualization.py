"""point_set_visualization.py

Visualization of a point set in 3D
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.visualization import core_visualization


class PointSetVisualization(core_visualization.NodeVisualization):
    def __init__(
        self,
        point_set_node,
        axes_3D,
        visualization_node=None,
        show_origin=False,
        **kwargs,
    ):
        """
        Add a visualization of a point-set Node to existing 3D axes. The
        coordinate system of the visualization is determined by the optional
        parameter visualization_node. If visualization_node is None, the root
        node of the point set in the pose graph is used.

        Args:
            point_set_node (PointSet): point set node to be visualized.

            axes_3D (Axes3D): 3D axes to which add the visualization.

            visualization_node (Node, optional): node in whose coordinate
            system the point set is to be visualized. Defaults to None, which
            case the coordinate system of the root node of the pose graph is
            used.

            show_origin (bool, optional): if True, show a glyph in the origin
            of the point-set coordinate system. Defaults to False.

            kwargs (dict, optional): visualization style parmeters.
        """

        # This defines self.node_object (point_set_node), self.axes, and
        # self.visualization_node.
        super().__init__(point_set_node, axes_3D, visualization_node)

        self.show_origin = show_origin
        self.kwargs = kwargs

        self.coordinates_inVisualizationNode = \
            point_set_node.get_coordinates_inOther(
                self.visualization_node
            ).clone().detach().numpy()

    def visualize(self):
        self.point_set_glyph = \
            self.axes.scatter(
                *self.coordinates_inVisualizationNode, **self.kwargs
            )

        if self.show_origin:
            super().visualize()

        self.adjust_3D(self.axes)
