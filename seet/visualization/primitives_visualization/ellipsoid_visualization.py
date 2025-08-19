"""ellipsoid_visualization.py

Visualization of an ellipsoid in 3D
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.visualization import core_visualization
from seet.visualization import primitives_visualization


class EllipsoidVisualization(core_visualization.NodeVisualization):
    def __init__(
        self,
        ellipsoid_node,
        axes_3D,
        visualization_node=None,
        min_level=0.0,
        max_level=None,
        num_ellipses=7,
        num_points_per_ellipse=30,
        show_origin=False,
        **kwargs
    ):
        """
        Add a visualization of an ellipsoid Node to existing 3D axes. The
        coordinate system of the visualization is determined by the optional
        parameter visualization_node. If visualization_node is None, the root
        node of the point set in the pose graph is used.

        Args:
            ellipsoid_node (Ellipsoid): ellipsoid node to be visualized.

            axes_3D (Axes3D): 3D axes to which add the visualization.


            visualization_node (Node, optional): node in whose coordinate
            system the ellipsoid is to be visualized. Defaults to None, in
            which case the coordinate system of the root node of the pose graph
            is used.

            min_level (float, optional): height of minimum level set of
            ellipsoid along its z axis. Defaults to 0.0, which corresponds to
            the center of the ellipsoid.

            max_level (float, optional): height of maximul level set of
            ellipsoid along its z axis. Defaults to None, which corresponds to
            the apex of the ellipsoid.

            num_ellipses (int, optional): number of elliptical level sets of
            visualization. Defaults to 7.

            num_points_per_ellipse (int, optional): number of points to sample
            on each elliptical level set. Defaults to 30.

            show_origin (bool, optional): if True, add a node glyph to the
            origin of the ellipsoid's coordinate system. Defaults to False.

            kwargs (dict, optional): style parmeters for plotting.
        """

        # This defines self.node_object (ellipsoid), self.axes, and
        # self.visualization_node.
        super().__init__(ellipsoid_node, axes_3D, visualization_node)

        self.show_origin = show_origin

        self.ellipse_glyphs = []
        self.ellipses = []
        for ellipse in \
                ellipsoid_node.sample_level_sets(
                    min_level=min_level,
                    max_level=max_level,
                    num_level_sets=num_ellipses
                ):

            ellipse_glyph = \
                primitives_visualization.EllipseVisualization(
                    ellipse,
                    self.axes,
                    visualization_node=self.visualization_node,
                    num_points=num_points_per_ellipse,
                    **kwargs
                )

            self.ellipse_glyphs = self.ellipse_glyphs + [ellipse_glyph, ]
            self.ellipses = self.ellipses + [ellipse, ]

    def visualize(self):
        """visualize.

        Actual visualization.
        """
        for single_ellipse_glyph in self.ellipse_glyphs:
            single_ellipse_glyph.visualize()

        # The sampling of level sets creates several additional nodes in the
        # pose graph. It is good bookeeping to remove them once no longer
        # needed.
        for single_ellipse in self.ellipses:
            self.node_object.remove_child(single_ellipse)

        if self.show_origin:
            super().visualize()

        self.adjust_3D(self.axes)
