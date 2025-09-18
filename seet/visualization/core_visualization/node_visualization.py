""""
"Abstract class for visualizing a Node in 3D.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import json
import seet.core as core
import torch


class NodeVisualization():
    def __init__(
        self,
        other_node,
        axes,
        visualization_node=None,
        style_file_name=None,
        scale_mm=2.0,
        origin_style={"c": "k", "marker": "o"},
        colors=["r", "g", "b"]
    ):
        """"
        "Establish the coordinate system in which visualization is to
        happen.

        Args:
            node_object (Node): Node object to be visualized.

            axes (Axes): axes to which to add the visualization.

            visualization_node (Node, optional): node in whose coordinate
            system the input node is to be visualized. Defaults to None, which
            case the coordinate system of the root node of the pose graph is
            used.

            scale_mm (float, optional): scale of axes of node glyph. Defaults
            to 2.0 (in mm)

            style_file_name (str, optional): path to visualization style file
            of other_node. Defaults to None.
        """

        self.node_object = other_node
        self.axes = axes
        if visualization_node is None:  # Parameter None must still be there.
            self.visualization_node = other_node.get_root()
        else:
            self.visualization_node = visualization_node

        self.style_dict = NodeVisualization.load_style(style_file_name)
        self.scale_mm = scale_mm
        self.origin_style = origin_style
        self.colors = colors

    @staticmethod
    def load_style(style_file_name):
        if style_file_name is not None:
            with core.Node.open(style_file_name) as style_file_stream:
                style_dict = json.load(style_file_stream)
        else:
            style_dict = dict()

        return style_dict

    @staticmethod
    def adjust_3D(axes_3D):
        """adjust_3D.

        Fix problems with pyplot aspect ratio in 3D by using data-proportional scaling.
        This preserves the actual physical proportions of the data while ensuring
        proper aspect ratios based on the real data ranges.
        """

        # Get current axis limits
        xlim = axes_3D.get_xlim3d()
        ylim = axes_3D.get_ylim3d()
        zlim = axes_3D.get_zlim3d()
        
        # Calculate actual data ranges
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]  
        z_range = zlim[1] - zlim[0]
        
        # Set box aspect based on actual data proportions
        # This ensures the visualization reflects true physical dimensions
        axes_3D.set_box_aspect([x_range, y_range, z_range])

    @staticmethod
    def adjust_2D(axes_2D, camera):
        """adjust_2D.

        Fix aspect ratio and bounds of axes.

        Args:
            axes_2D (Axes): pyplot axes to be adjusted.

            camera (PinholeCamera): pinhole camera object from which we get
            axis boundaries.
        """

        axes_2D.set_aspect("equal")
        axes_2D.set_xlim([camera.lower_bounds[0], camera.upper_bounds[0]])
        axes_2D.set_ylim([camera.upper_bounds[1], camera.lower_bounds[1]])

    @staticmethod
    def get_pieces(input_list, wrap=True):
        """get_pieces.

        Given a list of objects that are either None or something other than
        None and get separate lists with the contiguous parts containing only
        the not None values. For example, from a list

        list = [None, tensor1, tensor2, None, None, tensor3]

        we get

        pieces = [[tensor1, tensor2], [tensor3]].

        If wrap is True, we wrap around, so if we have

        list = [tensor0, tensor1, None, tensor3, tensor4, None, tensor6]

        we get

        pieces = [[tensor6, tensor0, tensor1], [tensor3, tensor4]].

        This allows us to plot each piece as a single line plot.

        Args:
            input_list (list): list containing either (3,) torch tensors or
            None.

            wrap (bool, optional): if True, wrap around to determine whether
            two entries are contiguous. Defaults to True.

        Returns:
            list of lists: each inner list is a part of the original list with
            contiguous valid (not None) values, wrapping around.
        """

        N = len(input_list)
        if N == 0:
            return []

        pieces = []
        new_piece = []
        merge = False
        for i in range(N):
            point = input_list[i]

            if point is not None:
                # If the point is good, we add it to our current piece, which
                # may be empty if we are just starting a new one.
                new_piece = new_piece + [point, ]

                # Keep track of whether this is the first piece, as we may have
                # to merge it with the last piece in the end.
                if i == 0:
                    merge = True
                elif i == N - 1:
                    # We add the last piece to the list
                    pieces = pieces + [new_piece, ]
            else:
                # If the point is bad, our current piece is finished. If it is
                # not empty, we add it to our set of pieces.
                if len(new_piece) > 0:
                    pieces = pieces + [new_piece, ]

                # We are now going to start a new piece.
                new_piece = []

                if i == N - 1:
                    merge = False

        if wrap and merge and len(pieces) > 1:
            # The last piece wraps up back to the first one.
            pieces[-1] = pieces[-1] + pieces[0]

            # We no longer need the first piece.
            pieces = pieces[1:]

        return pieces

    def visualize(self):
        """visualize.

        Add a glyph visualizing the origin (black) and x (red), y (green),
        and z (blue) axes of the node's coordinate system.
        """

        # Get coordinates of keypoints in the coordinate system of the
        # visualization node.
        origin_inSelf = torch.zeros(3)
        xyz_inSelf = torch.eye(3) * self.scale_mm
        transform_toVisualizationNode_fromSelf = \
            self.node_object.get_transform_toOther_fromSelf(
                self.visualization_node
            )
        origin_inVisualizationNode = \
            transform_toVisualizationNode_fromSelf.transform(origin_inSelf)
        xyz_inVisualizationNode = \
            transform_toVisualizationNode_fromSelf.transform(xyz_inSelf)

        # Show the coordinates.
        self.axes.scatter(
            *origin_inVisualizationNode.clone().detach().numpy(),
            **self.origin_style
        )
        for i in range(3):
            direction = \
                torch.stack(
                    (xyz_inVisualizationNode[:, i], origin_inVisualizationNode)
                ).clone().detach().numpy()
            self.axes.plot(*direction.T, c=self.colors[i])

        self.adjust_3D(self.axes)
