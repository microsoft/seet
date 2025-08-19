"""scene_model_visualization.py

Class for visualizing scene models.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.visualization import core_visualization
from seet.visualization import device_visualization
from seet.visualization import user_visualization
from seet.visualization.scene_visualization \
    import scene_visualization_configs
import os
import torch


class SceneModelVisualization(core_visualization.NodeVisualization):
    """SceneModelVisualization.

    Class for visualizing scene models.
    """

    def __init__(
        self,
        et_scene,
        axes_3D,
        visualization_node=None,
        style_file_name=os.path.join(
            scene_visualization_configs.SCENE_VISUALIZATION_DIR,
            r"default_scene_style/default_scene_model_style.json"
        )
    ):
        """
        Add a visualization of an eye-tracking scene to existing 3D axes.

        Args:
            et_scene (SceneModel): eye-tracking scene to be visualized.

            axes_3D (Axes3D): 3D axes to which add the visualization.

            visualization_node (Node, optional): node in whose coordinate
            system the ellipsoid is to be visualized. Defaults to None, in
            which case the coordinate system of the root node of the pose graph
            is used.

            style_file_name (str, optional): path for style-configuration file.
            Defaults to os.path.join(
            scene_visualization_configs.SCENE_VISUALIZATION_DIR,
            "default_scene_style/default_scene_model_style.json")
        """

        super().__init__(
            et_scene,
            axes_3D,
            visualization_node,
            style_file_name=style_file_name
        )

        # User only.
        user_style = self.style_dict["user"]

        # Device only.
        device_style = self.style_dict["device"]

        # User + device interaction.
        self.glint_style, \
            self.refracted_pupil_style, \
            self.occluding_contour_style = \
            self.get_eye_feature_styles(self.style_dict)

        # This defines self.node_object (et_scene), self.axes, and
        # self.visualization_node.
        super().__init__(
            et_scene, axes_3D, visualization_node=visualization_node
        )

        USER_VISUALIZATION_DIR = self.style_dict.get("path")
        if USER_VISUALIZATION_DIR is None:
            USER_VISUALIZATION_DIR = \
                user_visualization.USER_VISUALIZATION_DIR
        self.user_glyph = \
            user_visualization.UserModelVisualization(
                et_scene.user,
                self.axes,
                visualization_node=self.visualization_node,
                style_file_name=os.path.join(
                    USER_VISUALIZATION_DIR, user_style
                )
            )

        DEVICE_VISUALIZATION_DIR = self.style_dict.get("path")
        if DEVICE_VISUALIZATION_DIR is None:
            DEVICE_VISUALIZATION_DIR = \
                device_visualization.DEVICE_VISUALIZATION_DIR
        self.device_glyph = \
            device_visualization.DeviceModelVisualization(
                et_scene.device,
                self.axes,
                visualization_node=visualization_node,
                style_file_name=os.path.join(
                    DEVICE_VISUALIZATION_DIR, device_style
                )
            )

    @staticmethod
    def get_eye_feature_styles(style_dict):
        """get_eye_feature_styles.

        Read from the dictionary the style in which to plot eye features
        that depend on the scene, i.e., glints, refracted pupil, and occluding
        contour.

        Args:
            style_dict (dict): dictionary with style of eye features.

        Returns:
            list of dict: list with five dictionaries, each corresponding to
            the plotting style of a different eye feature: glint_style,
            refracted_pupil_style, and occluding_contour_style.
        """

        default_glint_style = {"s": 2, "c": "k", "marker": "."}
        glint_style = style_dict.get("glints", default_glint_style)
        for key, value in default_glint_style.items():
            glint_style[key] = glint_style.get(key, value)

        default_refracted_pupil_style = {"c": "orange", "linestyle": "-"}
        refracted_pupil_style = \
            style_dict.get("refracted pupil", default_refracted_pupil_style)
        for key, value in default_refracted_pupil_style.items():
            refracted_pupil_style[key] = refracted_pupil_style.get(key, value)

        default_occluding_contour_style = {"c": "m", "linestyle": "-"}
        occluding_contour_style = \
            style_dict.get(
                "occluding contour", default_occluding_contour_style
            )
        for key, value in default_occluding_contour_style.items():
            occluding_contour_style[key] = \
                occluding_contour_style.get(key, value)

        return glint_style, refracted_pupil_style, occluding_contour_style

    def visualize(self):
        """visualize.

        Visualize components that do not interact with each other. That
        means user-dependent elements such as limbus, cornea, and pupil, and
        device-dependent elements such as camera and occluders.
        """

        self.user_glyph.visualize()
        self.device_glyph.visualize()

        self.adjust_3D(self.axes)

    def visualize_glints(
        self,
        subsystem_index=0,
        camera_index=0,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True,
        **kwargs
    ):
        """
        Visualize glints in 3D.
        """

        # self.node_object is the original et_scene used to initialize the
        # visualization object.
        glints_inVisualizationNode = \
            self.node_object.generate_glints_inOther(
                other_node=self.visualization_node,
                subsystem_index=subsystem_index,
                camera_index=camera_index,
                apply_device_occluder=apply_device_occluder,
                apply_eyelids_occlusion=apply_eyelids_occlusion
            )

        # Prune out occluded glints.
        visible_glints_inVisualizationNode = \
            [
                single_glint_inVisualization
                for single_glint_inVisualization in glints_inVisualizationNode
                if single_glint_inVisualization is not None
            ]

        if len(visible_glints_inVisualizationNode) == 0:
            return

        self.glints_glyph = \
            self.axes.scatter(
                *torch.stack(
                    visible_glints_inVisualizationNode
                ).T.clone().detach().numpy(),
                s=self.glint_style["s"],
                c=self.glint_style["c"],
                marker=self.glint_style["marker"],
                **kwargs
            )

    def visualize_refracted_pupil(
        self,
        subsystem_index=0,
        camera_index=0,
        num_points=30,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True,
        **kwargs
    ):
        """
        Visualize the pupil refraction points in 3D.

        Args:
            subsystem_index (int, optional): index of subsystem used to produce
            the refracted pupil points in 3D. Defaults to 0.

            camera_index (int, optional): index of the camera in the subsystem
            used to produce the refracted pupil points in 3D. Defaults to 0.

            num_points (int, optional): number of points to sample along
            refracted pupil. Defaults to 30.

            appply_device_occluder (bool, optional): if True, apply occlusion
            by device's occluder, if any. Defaults to True.

            apply_eyelids_occlusion (bool, optional): if True, apply occlusion
            by the user's eyelids. Defaults to True.

            kwargs (dict, optional): dictionary with additional parameters to
            control appearance of plot. Defaults to None.
        """

        # self.node_object is the original et_scene used to initialize the
        # visualization object.
        refracted_pupil_inVisualizationNode, _ = \
            self.node_object.generate_refracted_pupil_inOther(
                other_node=self.visualization_node,
                subsystem_index=subsystem_index,
                camera_index=camera_index,
                apply_device_occluder=apply_device_occluder,
                apply_eyelids_occlusion=apply_eyelids_occlusion,
                num_points=num_points
            )

        N = len(refracted_pupil_inVisualizationNode)
        if N == 0:
            return

        # Let's visualize the pupil as a continuous curve rather than a set of
        # scattered points. To do that, we have to find all the contiguous
        # pieces.
        pieces = self.get_pieces(refracted_pupil_inVisualizationNode)

        # Plot each piece.
        self.pupil_glyph = []
        for single_piece in pieces:
            glyph = \
                self.axes.plot(
                    *torch.stack(single_piece).T.clone().detach().numpy(),
                    c=self.refracted_pupil_style["c"],
                    linestyle=self.refracted_pupil_style["linestyle"],
                    **kwargs
                )
            self.pupil_glyph = self.pupil_glyph + [glyph, ]

    def visualize_occluding_contour(
        self,
        subsystem_index=0,
        camera_index=0,
        num_points=30,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True,
        **kwargs
    ):
        """
        Visualize the occluding contour of the user's eye visible to the
        given camera in the given subsystem of the scene's device.

        Args:
            subsystem_index (int, optional): index of subsystem in scene's
            device used to generate the occluding contour. Defaults to 0.

            camera_index (int, optional): index of the camera in the device's
            subsystem used to generate the occluding contour. Defaults to 0.

            num_points (int, optional): number of points to sample along
            occluding contour. Defaults to 30.

            apply_device_occluder (bool, optional): if true, apply occlusion by
            the subsystem's occluder, if one is present. Defaults to True.

            apply_eyelids_occlusion (bool, optional): if True, do not show
            points of occluding contour under the eyelids. Defaults to True.

            kwargs (dict, optional): keyword arguments used to plot.

        Returns:
            handles of pieces of plotted occluding contours.
        """

        occluding_contour_inVisualizationNode = \
            self.node_object.generate_occluding_contour_inOther(
                other_node=self.visualization_node,
                subsystem_index=subsystem_index,
                camera_index=camera_index,
                num_points=num_points,
                apply_device_occluder=apply_device_occluder,
                apply_eyelids_occlusion=apply_eyelids_occlusion
            )

        pieces = \
            self.get_pieces(occluding_contour_inVisualizationNode, wrap=False)
        self.occluding_contour_glyphs = []
        for single_piece in pieces:
            glyph = \
                self.axes.plot(
                    *single_piece,
                    c=self.occluding_contour_style["c"],
                    linestyle=self.occluding_contour_style["linestyle"],
                    **kwargs
                )
            self.occluding_contour_glyphs = \
                self.occluding_contour_glyphs + [glyph, ]
