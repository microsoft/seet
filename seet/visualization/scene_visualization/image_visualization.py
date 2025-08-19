"""image_visualization.py

Class for generating images of a given scene.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


from seet.visualization import core_visualization, user_visualization
from seet.visualization.scene_visualization \
    import scene_visualization_configs, scene_model_visualization
import os
import torch

from seet.visualization.user_visualization import user_visualization_configs


class ImageVisualization(core_visualization.NodeVisualization):
    """ImageVisualization.

    Class for creating eye-tracking images.
    """

    def __init__(
        self,
        et_scene,
        axes_2D,
        subsystem_index=0,
        camera_index=0,
        style_file_name=os.path.join(
            scene_visualization_configs.SCENE_VISUALIZATION_DIR,
            r"default_scene_style/default_scene_model_style.json"
        )
    ):
        """
        Add an image to existing 2D axes.

        Args:
            et_scene (SceneModel): eye-tracking scene from which to generate
            images.

            axes_2D (Axes): 2D axes to which to add the image.

            subsystem_index (int, optional): subsystem of the scene which is
            used to visualize the eye. Defaults to 0.

            camera_index (int, optional): camera within the subsystem which is
            used to visualize the eye. Defaults to 0.

            style_file_name (_type_, optional): path for style-configuration
            file. Defaults to os.path.join(
            scene_visualization_configs.SCENE_VISUALIZATION_DIR,
            "default_scene_style/default_scene_model_style.json" ).
        """

        super().__init__(
            et_scene, axes_2D, None, style_file_name=style_file_name
        )

        self.subsystem_index = subsystem_index
        self.camera_index = camera_index
        self.subsystem = \
            self.node_object.device.subsystems[self.subsystem_index]
        self.camera = self.subsystem.cameras[self.camera_index]

        cls = scene_model_visualization.SceneModelVisualization
        self.glint_style, \
            self.refracted_pupil_style, \
            self.occluding_contour_style = \
            cls.get_eye_feature_styles(self.style_dict)

        USER_VISUALIZATION_DIR = self.style_dict.get("path")
        if USER_VISUALIZATION_DIR is None:
            USER_VISUALIZATION_DIR = \
                user_visualization_configs.USER_VISUALIZATION_DIR
        user_style_file_name = self.style_dict["user"]
        user_style_file_name = \
            os.path.join(USER_VISUALIZATION_DIR, user_style_file_name)
        user_style_dict = \
            core_visualization.NodeVisualization.load_style(
                user_style_file_name
            )

        USER_VISUALIZATION_DIR = user_style_dict.get("path")
        if USER_VISUALIZATION_DIR is None:
            USER_VISUALIZATION_DIR = \
                user_visualization.USER_VISUALIZATION_DIR
        eye_style_file_name = \
            user_style_dict["eyes"][subsystem_index]["eye"]
        eye_style_file_name = \
            os.path.join(USER_VISUALIZATION_DIR, eye_style_file_name)
        self.eye_style_dict = \
            core_visualization.NodeVisualization.load_style(
                eye_style_file_name
            )

    def visualize(self):
        """visualize.

        Helper method to visualize everything.
        """

        self.visualize_cornea()
        self.visualize_eyelids()
        self.visualize_limbus()
        self.visualize_occluding_contour()
        self.visualize_refracted_pupil()
        self.visualize_glints()

        self.adjust_2D(self.axes, self.camera)

    def visualize_glints(
        self,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True,
        **kwargs
    ):
        """
        Visualize the glints in 2D as seen from the indicated camera.

        Args:
            num_points (int, optional): number of points to sample along the
            glints. Defaults to 30.

            apply_device_occluder (bool, optional): whether to omit glints
            occluded by the occluder. Defaults to True.

            apply_eyelids_occlusion (bool, optional): if True, glints behind
            the eyelids are removed. Defaults to True.

            kwargs (dict, optional): dictionary with additional parameters to
            control appearance of plot. Defaults to None.

        Returns:
            handle: pyplot handles for visualization of glints.
        """

        glints_inCamera = \
            self.node_object.generate_glints_inOther(
                other_node=self.camera,
                subsystem_index=self.subsystem_index,
                camera_index=self.camera_index,
                apply_device_occluder=apply_device_occluder,
                apply_eyelids_occlusion=apply_eyelids_occlusion
            )

        glints_inPixels = []
        for single_glint_inCamera in glints_inCamera:
            if single_glint_inCamera is not None:
                single_glint_inPixels = \
                    self.camera.project_toPixels_fromCamera(
                        single_glint_inCamera
                    )

                glints_inPixels = glints_inPixels + [single_glint_inPixels, ]

        if len(glints_inPixels) > 0:
            glints_inPixels = torch.stack(glints_inPixels)

            # Plot as a scatter plot.
            self.glints_glyph = \
                self.axes.scatter(
                    *glints_inPixels.T.clone().detach().numpy(),
                    s=self.glint_style["s"],
                    c=self.glint_style["c"],
                    marker=self.glint_style["marker"],
                    **kwargs
                )

        self.adjust_2D(self.axes, self.camera)

    def visualize_refracted_pupil(
        self,
        num_points=30,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True,
        **kwargs
    ):
        """
        Visualize the refracted pupil in 2D as seen from the indicated
        camera.

        Args:
            num_points (int, optional): number of points to sample along the
            pupil. Defaults to 30.

            apply_device_occluder (bool, optional): whether to omit pupil
            points occluded by the occluder. Defaults to True.

            apply_eyelids_occlusion (bool, optional): if True, refraction
            points behind the eyelids are None. Defaults to True.

            kwargs (dict, optional): dictionary with additional parameters to
            control appearance of plot. Defaults to None.

        Returns:
            handle: pyplot handles for visualization of pupil.
        """

        pupil_inCamera, _ = \
            self.node_object.generate_refracted_pupil_inOther(
                other_node=self.camera,
                subsystem_index=self.subsystem_index,
                camera_index=self.camera_index,
                num_points=num_points,
                apply_device_occluder=apply_device_occluder,
                apply_eyelids_occlusion=apply_eyelids_occlusion
            )

        # Project the pupil 3D points onto the image.
        pupil_inPixels = []
        for pt_inCamera in pupil_inCamera:
            if pt_inCamera is None:
                pupil_inPixels = pupil_inPixels + [None, ]
            else:
                pt_inPixels = \
                    self.camera.project_toPixels_fromCamera(pt_inCamera)
                pupil_inPixels = pupil_inPixels + [pt_inPixels, ]

        # Let's visualize the pupil as a continuous curve rather than a set of
        # scattered points. To do that, we have to find all the contiguous
        # pieces.
        pieces = \
            core_visualization.NodeVisualization.get_pieces(pupil_inPixels)

        # Plot each piece.
        self.refracted_pupil_glyph = []
        for single_piece in pieces:
            self.refracted_pupil_glyph = \
                self.refracted_pupil_glyph + \
                [
                    self.axes.plot(
                        *torch.stack(single_piece).T.clone().detach().numpy(),
                        c=self.refracted_pupil_style["c"],
                        linestyle=self.refracted_pupil_style["linestyle"],
                        linewidth=self.refracted_pupil_style["linewidth"],
                        **kwargs
                    )
                ]

        self.adjust_2D(self.axes, self.camera)

    def visualize_occluding_contour(
        self,
        num_points=30,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True,
        **kwargs
    ):
        """
        Visualize the corneal bulge in the image acquired by the selected
        camera in the selected subsystem.

        Arg:
            num_points (int, optional): number of points to sample along the
            occluding contour. Defaults to 30.

            apply_device_occluder (bool, optional): whether to omit points
            occluded by the occluder. Defaults to True.

            apply_eyelids_occlusion (bool, optional): if True, points on the
            occluding contour behind the eyelids are None. Defaults to True.

            kwargs (dict, optional): dictionary with additional parameters to
            control appearance of plot. Defaults to None.

        Returns:
            handle: pyplot handles for visualization of occluding contour.
        """

        occluding_contour_inCamera = \
            self.node_object.generate_occluding_contour_inOther(
                other_node=self.camera,
                subsystem_index=self.subsystem_index,
                camera_index=self.camera_index,
                num_points=num_points,
                apply_device_occluder=apply_device_occluder,
                apply_eyelids_occlusion=apply_eyelids_occlusion
            )

        occluding_contour_inPixels = []
        for single_contour_inCamera in occluding_contour_inCamera:
            if single_contour_inCamera is not None:
                single_contour_inPixels = \
                    self.camera.project_toPixels_fromCamera(
                        single_contour_inCamera
                    )
            else:
                single_contour_inPixels = None

            occluding_contour_inPixels = \
                occluding_contour_inPixels + [single_contour_inPixels, ]

        pieces = \
            core_visualization.NodeVisualization.get_pieces(
                occluding_contour_inPixels, wrap=True
            )
        self.occluding_contour_glyphs = []
        for single_piece in pieces:
            glyph = \
                self.axes.plot(
                    *torch.stack(single_piece).T.clone().detach().numpy(),
                    c=self.occluding_contour_style["c"],
                    linestyle=self.occluding_contour_style["linestyle"],
                    linewidth=self.occluding_contour_style["linewidth"],
                    **kwargs
                )

            self.occluding_contour_glyphs = \
                self.occluding_contour_glyphs + [glyph, ]

        self.adjust_2D(self.axes, self.camera)

    def visualize_cornea(
        self,
        num_level_sets=7,
        num_points=30,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True,
        **kwargs
    ):
        """
        Visualize the cornea in the image acquired by the selected camera in
        the selected subsystem. This will also show the occluding contour of
        the cornea from the view point of the camera, if one is present.

        Args:
            num_level_sets (int, optional): number of level sets to be sampled
            from the cornea. Defaults to 7.

            num_points (int, optional): number of points to sample along the
            level sets. Defaults to 30.

            apply_device_occluder (bool, optional): whether to omit points
            occluded by the occluder. Defaults to True.

            apply_eyelids_occlusion (bool, optional): if True, points on the
            occluding contour behind the eyelids are None. Defaults to True.

            kwargs (dict, optional): dictionary with additional parameters to
            control appearance of plot. Defaults to None.

        Returns:
            handle: pyplot handles for visualization of cornea.
        """

        # To visualize the cornea, we sample ellipses as level sets, and
        # visualize each ellipse independently.

        eye = self.node_object.user.eyes[self.subsystem_index]
        ellipses = eye.sample_cornea_level_sets(num_level_sets=num_level_sets)

        # Project the points from each ellipse.
        cornea_style = self.eye_style_dict["cornea"]
        self.cornea_glyphs = []
        for single_ellipse in ellipses:
            points_inCamera, _ = \
                single_ellipse.get_points_inOther(
                    self.camera, num_points=num_points
                )
            points_inPixels = []
            for single_point_inCamera in points_inCamera.T:
                single_point_inPixels = \
                    self.camera.project_toPixels_fromCamera(
                        single_point_inCamera
                    )
                points_inPixels = points_inPixels + [single_point_inPixels, ]

            glyph = \
                self.axes.plot(
                    *torch.stack(points_inPixels).T.clone().detach().numpy(),
                    c=cornea_style["c"],
                    linestyle=cornea_style["linestyle"],
                    alpha=cornea_style["alpha"],
                    **kwargs
                )
            self.cornea_glyphs = self.cornea_glyphs + [glyph, ]

            # Now that we have plotted this ellipse, we can remove it from the
            # pose graph.
            eye.cornea.remove_child(single_ellipse)

        self.visualize_occluding_contour(
            num_points=num_points,
            apply_device_occluder=apply_device_occluder,
            apply_eyelids_occlusion=apply_eyelids_occlusion,
            **kwargs
        )

    def visualize_limbus(
        self,
        num_points=30,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True,
        apply_corneal_bulge_occlusion=True,
        **kwargs
    ):
        """
        Visualize the limbus observed by the selected camera in the selected
        subsystem.

        Args:
            num_points (int, optional): Number of points to sample along the
            limbus. Defaults to 30.

            apply_device_occluder (bool, optional): If True, exclude limbus
            points occluded by the device occluder. Defaults to True.

            apply_eyelids_occlusion (bool, optional): If True, exclude limbus
            points behind the eyelids. Defaults to True.

            kwargs (dict, optional): dictionary with additional parameters to
            control appearance of plot. Defaults to None.
        """

        limbus_inCamera, _ = self.node_object.generate_limbus_inOther(
            self.camera,
            subsystem_index=self.subsystem_index,
            camera_index=self.camera_index,
            num_points=num_points,
            apply_device_occluder=apply_device_occluder,
            apply_eyelids_occlusion=apply_eyelids_occlusion,
            apply_corneal_bulge_occlusion=apply_corneal_bulge_occlusion
        )

        limbus_inPixels = []
        for single_point_inCamera in limbus_inCamera:
            if single_point_inCamera is not None:
                single_limbus_inPixels = \
                    self.camera.project_toPixels_fromCamera(
                        single_point_inCamera
                    )
            else:
                single_limbus_inPixels = None

            limbus_inPixels = limbus_inPixels + [single_limbus_inPixels, ]

        pieces = \
            core_visualization.NodeVisualization.get_pieces(
                limbus_inPixels, wrap=True
            )
        limbus_style = self.eye_style_dict["limbus"]
        self.limbus_glyphs = []
        for single_piece in pieces:
            glyph = \
                self.axes.plot(
                    *torch.stack(single_piece).T.clone().detach().numpy(),
                    c=limbus_style["c"],
                    linestyle=limbus_style["linestyle"],
                    **kwargs
                )

            self.limbus_glyphs = self.limbus_glyphs + [glyph, ]

        self.adjust_2D(self.axes, self.camera)

    def visualize_eyelids(
        self,
        num_points=30,
        apply_device_occluder=True,
        apply_corneal_bulge_occlusion=True,
        **kwargs
    ):
        """
        Visualize the eyelids observed by the object's camera in the
        selected subsystem.

        Args:
            num_points (int, optional): Number of points to sample along each
            eyelid. Defaults to 30.

            apply_device_occluder (bool, optional): If True, exclude points
            occluded by the device's occluder. Defaults to True.

            apply_corneal_bulge_occlusion (bool, optional): If True, exclude
            points occluded by the corneal bulge. Defaults to True.

            kwargs (dict, optional): dictionary with additional parameters to
            control appearance of plot. Defaults to None.
        """

        eye = self.node_object.user.eyes[self.subsystem_index]
        eyelids_inCamera = \
            eye.generate_eyelids_inOther(
                self.camera,
                num_points=num_points,
                remove_points_beyond_limbus=True
            )

        # Remove points occluded by the device occluder, if required.
        if apply_device_occluder:
            # There are two eyelids.
            for i in range(len(eyelids_inCamera)):
                eyelids_inCamera[i] = \
                    self.subsystem.apply_occluder_inOther(
                        self.camera,
                        eyelids_inCamera[i],
                        reference_point_inOther=torch.zeros(3)
                )

        # Remove points beyond the corneal bulge, if required.
        if apply_corneal_bulge_occlusion:
            # Two eyelids
            for i in range(len(eyelids_inCamera)):
                eyelids_inCamera[i] = \
                    self.node_object.apply_occluding_contour_inOther(
                        self.camera,
                        eyelids_inCamera[i],
                        subsystem_index=self.subsystem_index,
                        reference_point_inOther=torch.zeros(3)
                )

        eyelid_keys = ["upper eyelid", "lower eyelid"]
        self.eyelid_glyphs = []
        for i in range(len(eyelids_inCamera)):
            pieces = \
                core_visualization.NodeVisualization.get_pieces(
                    eyelids_inCamera[i], wrap=False  # Not closed curves.
                )

            eyelid_style_dict = self.eye_style_dict[eyelid_keys[i % 2]]
            single_eyelid_glyph = []
            for single_piece in pieces:
                # Compute the projection of the eyelid cameras onto the image
                # coordinate system.
                eyelids_inPixels = []
                for single_eyelid_point_inCamera in single_piece:
                    single_eyelid_point_inPixels = \
                        self.camera.project_toPixels_fromCamera(
                            single_eyelid_point_inCamera
                        )
                    eyelids_inPixels = eyelids_inPixels + \
                        [single_eyelid_point_inPixels, ]

                glyph = \
                    self.axes.plot(
                        *torch.stack(
                            eyelids_inPixels
                        ).T.clone().detach().numpy(),
                        c=eyelid_style_dict["c"],
                        linestyle=eyelid_style_dict["linestyle"],
                        label=eyelid_style_dict["label"],
                        **kwargs
                    )
                single_eyelid_glyph = single_eyelid_glyph + [glyph, ]

            self.eyelid_glyphs = self.eyelid_glyphs + [single_eyelid_glyph, ]

        self.adjust_2D(self.axes, self.camera)
