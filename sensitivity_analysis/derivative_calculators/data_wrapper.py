"""Wrap data required for the computation of derivatives.

By having the derivative-calculator classes hold a reference to the data
wrapper will give access by those classes to the same object in the pytorch
computational graph, saving a lot of computational effort.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.scene as scene
from sensitivity_analysis.derivative_calculators import \
    derivative_parameters


class DataWrapper():
    """Wraps objects required to compute derivatives for sensitivity analysis.
    """

    def __init__(self, et_scene: scene.SceneModel):
        """Sets the subsystem with respect to which we compute derivatives.

        Args:
            et_scene (scene.SceneModel): eye-tracking scene containing the eye
            and device in which to perform sensitivity analysis.
        """

        self.scene = et_scene
        self.device = self.scene.device
        self.user = self.scene.user

        self.set_eye_and_subsystem(index=0)

    def set_eye_and_subsystem(
        self, index=0, num_pupil_points=30, num_limbus_points=30
    ):
        """Select the 'side' (left/right) of the device and corresponding eye.

        Args:
            index (int, optional): 0 corresponds to left side, anything else to
            right side. Defaults to 0.

            num_pupil_points (int, optional): number of points to be sampled
            along the pupil. Defaults to 30.

            num_limbus_points (int, optional): number of points to be sampled
            along the limbus. Defaults to 30.
        """

        self._set_subsystem(index)
        self._set_eye(num_pupil_points, num_limbus_points)

        # Set parameters that depend on both the eye and the device.
        self._compute_camera_origin_inCornea()
        self._compute_leds_inCornea()

    def _set_subsystem(self, index):
        """Select the subsystem of the scene's device used for analysis.

        Args:
            index (int): index of subsystem.
        """

        # Easy access to some data.
        self.subsystem_index = index
        self.subsystem = self.device.subsystems[self.subsystem_index]
        self.leds = self.subsystem.led_set
        self.camera = self.subsystem.cameras[0]

        # Set device-dependent parameters with respect to which we compute
        # derivatives.
        self.led_locations = \
            derivative_parameters.LEDsParameters(self.subsystem)
        self.camera_extrinsics = \
            derivative_parameters.CameraExtrinsicParameters(self.camera)
        self.camera_intrinsics = \
            derivative_parameters.CameraIntrinsicParameters(self.camera)

    def _set_eye(self, num_pupil_points, num_limbus_points):
        """Set the eye and eye-dependent parameters.

        Once we set the eye we can then compute the camera origin and LEDs in
        the coordinate system of the eye's cornea.

        Args:
            num_pupil_points (int): number of pupil points that will be sampled
            around the pupil. May be different from number of visible pupil
            points if there is occlusion.

            num_limbus_points (int): number of limbus points that will be
            sampled around the limbus. May be different from number of visible
            limbus points if the is occlusion.
        """

        # Easy access to some data
        self.eye = self.user.eyes[self.subsystem.eye_index]
        self.cornea = self.eye.cornea
        self.num_pupil_angles = num_pupil_points
        self.num_limbus_angles = num_limbus_points

        # Set eye-dependent parameters with respect to which we compute
        # derivatives.
        self.eye_pose_parameters = \
            derivative_parameters.EyePoseParameters(self.eye)
        self.eye_shape_parameters = \
            derivative_parameters.EyeShapeParameters(self.eye)

    def _compute_camera_origin_inCornea(self):
        """Compute the camera origin in the coordinate system of the cornea.
        """

        self.origin_inCornea = \
            self.camera.get_optical_center_inOther(self.cornea)

    def _compute_leds_inCornea(self):
        """Compute the LED coordinates in the coordinate system of the cornea.
        """

        self.leds_inCornea = self.leds.get_coordinates_inOther(self.cornea)

    def rotate_eye(self, angles_deg):
        """Rotate the eye and reset the derivative parameters.

        Once the eye is rotated, we reset the parameters with respect to which
        we compute derivatives.

        This method surfaces up the lower-level method rotate_eye from the
        EyePoseParameters class and resets the camera origin and led locations
        in the camera coordinate system.

        Args:
            angles_deg (torch.Tensor): (2,) torch tensor corresponding to the
            horizontal and vertical angles in degrees by which the gaze is
            rotated.
        """

        self.eye_pose_parameters.rotate_eye(angles_deg)

        # Parameters of the device in the eye coordinate system have changed.
        self._compute_camera_origin_inCornea()
        self._compute_leds_inCornea()

    def unrotate_eye(self, angles_deg):
        """Reverse the effect or eye rotation.

        Args:
            angles_deg (torch.Tensor): (2,) torch tensor corresponding to the
            horizontal and vertical angles in degrees by which the gaze is
            rotated.
        """

        self.eye_pose_parameters.unrotate_eye(angles_deg)

        # Parameters of the device in the eye coordinate system have changed.
        self._compute_camera_origin_inCornea()
        self._compute_leds_inCornea()

    def _generate_feature_inPixels(
        self, feature_name, method, *args, **kwargs
    ):
        """Generate glints, pupil, or limbus points.

        Args:
            feature_name (str): string specifying the name of the feature as
            'glint', 'pupil' or 'limbus'.

            method (callable): method to compute feature.

            *args: list of arguments to be passed to the method that computes
            the features.

        Returns:
            list of torch.Tensor: list of (2,) torch tensors corresponding to
            feature (glint, pupil, or limbus point) in pixel coordinates. The
            list will hold a value None for occluded or otherwise not visible
            features.
        """

        if feature_name.lower() == "glints":
            feature_inCornea = method(*args, **kwargs)
            angles_rad = None
        else:
            feature_inCornea, angles_rad = method(*args, **kwargs)

        return \
            self.camera.project_all_toPixels_fromOther(
                feature_inCornea, self.cornea
            ), \
            angles_rad

    def generate_glints_inPixels(self):
        """Generate glints common to the computation of derivatives.
        """

        self.all_glints_inPixels, _ = \
            self._generate_feature_inPixels(
                "glints",
                self.eye.generate_all_glints_inCornea,
                self.origin_inCornea,
                self.leds_inCornea
            )

        # We don't need to count the number of visible pupil and limbus points,
        # as those match the number of their respective lifting parameters. But
        # there are no lifting parameters for glints.
        self.num_visible_glints = 0
        for glint in self.all_glints_inPixels:
            if glint is not None:
                self.num_visible_glints += 1

    def generate_pupil_inPixels(self):
        """Generate pupil points common to the computation of derivatives.
        """

        self.pupil_inPixels, self.pupil_angles_rad = \
            self._generate_feature_inPixels(
                "pupil",
                self.eye.generate_refracted_pupil_inCornea,
                self.origin_inCornea,
                num_points=self.num_pupil_angles
            )

    def generate_limbus_inPixels(self):
        """Generate limbus points common to the computation of derivatives.
        """

        self.limbus_inPixels, self.limbus_angles_rad = \
            self._generate_feature_inPixels(
                "limbus",
                self.eye.generate_limbus_inOther,
                self.cornea,
                num_points=self.num_limbus_angles
            )
