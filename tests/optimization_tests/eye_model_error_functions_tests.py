"""eye_model_error_functions_tests.py

Tests for eye-model error functions.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.optimization as optimization
import seet.scene as scene
import torch
import unittest


class TestEyeModelErrorFunctions(unittest.TestCase):
    """TestEyeModelErrorFunctions.

    Unit tests for error functions of eye model.
    """

    def setUp(self):
        """setUp.

        Generate data for tests
        """

        super().setUp()

        # Create the default scene, computing gradients.
        self.et_scene = scene.SceneModel(requires_grad=True)

        # Generate gaze directions.
        half_fov = self.et_scene.device.display_fov.numpy() / 2
        # Fov is horizontal by vertical, i.e., horizontal extent comes first.
        # When we rotate the eye the horizontal rotation comes first.
        # Therefore, the first index in the paris below is 0, the second index
        # is 1.
        all_rotation_angles_inParent = \
            torch.tensor(
                [
                    [-half_fov[0], -half_fov[1]],
                    [0.0, -half_fov[1]],
                    [+half_fov[0], -half_fov[1]],
                    [-half_fov[0], 0.0],
                    [0.0, 0.0],
                    [+half_fov[0], 0.0],
                    [-half_fov[0], +half_fov[1]],
                    [0.0, +half_fov[1]],
                    [+half_fov[0], +half_fov[1]]
                ],
                requires_grad=True
            )

        # Generate translations of rotation center.
        all_translations_inParent = \
            torch.tensor(
                [
                    [-2.0, -2.0, -4.0],
                    [+2.0, -2.0, -4.0],
                    [+2.0, +2.0, -4.0],
                    [-2.0, +2.0, -4.0],
                    [+0.0, +0.0, +0.0],
                    [+2.0, +2.0, +4.0],
                    [-2.0, -2.0, +4.0],
                    [+2.0, -2.0, +4.0],
                    [-2.0, +2.0, +4.0]
                ],
                requires_grad=True
            )

        self.num_poses = all_rotation_angles_inParent.shape[0]
        assert (self.num_poses == all_translations_inParent.shape[0])

        self.eyes_poses_pupil_points_inPixels = []
        self.eyes_poses_limbus_points_inPixels = []
        self.eyes_poses_glint_point_inPixels = []
        self.eyes_poses_matching_leds_inCamera = []
        self.eyes_poses = []

        for i, eye in enumerate(self.et_scene.user.eyes):
            # Generate pupil points for each frame.
            camera = self.et_scene.device.subsystems[i].cameras[0]

            eye_poses_pupil_points_inPixels = []
            eye_poses_limbus_points_inPixels = []
            eye_poses_glint_points_inPixels = []
            eye_poses = []
            for pose_index in range(self.num_poses):
                angles_deg = all_rotation_angles_inParent[pose_index, :]
                translation_inParent = all_translations_inParent[pose_index, :]

                ###############################################################
                # Apply a transformation of pose to the eye.
                eye.rotate_from_gaze_angles_inParent(angles_deg)
                eye.translate_inParent(translation_inParent)

                # Save the eye.
                eye_poses = eye_poses + [eye.branchcopy(eye), ]

                cornea = eye.cornea
                origin_inCornea = camera.get_optical_center_inOther(cornea)

                ###############################################################
                #  Pupil points.
                all_pupil_inCornea, _ = \
                    eye.generate_refracted_pupil_inCornea(origin_inCornea)

                # Project the pupil points into pixel coordinates.
                pupil_points_inPixels = []
                for pupil_inCornea in all_pupil_inCornea:
                    if pupil_inCornea is None:
                        pupil_inPixels = None
                    else:
                        # Get the pupil points in the image, pixel coordinates.
                        pupil_inPixels = \
                            camera.project_toPixels_fromOther(
                                pupil_inCornea, cornea
                            )

                    pupil_points_inPixels = \
                        pupil_points_inPixels + [pupil_inPixels, ]

                # Save the pupil points
                eye_poses_pupil_points_inPixels = \
                    eye_poses_pupil_points_inPixels + [pupil_points_inPixels, ]

                ###############################################################
                # Limbus points.
                all_limbus_inCamera, _ = \
                    eye.generate_limbus_inOther(other_node=camera)

                # Project the limbus points into pixel coordinates.
                limbus_points_inPixels = []
                for limbus_inCamera in all_limbus_inCamera:
                    if limbus_inCamera is None:
                        limbus_inPixels = None
                    else:
                        # Get the limbus in the image, pixel coordinates.
                        limbus_inPixels = \
                            camera.project_toPixels_fromCamera(limbus_inCamera)

                    limbus_points_inPixels = \
                        limbus_points_inPixels + [limbus_inPixels, ]

                # Save the limbus points.
                eye_poses_limbus_points_inPixels = \
                    eye_poses_limbus_points_inPixels + \
                    [limbus_points_inPixels, ]

                ###############################################################
                # Glints.
                all_leds_inCornea = \
                    self.\
                    et_scene.\
                    device.\
                    subsystems[i].led_set.get_coordinates_inOther(cornea)
                all_glints_inCornea = eye.generate_all_glints_inCornea(
                    origin_inCornea, all_leds_inCornea
                )

                # Project the glints into pixel coordinates.
                glints_inPixels = []
                for glint_inCornea in all_glints_inCornea:
                    if glint_inCornea is None:
                        glint_inPixels = None
                    else:
                        # Get the glint in the image, pixel coordinates.
                        glint_inPixels = \
                            camera.project_toPixels_fromOther(
                                glint_inCornea, cornea
                            )

                    glints_inPixels = glints_inPixels + [glint_inPixels, ]

                # Save the glints.
                eye_poses_glint_points_inPixels = \
                    eye_poses_glint_points_inPixels + [glints_inPixels, ]

                # Undo the transformation of pose to the eye.
                eye.translate_inParent(-translation_inParent)
                eye.unrotate_from_gaze_angles_inParent(angles_deg)

            self.eyes_poses = self.eyes_poses + [eye_poses, ]

            self.eyes_poses_pupil_points_inPixels = \
                self.eyes_poses_pupil_points_inPixels + \
                [eye_poses_pupil_points_inPixels, ]

            self.eyes_poses_limbus_points_inPixels = \
                self.eyes_poses_limbus_points_inPixels + \
                [eye_poses_limbus_points_inPixels, ]

            self.eyes_poses_glint_point_inPixels = \
                self.eyes_poses_glint_point_inPixels + \
                [eye_poses_glint_points_inPixels, ]

    def sampson_error_for_feature_inCamera(
        self,
        features_inPixels,
        error_function,
        is_glint=False
    ):
        """sampson_error_for_feature.

        Generic method to test errors in pupil in cornea and camera coordinate
        systems, limbus in limbus plane and camera coordinate systems, and
        glints in cornea and camera coordinate system.

        Args:
            features_inPixels (list of torch.Tensor): list of (2,) tensors
            holding coordinates of pupil points, limbus points, or glints in
            pixels.

            error_function (callable): Sampson error for pupil, limbus or
            glint.

            is_glint (bool, optional): if True, pass leds as argument of
            error_function. Defaults to false
        """

        # We need some margin of error, as Sampson error is an approximation.
        atol = 1e-1  # ... One tenth of a pixel and...
        rtol = 1e-2  # ... One percent.

        # Left and right.
        for i in range(len(self.eyes_poses)):
            camera = self.et_scene.device.subsystems[i].cameras[0]
            leds_inCamera = \
                self. \
                et_scene. \
                device.subsystems[i]. \
                led_set.get_coordinates_inOther(camera)

            eye_features_inPixels = features_inPixels[i]
            eye_poses = self.eyes_poses[i]
            for all_points_inPixels, eye in zip(
                eye_features_inPixels, eye_poses
            ):
                # Iterate over feature points.
                for led_idx, point_inPixels in enumerate(all_points_inPixels):
                    if point_inPixels is None:
                        # Skip occluded or otherwise invisible features.
                        continue

                    # Compute Sampson error with and without noise.
                    for std in (0.0, 1.0):
                        # Perturb the data according to std.
                        point_inPixels_ = \
                            point_inPixels + \
                            std * torch.randn_like(point_inPixels)

                        # Compute actual magnitude of perturbation, to generate
                        # an upper bound on the error.
                        magnitude_pix = \
                            torch.linalg.norm(point_inPixels_ - point_inPixels)

                        # Why the convoluted argument passing below? It is
                        # because this way we can call the function with
                        # arguments point, camera and eye, as in
                        #
                        # f(point, camera, eye)
                        #
                        # or we can call it as
                        #
                        # f(point, camera, eye, leds_inCamera)
                        #
                        # regardless of the function signature.
                        args = [camera, eye]
                        if is_glint:
                            args = args + [leds_inCamera[:, led_idx], ]
                        sampson_error = error_function(point_inPixels_, *args)

                        criterion = \
                            magnitude_pix * (1 + rtol) - \
                            torch.linalg.norm(sampson_error) > -atol
                        self.assertTrue(torch.all(criterion))

    def test_sampson_error_glint_inCamera(self):
        """test_sampson_error_glint_inCamera.

        Test Sampson errors for glint points.
        """

        self.sampson_error_for_feature_inCamera(
            self.eyes_poses_glint_point_inPixels,
            optimization.EyeModelErrorFunctions().sampson_error_glint_inCamera,
            is_glint=True
        )

    def test_sampson_error_pupil_inCamera(self):
        """test_sampson_error_pupil_inCamera.

        Test sampson errors for pupil points.
        """

        self.sampson_error_for_feature_inCamera(
            self.eyes_poses_pupil_points_inPixels,
            optimization.EyeModelErrorFunctions().sampson_error_pupil_inCamera,
            is_glint=False
        )

    def test_sampson_error_limbus_in_Camera(self):
        """test_sampson_error_limbus_in_Camera.

        Test Sampson errors for limbus points.
        """

    def test_sampson_error_pupil_inCornea(self):
        """test_sampson_error_pupil_inCornea.

        Test sampson errors for pupil points.
        """

        # We need some margin of error. Sampson error approximates reprojection
        # error within...
        atol = 1e-1  # ... One tenth of a pixel and...
        rtol = 1e-2  # ... One percent.

        # Left and right.
        for i in range(len(self.eyes_poses)):
            camera = self.et_scene.device.subsystems[i].cameras[0]

            eye_poses_pupil_points_inPixels = \
                self.eyes_poses_pupil_points_inPixels[i]
            eye_poses = self.eyes_poses[i]
            for pupil_points_inPixels, eye in zip(
                eye_poses_pupil_points_inPixels, eye_poses
            ):
                T_toCornea_fromCamera = \
                    camera.get_transform_toOther_fromSelf(eye.cornea)
                origin_inCornea = camera.get_optical_center_inOther(eye.cornea)

                # Iterate over pupil points.
                for point_inPixels in pupil_points_inPixels:
                    if point_inPixels is None:
                        continue

                    # Compute Sampson error with and without noise.
                    for std in (0.0, 1.0):
                        # Perturb the data according to std.
                        point_inPixels_ = \
                            point_inPixels + \
                            std * torch.randn_like(point_inPixels)

                        # Compute actual magnitude of perturbation, to generate
                        # an upper bound on the error.
                        magnitude_pix = \
                            torch.linalg.norm(point_inPixels_ - point_inPixels)

                        direction_inCamera = \
                            camera.compute_direction_toCamera_fromPixels(
                                point_inPixels_
                            )
                        direction_inCornea = \
                            T_toCornea_fromCamera.rotation.transform(
                                direction_inCamera
                            )

                        # We have to make sure that the ray through the
                        # perturbed pixel still intersects the cornea. The
                        # Sampson approximation will not work otherwise
                        # (although it will still be a good figure of merit for
                        # error.)
                        intersection_inCornea, _ = \
                            eye.cornea.\
                            intersect_from_origin_and_direction_inEllipsoid(
                                origin_inCornea, direction_inCornea
                            )

                        if intersection_inCornea is None:
                            continue

                        error = \
                            optimization.EyeModelErrorFunctions().\
                            sampson_error_pupil_inCornea(
                                point_inPixels_,
                                origin_inCornea,
                                direction_inCornea,
                                eye
                            )

                        # It is hard to predict what the error is going to be,
                        # because may may have just slid the point along the
                        # contour, in which case the error would be zero. But
                        # we know it cannot be much larger than the magnitude
                        # of the perturbation.
                        criterion = \
                            magnitude_pix * (1 + rtol) - \
                            torch.abs(error) > -atol
                        self.assertTrue(torch.all(criterion))

    def test_sampson_error_limbus_inLimbusPlane(self):
        """test_sampson_error_limbus_ray_inLimbusPlane.

        Test sampson errors for limbus points.
        """

        # We need some margin of error. Sampson error approximates reprojection
        # error within...
        atol = 1e-1  # ... One tenth of a pixel and...
        rtol = 1e-2  # ... One percent.

        # Left and right.
        for i in range(len(self.eyes_poses)):
            camera = self.et_scene.device.subsystems[i].cameras[0]

            eye_poses_limbus_points_inPixels = \
                self.eyes_poses_limbus_points_inPixels[i]
            eye_poses = self.eyes_poses[i]
            for limbus_points_inPixels, eye in zip(
                eye_poses_limbus_points_inPixels, eye_poses
            ):
                T_toLimbusPlane_fromCamera = \
                    camera.get_transform_toOther_fromSelf(eye.limbus_plane)
                origin_inPlane = \
                    camera.get_optical_center_inOther(eye.limbus_plane)

                # Iterate over limbus points.
                for point_inPixels in limbus_points_inPixels:
                    if point_inPixels is None:
                        continue

                    # Compute Sampson error with and without noise.
                    for std in (0.0, 1.0):
                        # Perturb the data according to std.
                        point_inPixels_ = \
                            point_inPixels + \
                            std * torch.randn_like(point_inPixels)

                        # Compute actual magnitude of perturbation, to generate
                        # an upper bound on the error.
                        magnitude_pix = \
                            torch.linalg.norm(point_inPixels_ - point_inPixels)

                        direction_inCamera = \
                            camera.compute_direction_toCamera_fromPixels(
                                point_inPixels_
                            )
                        direction_inPlane = \
                            T_toLimbusPlane_fromCamera.rotation.transform(
                                direction_inCamera
                            )

                        error = \
                            optimization.EyeModelErrorFunctions().\
                            sampson_error_limbus_inLimbusPlane(
                                point_inPixels_,
                                origin_inPlane,
                                direction_inPlane,
                                eye
                            )

                        # It is hard to predict what the error is going to be,
                        # because may may have just slid the point along the
                        # contour, in which case the error would be zero. But
                        # we know it cannot be much larger than the magnitude
                        # of the perturbation.
                        criterion = \
                            magnitude_pix * (1 + rtol) - \
                            torch.abs(error) > -atol
                        self.assertTrue(torch.all(criterion))
