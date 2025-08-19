"""eye_model_errors_functions.py

Error functions to optimize eye model.
"""


__author__ = "Chris Aholt and Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.primitives as primitives
import torch


class EyeModelErrorFunctions():
    """EyeModelErrorFunctions.

    Class for computation of Sampson errors for glint, pupil, and limbus
    measurements. All methods are static, so they behave as standalone
    functions with a namespace. Errors are signed, and therefore appropriate
    for use with scipy non-linear least squares method.
    """

    @staticmethod
    def sampson_error_pupil_inCornea(
        pupil_point_inPixels,
        origin_pupil_ray_inCornea,
        direction_pupil_ray_inCornea,
        eye
    ):
        """sampson_error_pupil_inCornea.

        Sampson error approximation for the difference between a predicted and
        measured pupil point in an image. The cornea is the natural coordinate
        system for this computation, since the bulk of the problem is getting
        refraction right.

        Args:
            pupil_point_inPixels (torch.Tensor): (2,) pupil point in pixel
            coordinates corresponding. This input is required to rescale the 3D
            error in mm back to image coordinates in pixels.

            origin_pupil_ray_inCornea (torch.Tensor): (3,) origin of the pupil
            ray - typically the optical center of a camera - in the coordinate
            system of the cornea of the input eye.

            direction_pupil_ray_inCornea (torch.Tensor): (3,) direction of the
            pupil ray - i.e., the direction of the unprojected ray through the
            image of the input pupil point in the image plane of a camera - in
            the coordinate system of the cornea of the input eye.

            eye (EyeModel): eye model object.

        torch.Tensor: torch scalar corresponding to signed reprojection error
            in pixels of pupil point (negative if point is inside pupil,
            positive otherwise.)
        """

        # If the input ray does not intersect the cornea, we still intercept it
        # with the pupil plane.
        #
        #          Origin *->___
        #                       \___  Direction
        #                 ______    \___
        #               /        \      \___
        #              /  Cornea  \         \___
        # ____________/____________\____________\_______________
        # Pupil plane        * Pupil center      * Intersection
        #
        #                     <-------------------
        #                      Direction for which
        #                     error decreases brings
        #                      ray closer to cornea
        #
        intersection_inPupilPlane = \
            eye.intersect_ray_withPupilPlane_fromCornea(
                origin_pupil_ray_inCornea, direction_pupil_ray_inCornea
            )

        distance_to_center = torch.linalg.norm(intersection_inPupilPlane)

        unscaled_error = distance_to_center - eye.pupil.radius

        # We detach to match the C++ implementation, which does not consider
        # the derivative of the scale of the Sampson error.
        with torch.enable_grad():  # In case grad is disable in the context.
            d_error_d_pupil_point = \
                core.compute_auto_jacobian_from_tensors(
                    unscaled_error, pupil_point_inPixels
                ).detach()

        return unscaled_error / torch.linalg.norm(d_error_d_pupil_point)

    @staticmethod
    def sampson_error_pupil_inCamera(
        pupil_point_inPixels,
        camera,
        eye
    ):
        """Sampson error for pupil point from ray in camera coordinates.

        Compute the Sampson approximation for the reprojection error of a pupil
        point given the point, the eye, and the camera observing the eye.

        Args:
            pupil_point_inPixels (torch.Tensor): (2,) pupil point in pixels.

            camera (PinholeCamera): camera observing pupil point from eye. Note
            that a Polynomial3KCamera is a PinholeCamera.

            eye (EyeModel): eye-model object.

        Returns:
            torch.Tensor: torch scalar corresponding to signed reprojection
            error in pixels of pupil point (negative if point is inside pupil,
            positive otherwise.)
        """

        origin_inCornea = camera.get_optical_center_inOther(eye.cornea)
        direction_inCamera = \
            camera.compute_direction_toCamera_fromPixels(pupil_point_inPixels)

        T_toCornea_fromCamera = \
            camera.get_transform_toOther_fromSelf(eye.cornea)
        direction_inCornea = \
            T_toCornea_fromCamera.rotation.transform(direction_inCamera)

        return \
            EyeModelErrorFunctions().sampson_error_pupil_inCornea(
                pupil_point_inPixels,
                origin_inCornea,
                direction_inCornea,
                eye
            )

    @staticmethod
    def sampson_error_limbus_inLimbusPlane(
        limbus_point_inPixels,
        origin_limbus_ray_inPlane,
        direction_limbus_ray_inPlane,
        eye
    ):
        """sampson_error_limbus_inLimbusPlane.

        Compute the sampson error for limbus point. The limbus plane is the
        natural coordinate system for this computation, since the bulk of it is
        computing the intersection of a ray with the limbus plane.

        Args:
            limbus_point_inPixels (torch.Tensor): (2,) limbus point in pixel
            coordinates corresponding. This input is required to rescale the 3D
            error in mm back to image coordinates in pixels.

            origin_limbus_ray_inPlane (torch.Tensor): (3,) origin of the limbus
            ray - typically the optical center of a camera - in the coordinate
            system of the limbus plane.

            direction_limbus_ray_inPlane (torch.Tensor): (3,) direction of the
            limbus ray - i.e., the direction of the unprojected ray through the
            image of the input limbus point in the image plane of a camera - in
            the coordinate system of the limbus plane.

            eye (EyeModel): eye model object.

        torch.Tensor: torch scalar corresponding to signed reprojection error
            error in pixels of limbus point (negative if point is inside
            limbus, positive otherwise.)
        """

        # If the input ray does not intersect the cornea, we still intercept it
        # with the limbus plane.
        #
        #          Origin *->___
        #                       \___  Direction
        #                 ______    \___
        #               /        \      \___
        #              /  Cornea  \         \___
        # ____________/____________\____________\_______________
        # Limbus plane        * Limbus center      * Intersection
        #
        #                     <-------------------
        #                      Direction for which
        #                     error decreases brings
        #                      ray closer to cornea
        #
        intersection_inLimbusPlane = \
            eye.limbus_plane.intersect_from_origin_and_direction_inPlane(
                origin_limbus_ray_inPlane,
                direction_limbus_ray_inPlane
            )

        distance_to_center = torch.linalg.norm(intersection_inLimbusPlane)

        unscaled_error = distance_to_center - eye.limbus.radius

        # As with the case of the pupil, we detach to match the C++
        # implementation, which does not consider the derivative of the scale
        # of the Sampson error.
        with torch.enable_grad():  # In case grad is disable in the context.
            d_error_d_limbus_point = \
                core.compute_auto_jacobian_from_tensors(
                    unscaled_error, limbus_point_inPixels
                ).detach()

        return unscaled_error / torch.linalg.norm(d_error_d_limbus_point)

    @staticmethod
    def sampson_error_limbus_inCamera(
        limbus_point_inPixels,
        camera,
        eye
    ):
        """
        Sampson error approximation for the difference between a predicted and
        measured limbus point in an image.

        Args:
            limbus_point_inPixels (torch.Tensor): (2,) limbus point in pixel
            coordinates corresponding. This input is required to rescale the 3D
            error in mm back to image coordinates in pixels.

            camera (PinholeCamera): camera observing pupil point from eye. Note
            that a Polynomial3KCamera is a PinholeCamera.

            eye (EyeModel): eye-model object.

        torch.Tensor: torch scalar corresponding to signed reprojection error
            error in pixels of limbus point (negative if point is inside pupil,
            positive otherwise.)
        """

        origin_inPlane = camera.get_optical_center_inOther(eye.limbus_plane)
        direction_inCamera = \
            camera.compute_direction_toCamera_fromPixels(limbus_point_inPixels)

        T_toPlane_fromCamera = \
            camera.get_transform_toOther_fromSelf(eye.limbus_plane)
        direction_inPlane = \
            T_toPlane_fromCamera.rotation.transform(direction_inCamera)

        return \
            EyeModelErrorFunctions().sampson_error_limbus_inLimbusPlane(
                limbus_point_inPixels,
                origin_inPlane,
                direction_inPlane,
                eye
            )

    @staticmethod
    def sampson_error_glint_inCornea(
        glint_inPixels,
        origin_glint_ray_inCornea,
        direction_glint_ray_inCornea,
        eye,
        led_inCornea
    ):
        """sampson_error_glint_inCornea.

        Sampson error approximation for the difference between the predicted
        and measured glint in an image. The cornea coordinate system is the
        most natural coordinate system in which to carry out this computation,
        as the bulk of it is a reflection on the cornea surface.

        Args:
            glint_inPixels (torch.Tensor): (2,) coordinates of a glint in an
            image.

            origin_glint_ray_inCornea (torch.Tensor): (3,) origin of the
            unprojected ray through the glint. Typically, this is the optical
            center of the camera observing the glint.

            direction_glint_ray_inCornea (torch.Tensor): (3,) direction of the
            unprojected ray through the glint.

            eye (EyeModel): eye-model object.

            led_inCornea (torch.Tensor): (3,) tensor corresponding to the
            coordinates of the LED to which the glint corresponds in the
            coordinate system of the cornea.

        Returns:
            torch.Tensor: (2,) tensor corresponding to the x and y reprojection
            differences in pixels of glint points.
        """

        intersection_inEllipsoid, \
            reflection_direction_inEllipsoid, \
            is_good = \
            eye.cornea.reflect_from_origin_and_direction_inEllipsoid(
                origin_glint_ray_inCornea, direction_glint_ray_inCornea
            )

        # The method above returns the original ray if there is no reflection.
        # In this case the ray will be pushed even more towards the LED and
        # away from the cornea, which is the opposite of what we want. If this
        # occurs, we will use the cornea apex rather than the LED to bring in
        # the ray towards the right direction.
        if is_good is False:
            # This is a "fake" led; it really is the cornea apex.
            led_inCornea = \
                torch.tensor([0.0, 0.0, 1.0]) * eye.cornea.shape_parameters[-1]

        # Create a Ray node, so we can benefit from its functionality. Make it
        # ephemeral so that we don't have to worry about bloatness of the pose
        # graph.
        reflected_ray = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                eye.cornea,
                intersection_inEllipsoid,
                reflection_direction_inEllipsoid
            )
        with core.make_ephemeral(reflected_ray):  # type: ignore
            closest_point_inCornea = \
                reflected_ray.project_to_ray_inParent(led_inCornea)

            unscaled_error_inCornea = closest_point_inCornea - led_inCornea

            # We detach the error to better approximate the C++ implementation.
            with torch.enable_grad():  # In case grad is disabled in the ctxt.
                pinv_error_scale = \
                    core.compute_auto_jacobian_from_tensors(
                        unscaled_error_inCornea, glint_inPixels
                    ).detach()

            return \
                torch.linalg.pinv(pinv_error_scale) @ unscaled_error_inCornea

    @staticmethod
    def sampson_error_glint_inCamera(
        glint_inPixels,
        camera,
        eye,
        led_inCamera
    ):
        """Sampson error for glints from ray in camera coordinates.

        Compute the Sampson approximation for the reprojection error of a glint
        given the point, the eye, and the camera observing the eye.

        Args:
            glint_inPixels (torch.Tensor): (2,) glint in pixels.

            camera (PinholeCamera): camera observing glint. Note that a
            Polynomial3KCamera is a PinholeCamera.

            eye (EyeModel): eye-model object.

            led_inCamera (torch.Tensor): (3,) tensor corresponding to the
            coordinates of the LED that generated the glint, expressed in the
            coordinate system of the camera.

        Returns:
            torch.Tensor: (2,) tensor corresponding to the x and y reprojection
            differences in pixels of glint points.
        """

        origin_inCornea = camera.get_optical_center_inOther(eye.cornea)
        direction_inCamera = \
            camera.compute_direction_toCamera_fromPixels(glint_inPixels)

        T_toCornea_fromCamera = \
            camera.get_transform_toOther_fromSelf(eye.cornea)

        direction_inCornea = \
            T_toCornea_fromCamera.rotation.transform(direction_inCamera)

        led_inCornea = \
            T_toCornea_fromCamera.transform(led_inCamera)

        return \
            EyeModelErrorFunctions().sampson_error_glint_inCornea(
                glint_inPixels,
                origin_inCornea,
                direction_inCornea,
                eye,
                led_inCornea
            )
