"""
File for class StereoPerceptionErrors.

File with definition of class for simulating depth-perception errors in stereo
images generate via pass-through cameras, and evaluating the sensitivity of
these errors to system parameters such as camera and display calibrations and
eye position.
"""

import kiruna
import os
import torch


class StereoPerceptionErrors():
    """
    Class for simulation of stereo-perception errors.

    Class to generate data to characterize stereo-perception errors in the
    visualization of images from the low-light sensors.
    """

    def __init__(
        self,
        baseline_mm=64.7000,
        height_mm=63.6449,
        forward_mm=94.5415,
        display_depth_mm=2_500.0,
        reprojection_depth_mm=25_000.0,
        camera_file_name=os.path.join(
            os.path.join(
                kiruna.device.DEVICE_DIR,
                r"atlas_1.2_device/atlas_1.2_left_camera.json"
            )
        )
    ):
        """
        Initialize basic data for the simulation.

        Args:
            baseline_mm (float, optional): horizontal (in user coordinate
            system) distance in mm between the low-light cameras. Defaults to
            64.7000.

            height_mm (float, optional): height (vertical position in user
            coordinate system) of the cameras in mm. Defaults to 63.6449.

            forward_mm (float, optional): forward position in mm of the cameras
            in user coordinate system. Defaults to 94.5415.

            display_depth_mm (float, optional): depth of the focal plane of the
            display in mm. Defaults to 2_000.0.

            reprojection_depth_mm (float, optional): depth of the reprojection
            plane in mm. Defaults to 25_000.0.
        """
        #######################################################################
        # Data is collected with true sensors and users. Operations are
        # performed with estimated sensors and users.

        ########
        # Truth.

        # Create a user.
        self.user_translation_mm = torch.zeros(3, requires_grad=True)
        self.user_rotation_rad = torch.zeros(3, requires_grad=True)
        translation_toRoot_fromUser = kiruna.core.SE3(self.user_translation_mm)
        rotation_toRoot_fromUser = kiruna.core.SO3(self.user_rotation_rad)
        transform_toRoot_fromUser = \
            kiruna.core.SE3.compose_transforms(
                translation_toRoot_fromUser, rotation_toRoot_fromUser
            )
        self.user = kiruna.user.UserModel(None, transform_toRoot_fromUser)

        # Create pass-through cameras. We root the
        # pass-through cameras on points located on the cameras presumed
        # optical centers, so that we can more easily represent perturbations
        # of those cameras as deltas on top of their original settings.
        self.left_camera_center, \
            self.left_camera = \
            StereoPerceptionErrors.create_camera_at_point(
                self.user,
                baseline_mm / 2,
                height_mm,
                forward_mm,
                parameter_file_name=camera_file_name
            )

        self.right_camera_center, \
            self.right_camera = \
            StereoPerceptionErrors.create_camera_at_point(
                self.user,
                -baseline_mm / 2,
                height_mm,
                forward_mm,
                parameter_file_name=camera_file_name
            )

        # Create reprojection plane.
        self.reprojection_plane = \
            StereoPerceptionErrors.create_plane_at_depth(
                self.user, reprojection_depth_mm
            )

        # Create display planes.
        self.left_display_plane = \
            StereoPerceptionErrors.create_plane_at_depth(
                self.user, display_depth_mm
            )

        self.right_display_plane = \
            StereoPerceptionErrors.create_plane_at_depth(
                self.user, display_depth_mm
            )

        ############
        # Estimates.

        self.hat_user, \
            self.left_eye_translation_perturbation_mm, \
            self.right_eye_translation_perturbation_mm = \
            StereoPerceptionErrors.created_perturbed_user(self.user)

        self.hat_left_camera, \
            self.left_camera_translation_perturbation_mm, \
            self.left_camera_rotation_perturbation_rad, \
            self.left_camera_focal_lengths_perturbation_pix, \
            self.left_camera_principal_point_perturbation_pix, \
            self.left_camera_distortion_center_perturbation, \
            self.left_camera_distortion_coeffs_perturbation = \
            StereoPerceptionErrors.create_perturbed_camera(self.left_camera)

        self.hat_right_camera, \
            self.right_camera_translation_perturbation_mm, \
            self.right_camera_rotation_perturbation_rad, \
            self.right_camera_focal_lengths_perturbation_pix, \
            self.right_camera_principal_point_perturbation_pix, \
            self.right_camera_distortion_center_perturbation, \
            self.right_camera_distortion_coeffs_perturbation = \
            StereoPerceptionErrors.create_perturbed_camera(self.right_camera)

        self.hat_left_display_plane, \
            self.left_display_plane_translation_perturbation_mm, \
            self.left_display_plane_rotation_perturbation_mm = \
            StereoPerceptionErrors.create_perturbed_plane(
                self.left_display_plane
            )

        self.hat_right_display_plane, \
            self.right_display_plane_translation_perturbation_mm, \
            self.right_display_plane_rotation_perturbation_mm = \
            StereoPerceptionErrors.create_perturbed_plane(
                self.right_display_plane
            )

        # Package the data nicely.
        self.left_side_data = list()
        self.left_side_data.append(self.user.eyes[0])
        self.left_side_data.append(self.hat_user.eyes[0])
        self.left_side_data.append(self.left_camera)
        self.left_side_data.append(self.hat_left_camera)
        self.left_side_data.append(self.left_display_plane)
        self.left_side_data.append(self.hat_left_display_plane)

        self.right_side_data = list()
        self.right_side_data.append(self.user.eyes[1])
        self.right_side_data.append(self.hat_user.eyes[1])
        self.right_side_data.append(self.right_camera)
        self.right_side_data.append(self.hat_right_camera)
        self.right_side_data.append(self.right_display_plane)
        self.right_side_data.append(self.hat_right_display_plane)

    @staticmethod
    def create_camera_at_point(
        node,
        x,
        y,
        z,
        parameter_file_name=os.path.join(
            kiruna.device.DEVICE_DIR,
            r"default_device\default_left_camera.json"
        )
    ):
        """
        Create point and camera at x, y, z in coordinate system of node.

        Create a point at x, y, z and an axis-aligned camera with optical
        center coinciding with the point.

        Args:
            node (Node): node in pose graph.

            x (float): x (horizontal) coordinate of point and camera center in
            coordinate system of input node.

            y (float): y (vertical) coordinate of point and camera center in
            coordinate system of input node.

            z (float): x (forward) coordinate of point and camera center in
            coordinate system of input node.

            parameter_file_name (Union[dict | string | None], optional):
            definition of camera parameters. Defaults to None, in which case
            the default camera is used. If a string, this is the path to a
            camera-configuration json file. If a dictionary, this is a
            dictionary with the camera-configuration parameters formatted in
            the same way as in a camera-configuration json file.

            Returns:
            Point: point at x, y, z in the coordinate system of node.

            Camera: axis-aligned camera with optical center on the point.
        """
        translation_toNode_fromPoint = torch.tensor((x, y, z))
        transform_toNode_fromPoint = \
            kiruna.core.SE3(translation_toNode_fromPoint)
        point = kiruna.primitives.Point(node, transform_toNode_fromPoint)
        camera = \
            kiruna.device.Polynomial3KCamera(
                point,
                kiruna.core.SE3.create_identity(),
                parameter_file_name=parameter_file_name
            )

        return point, camera

    @staticmethod
    def create_plane_at_depth(node, depth_mm):
        """
        Create a plane attached to node.

        Create a plane with normal (0, 0, 1) and at the given depth in the
        coordinate system of node.

        Args:
            node (Node): node in pose graph.

            depth_mm (Node): depth in mm of plane along z in coordinate system
            of node.

        Returns:
            Plane: plane at the required depth in node.
        """
        translation_toNode_fromPlane = torch.tensor((0.0, 0.0, depth_mm))
        transform_toNode_fromPlane = \
            kiruna.core.SE3(translation_toNode_fromPlane)

        return kiruna.primitives.Plane(node, transform_toNode_fromPlane)

    @staticmethod
    def create_perturbed_camera(camera):
        """
        Create a perturbed version of the input camera.

        Add zero perturbations to the camera creating a new camera that is
        differentiable with respect to the perturbations. Perturbations are
        applied on the coordinate system of the camera's parent node.

        Args:
            camera (Polynomial3KCamera): polynomial 3K camera to be perturbed.

        Returns:
            Polynomial3KCamera: perturbed version of input camera.

            torch.Tensor: (3,) zero torch.Tensor delta in translation vector in
            mm with respect to which the perturbed camera can be
            differentiated.

            torch.Tensor: (3,) zero torch.Tensor delta in rotation vector in
            rad with respect to which the perturbed camera can be
            differentiated.

            torch.Tensor: (2,) zero torch.Tensor delta in focal lengths vector
            in pixels with respect to which the perturbed camera can be
            differentiated.

            torch.Tensor: (2,) zero torch.Tensor delta in principal point
            vector in pixels with respect to which the perturbed camera can be
            differentiated.

            torch.Tensor: (2,) zero torch.Tensor delta in adimensional
            distortion center for the polynomial 3K camera model with respect
            to which the perturbed camera can be differentiated.

            torch.Tensor: (3,) zero torch.Tensor delta in adimensional
            distortion center with for the polynomial 3K camera model with
            respect to which the perturbed camera can be differentiated.
        """
        # Left camera.
        hat_camera = kiruna.device.Polynomial3KCamera.shallowcopy(camera)
        # Perturbation to camera pose.
        translation_perturbation_mm = torch.zeros(3, requires_grad=True)
        rotation_perturbation_rad = torch.zeros(3, requires_grad=True)
        se3_perturbation = \
            kiruna.core.SE3.compose_transforms(
                kiruna.core.SE3(translation_perturbation_mm),
                kiruna.core.SO3(rotation_perturbation_rad)
            )
        # Apply perturbation.
        hat_camera.update_transform_toParent_fromSelf(se3_perturbation)

        # Perturbation to pinhole intrinsics.
        focal_lengths_perturbation_pix = torch.zeros(2, requires_grad=True)
        principal_point_perturbation_pix = torch.zeros(2, requires_grad=True)
        new_focal_lengths_pix = \
            hat_camera.focal_lengths + focal_lengths_perturbation_pix
        new_principal_point_pix = \
            hat_camera.principal_point + principal_point_perturbation_pix
        # Apply perturbation
        hat_camera.set_pinhole_intrinsics(
            focal_lengths=new_focal_lengths_pix,
            principal_point=new_principal_point_pix
        )

        # Perturbation to distortion coefficients.
        distortion_center_perturbation = torch.zeros(2, requires_grad=True)
        distortion_coeffs_perturbation = torch.zeros(3, requires_grad=True)
        new_distortion_center = \
            hat_camera.distortion_center + distortion_center_perturbation
        new_distortion_coeffs = \
            hat_camera.distortion_coefficients + distortion_coeffs_perturbation
        # Apply perturbation.
        hat_camera.set_distortion_parameters(
            distortion_center=new_distortion_center,
            distortion_coefficients=new_distortion_coeffs
        )

        return \
            hat_camera, \
            translation_perturbation_mm, \
            rotation_perturbation_rad, \
            focal_lengths_perturbation_pix, \
            principal_point_perturbation_pix, \
            distortion_center_perturbation, \
            distortion_coeffs_perturbation

    @staticmethod
    def create_perturbed_plane(plane):
        """
        Create a perturbed version of the plane.

        Add zero perturbation to the input plane, creating a new plane that is
        differentiable with respect to the perturbations. Perturbations are
        applied in the coordinate system of the planes's parent

        Args:
            plane (Plane): plane to be perturbed.

        Returns:
            Plane: perturbed version of input plane.

            torch.Tensor: (3,) zero torch.Tensor corresponding to translation
            of input tensor in mm with respect to which output tensor can be
            differentiated.

            torch.Tensor: (3,) zero torch.Tensor corresponding to rotation axis
            in radians with respect to which the output tensor can be
            differentiated.
        """
        hat_plane = kiruna.primitives.Plane.shallowcopy(plane)
        translation_perturbation_mm = torch.zeros(3, requires_grad=True)
        rotation_perturbation_rad = torch.zeros(3, requires_grad=True)
        se3_perturbation = \
            kiruna.core.SE3.compose_transforms(
                kiruna.core.SE3(translation_perturbation_mm),
                kiruna.core.SO3(rotation_perturbation_rad)
            )
        # Apply perturbation.
        hat_plane.update_transform_toParent_fromSelf(se3_perturbation)

        return \
            hat_plane, \
            translation_perturbation_mm, \
            rotation_perturbation_rad

    @staticmethod
    def created_perturbed_user(user):
        """
        Create a user with pupil positions perturbed by a translation in mm.

        Create a copy of the input user and perturb its eye positions with a
        zero translation with respect to which the eye positions of the
        perturbed user can be differentiated.

        Args:
            user (User): input user.

        Returns:
            User: copy of input user.

            torch.Tensor: (3,) zero torch.Tensor with respect to which the
            left-eye position of the output user can be differentiated.

            torch.Tensor: (3,) zero torch.Tensor with respect to which the
            right-eye position of the output user can be differentiated.
        """
        def perturb_eye(eye):
            """Apply a zero translation to the eye.

            The position of the new eye will be differentiable with respect to
            the translation.

            Args:
                eye (EyeModel): kiruna eye model.

            Outputs:
                torch.Tensor: (3,) zero torch.Tensor with respect to which the
                position of the input eye is differentiable.
            """
            translation_perturbation_mm = torch.zeros(3, requires_grad=True)
            se3_perturbation = kiruna.core.SE3(translation_perturbation_mm)
            eye.update_transform_toParent_fromSelf(se3_perturbation)

            return translation_perturbation_mm

        new_user = kiruna.user.UserModel.shallowcopy(user)
        return_values = list()
        return_values.append(new_user)
        for eye in new_user.eyes:
            return_values.append(perturb_eye(eye))

        return return_values

    def run_rendering_pipeline(self, point_inUser):
        """Generate 3D point as output of rendering pipeline.

        Given an input point in the user coordinate system, generate the
        corresponding point rendered by the system and towards which the user
        gaze should verge.

        Args:
            point_inUser (torch.Tensor): (3,) torch Tensor representing the
            point in the user coordinate system to be rendered by the rendering
            pipeline.

        Returns:
            torch.Tensor: (3,) torch Tensor representing the coordinates of the
            input point as rendered by the rendering pipeline.

            torch.Tensor: (1,) torch Tensor representing the vertical disparity
            in mrad for the rendered output.
        """
        # Verge the user gaze to the true point.
        self.user.verge_at_point_inSelf(point_inUser)
        self.hat_user.verge_at_point_inSelf(point_inUser)

        all_rays = list()
        for side in (self.left_side_data, self.right_side_data):
            eye, \
                hat_eye, \
                camera, \
                hat_camera, \
                display_plane, \
                hat_display_plane = \
                side

            # Generate the pixel with the true camera
            pixel = camera.project_toPixels_fromOther(point_inUser, self.user)

            # Reproject the pixel with the estimated camera.
            hat_origin_inUser, \
                hat_direction_inUser = \
                hat_camera.compute_origin_and_direction_inOther_fromPixels(
                    self.user, pixel
                )

            # Intersect the reprojection ray with the reprojection plane.
            hat_reprojection_inUser = \
                self.reprojection_plane.\
                intersect_from_origin_and_direction_inParent(
                    hat_origin_inUser, hat_direction_inUser
                )  # Parent is the user.

            # Create a ray from the estimated eye to the estimated reprojection
            # point and intersect it with the estimated display plane.
            hat_pupil_inUser, _ = \
                hat_eye.pupil.get_center_and_normal_inOther(self.user)
            hat_gaze_inUser = hat_reprojection_inUser - hat_pupil_inUser
            hat_display_point_inUser = \
                hat_display_plane.intersect_from_origin_and_direction_inParent(
                    hat_pupil_inUser, hat_gaze_inUser
                )  # Parent is the user

            hat_display_point = \
                kiruna.primitives.Point.create_from_coordinates_inParent(
                    self.user, hat_display_point_inUser
                )  # Parent is the user.
            hat_display_point_inHatDisplayPlane = \
                hat_display_point.get_coordinates_inOther(hat_display_plane)

            # Cleanup the pose graph.
            self.user.remove_child(hat_display_point)

            # This is a tricky bit. We think the coordinates of
            # hat_display_point_inHatDisplayPlane are valid for our estimated
            # display plane (hat_display_plane), but when we request for the
            # point to be rendered, we use those estimates and feed them into
            # the true display, which is then observed by a true eye.

            # So we transform hat_display_point_inHatDisplay plane to our user
            # coordinate system under a wrong assumption... On purpose!
            transform_toUser_fromDisplay = \
                display_plane.get_transform_toOther_fromSelf(self.user)
            # The transformation moves points to the user coordinate system
            # from the true display coordinate system, but we feed it with a
            # point in the estimated display coordinate system.
            display_point_inUser = \
                transform_toUser_fromDisplay.transform(
                    hat_display_point_inHatDisplayPlane
                )
            # And this point is observed by a true eye, not the estimated eye.
            pupil_inUser, _ = \
                eye.pupil.get_center_and_normal_inOther(self.user)
            ray = \
                kiruna.primitives.Ray.create_from_origin_and_dir_inParent(
                    self.user,
                    pupil_inUser,
                    display_point_inUser - pupil_inUser
                )
            all_rays.append(ray)

        # Get the best approximation to the vergence point.
        hat_point_inUser = \
            kiruna.primitives.Ray.intersect_rays_inOther(self.user, *all_rays)
        disparity_mrad = \
            kiruna.primitives.Ray.get_disparity_mrad(*all_rays)

        # Cleanup the pose graph.
        for ray in all_rays:
            self.user.remove_child(ray)

        return hat_point_inUser, disparity_mrad
