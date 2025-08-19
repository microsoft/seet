"""scene_model.py

Scene class. A scene is a user and a device registered to the same scene
coordinate system. In general, a scene is the highest-level or root node of the
pose graph.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.device as device
import seet.primitives as primitives
from seet.scene import scene_configs
import seet.user as user
import json
import os
from pathlib import PurePath
import torch


class SceneModel(core.Node):
    """SceneModel.

    An eye-tracking scene consisting of a user and a device in the same
    coordinate system. Typically, the origin of the scene's coordinate system
    will be a point between the apexes of the corneas of the canonical user
    with directions aligned to those of the user's coordinate system.
    """

    def __init__(
        self,
        parent=None,
        transform_toParent_fromSelf=None,
        name="",
        parameter_file_name=os.path.join(
            scene_configs.SCENE_DIR,
            r"default_scene/default_scene.json"
        ),
        requires_grad=False
    ):
        """
        Initialize a scene model.

        Args:
            parameter_file_name (str, optional): name of json file with
            parameters of the scene. Defaults to os.path.join(
            user.USER_MODELS_DIR, r"default_scene/default_scene.json" ).

            requires_grad (bool, optional): whether the parameters of the scene
            are differentiable. Defaults to False.
        """
        self.parameter_file_name = parameter_file_name

        # Specify scene components.
        with core.Node.open(parameter_file_name) as parameter_file_stream:
            scene_parameters = json.load(parameter_file_stream)

        # Name of the scene.
        if "name" in scene_parameters.keys():
            name = scene_parameters["name"]
        else:
            name = ""

        # Scene is the root node of the pose graph.
        super().__init__(
            parent,
            transform_toParent_fromSelf,
            name=name,
            requires_grad=requires_grad
        )

        if requires_grad is None or requires_grad is False:
            requires_grad = scene_parameters.get("requires grad", False)

        #######################################################################
        # Initialize user parameters. Key "user" has to be there. Break it
        # otherwise.
        user_dic = scene_parameters["user"]

        if "extrinsics" in user_dic.keys():
            transform_matrix_toScene_fromUser = \
                torch.tensor(
                    user_dic["extrinsics"],
                    requires_grad=requires_grad
                )
        else:
            transform_matrix_toScene_fromUser = \
                torch.eye(4, requires_grad=requires_grad)

        if requires_grad is None:
            user_requires_grad = user_dic.get("requires grad", False)
        else:
            user_requires_grad = requires_grad

        val = user_dic["parameter file"]
        path = user_dic.get("path", user.USER_DIR)
        if path is None:
            path = user.USER_DIR

        if isinstance(val, str):
            parameter_file_name = os.path.join(path, val)
        else:
            parameter_file_name = val
        self.user = user.UserModel(
            self,  # Adds newly created user as a child of scene.
            core.SE3(transform_matrix_toScene_fromUser),
            parameter_file_name=parameter_file_name,
            requires_grad=user_requires_grad
        )

        #######################################################################
        # Initialize device. Key "device" has to be there. Break it otherwise.
        device_dic = scene_parameters["device"]
        path = device_dic.get("path", device.DEVICE_DIR)
        if path is None:
            path = device.DEVICE_DIR

        if requires_grad is None:
            device_requires_grad = device_dic.get("requires grad", False)
        else:
            device_requires_grad = requires_grad

        val = device_dic["parameter file"]
        if isinstance(val, str):
            parameter_file_name = os.path.join(path, val)
        else:
            parameter_file_name = val

        # Create from calibration file.
        if device_dic.get("device type", None) == "calibration":
            self.device = device.DeviceModel.create_device_from_real_data(
                self,
                core.SE3.create_identity(requires_grad=device_requires_grad),
                parameter_file_name,
                requires_grad=device_requires_grad
            )

        # Create from configuration file.
        else:
            if "extrinsics" in device_dic.keys():
                transform_matrix_toScene_fromDevice = \
                    torch.tensor(
                        device_dic["extrinsics"],
                        requires_grad=device_requires_grad
                    )
            else:
                transform_matrix_toScene_fromDevice = \
                    torch.eye(4, requires_grad=requires_grad)

            self.device = device.DeviceModel(
                self,  # Adds newly created device as a child of scene.
                core.SE3(transform_matrix_toScene_fromDevice),
                parameter_file_name=parameter_file_name,
                # Parameter file has to be there. Break it otherwise.
                requires_grad=device_requires_grad
            )

        # Now that we have a user and a device, we can compute an eye relief
        # plane.
        self._compute_relief_plane()

    def get_kwargs(self):
        """Augment base-class method.

        Returns:
            dict: dictionary with keyword arguments.
        """
        base_kwargs = super().get_kwargs()
        this_kwargs = {"parameter_file_name": self.parameter_file_name}

        return {**base_kwargs, **this_kwargs}

    @staticmethod
    def create_secrets_scene_file(
        left_eye_calibration_blob,
        right_eye_calibration_blob,
        device_calibration_blob,
        user_save_path=None,
        device_save_path=None,
        scene_save_path=None,
        requires_grad=False
    ):
        """
        Create a scene file from user-calibration files for left and right
        eyes as well as a device calibration blob.

        Args:
            left_eye_calibration_blob (dict or str): calibration blob for
            user's left eye. If it is a string, it corresponds to the path and
            file name for the user's left eye calibration blob.

            right_eye_calibration_blob (dict or str): calibration blob for
            user's right eye. If it is a string, it corresponds to the path and
            file name for the user's right calibration blob.

            device_calibration_blob (dict or str): calibration blob for device.
            If it is a string, it corresponds to the path and file name for the
            device calibration blob.

            user_save_path (str, optional): path where converted
            user-calibration blob will be saved. Defaults to None, in which
            case the converted file will not be saved.

            device_save_path (_type_, optional): path where converted
            device-calibration blob will be saved. Defaults to None, in which
            case the converted file will not be saved.

            scene_save_path (_type_, optional): path where converted scene blob
            will be saved. Defaults to None, in which case the converted file
            will not be saved.

            requires_grad (bool, optional): whether the user and device
            parameters are assumed to be differentiable. Defaults to False.
        """
        secrets_scene_dict = dict()
        secrets_scene_dict["name"] = \
            "Created fromm " + \
            left_eye_calibration_blob + \
            " and " + \
            right_eye_calibration_blob + \
            " and " + \
            device_calibration_blob

        user_blob, \
            transform_toLeftCamera_fromLeftEye, \
            transform_toRightCamera_fromRightEye = \
            user.UserModel.create_secrets_user_file(
                left_eye_calibration_blob=left_eye_calibration_blob,
                right_eye_calibration_blob=right_eye_calibration_blob,
                user_save_path=user_save_path,
                requires_grad=requires_grad
            )

        device_blob = \
            device.DeviceModel.create_secrets_device_file(
                device_calibration_blob,
                device_save_path=device_save_path,
                requires_grad=requires_grad
            )

        # Now that we have a user and a device we can appropriately place the
        # user's eyes in the device's coordinate system.
        transform_matrices_toCamera_fromDevice = []
        for i in range(2):
            et_subsystem = device_blob["et subsystems"][i]["subsystem"]
            camera_extrinsics = \
                et_subsystem["cameras"][0]["extrinsics"]

            transform_matrix = torch.tensor(camera_extrinsics)
            if et_subsystem["cameras"][0]["extrinsics type"] == "inverse":
                transform_matrix_toCamera_fromDevice = \
                    torch.linalg.pinv(transform_matrix)
            else:
                transform_matrix_toCamera_fromDevice = transform_matrix

            transform_matrices_toCamera_fromDevice = \
                transform_matrices_toCamera_fromDevice + \
                [transform_matrix_toCamera_fromDevice, ]

        transform_toLeftCamera_fromDevice = \
            core.SE3(transform_matrices_toCamera_fromDevice[0])
        transform_toRightCamera_fromDevice = \
            core.SE3(transform_matrices_toCamera_fromDevice[1])

        transform_toLeftEye_fromDevice = \
            core.SE3.compose_transforms(
                transform_toLeftCamera_fromLeftEye.create_inverse(),
                transform_toLeftCamera_fromDevice
            )
        transform_toRightEye_fromDevice = \
            core.SE3.compose_transforms(
                transform_toRightCamera_fromRightEye.create_inverse(),
                transform_toRightCamera_fromDevice
            )

        # Let's get user coordinates expressed in device coordinates.
        cornea_apexes_inEye = []
        for i in range(2):
            eye_dict = user_blob["eyes"][i]["intrinsics"]  # 0: left; 1: right
            z_cornea_apex_inEye = \
                eye_dict[
                    "distance from rotation center to cornea center"
                ] + \
                eye_dict["cornea"]["radius of curvature"] * \
                eye_dict["cornea"]["radii factors"][2]  # 2 for z.
            cornea_apexes_inEye = \
                cornea_apexes_inEye + \
                [torch.tensor([0.0, 0.0, z_cornea_apex_inEye]), ]

        left_cornea_apex_inDevice = \
            transform_toLeftEye_fromDevice.inverse_transform(
                cornea_apexes_inEye[0]
            )
        right_cornea_apex_inDevice = \
            transform_toRightEye_fromDevice.inverse_transform(
                cornea_apexes_inEye[1]
            )

        user_origin_inDevice = \
            (left_cornea_apex_inDevice + right_cornea_apex_inDevice) / 2

        left_eye_axis_inDevice = \
            transform_toLeftEye_fromDevice.rotation.inverse_transform(
                torch.tensor([0.0, 0.0, 1.0])
            )
        right_eye_axis_inDevice = \
            transform_toRightEye_fromDevice.rotation.inverse_transform(
                torch.tensor([0.0, 0.0, 1.0])
            )
        user_z_axis_inDevice = \
            core.normalize(
                (left_eye_axis_inDevice + right_eye_axis_inDevice) / 2
            )
        user_y_axis_inDevice = \
            core.normalize(
                torch.cross(
                    user_z_axis_inDevice,
                    left_cornea_apex_inDevice - right_cornea_apex_inDevice
                )
            )
        user_x_axis_inDevice = \
            torch.cross(user_y_axis_inDevice, user_z_axis_inDevice)

        # Note that if we build a rotation matrix R from the axes above, given
        # by
        #
        #     [user_x_axis_inDevice.T]
        # R = [user_y_axis_inDevice.T]
        #     [user_z_axis_inDevice.T]
        #
        # we can define a transformation matrix T given by
        #
        # T = [R, -R * user_origin_inDevice],
        #
        # such that
        #
        # T * (user_origin_inDevice) = 0
        # T * (user_origin_inDevice + user_x_axis_inDevice) = [1, 0, 0].T
        # T * (user_origin_inDevice + user_y_axis_inDevice) = [0, 1, 0].T
        # T * (user_origin_inDevice + user_z_axis_inDevice) = [0, 0, 1].T
        #
        # and therefore T is the transformation to-user from-device coordinate
        # systems.

        user_rotation_matrix_toUser_fromDevice = \
            torch.vstack(
                (
                    user_x_axis_inDevice,
                    user_y_axis_inDevice,
                    user_z_axis_inDevice
                )
            )

        transform_toUser_fromDevice = \
            core.SE3(
                user_rotation_matrix_toUser_fromDevice @
                torch.hstack((torch.eye(3), -user_origin_inDevice.view(3, 1)))
            )

        # We can change the default IPD to the value obtained from user
        # calibration.
        IPD_mm = \
            torch.linalg.norm(
                left_cornea_apex_inDevice - right_cornea_apex_inDevice
            )
        flip = 1
        for eye in user_blob["eyes"]:
            eye["extrinsics"] = \
                [flip * IPD_mm.clone().detach().item() / 2, 0.0, 0.0]
            flip = -flip

        # And now we create the secrets scene blob.
        secrets_scene_blob = dict()
        secrets_scene_blob["name"] = \
            "Created from " + user_blob["name"] + " and " + device_blob["name"]
        secrets_scene_blob["user"] = \
            {
                "extrinsics": torch.eye(4).tolist(),
                "requires grad": requires_grad,
                "path": None,
                "parameter file": user_blob
        }
        secrets_scene_blob["device"] = \
            {
                "extrinsics":
                    transform_toUser_fromDevice.transform_matrix.clone().
                    detach().tolist(),
                "requires grad": requires_grad,
                "path": None,
                "parameter file": device_blob
        }

        if scene_save_path is not None:
            base_name = PurePath(scene_save_path).parts[-1]
            scene_json_file_name = base_name + "_scene_model.json"
            file_name = os.path.join(scene_save_path, scene_json_file_name)
            os.makedirs(scene_save_path, exist_ok=True)
            with open(file_name, "w") as file_stream:
                json.dump(secrets_scene_blob, file_stream)

        return secrets_scene_blob

    @classmethod
    def create_scene_from_real_data(
        cls,
        left_eye_calibration_blob,
        right_eye_calibration_blob,
        device_calibration_blob,
        user_save_path=None,
        device_save_path=None,
        scene_save_path=None,
        requires_grad=False
    ):
        """
        Create a scene from user-calibration files for left and right
        eyes as well as a device calibration blob.

        Args:
            left_eye_calibration_blob (dict or str): calibration blob for
            user's left eye. If it is a string, it corresponds to the path and
            file name for the user's left eye calibration blob.

            right_eye_calibration_blob (dict or str): calibration blob for
            user's right eye. If it is a string, it corresponds to the path and
            file name for the user's right calibration blob.

            device_calibration_blob (dict or str): calibration blob for device.
            If it is a string, it corresponds to the path and file name for the
            device calibration blob.

            user_save_path (str, optional): path where converted
            user-calibration blob will be saved. Defaults to None, in which
            case the converted file will not be saved.

            device_save_path (_type_, optional): path where converted
            device-calibration blob will be saved. Defaults to None, in which
            case the converted file will not be saved.

            scene_save_path (_type_, optional): path where converted scene blob
            will be saved. Defaults to None, in which case the converted file
            will not be saved.

            requires_grad (bool, optional): whether the user and device
            parameters are assumed to be differentiable. Defaults to False.
        """
        secrets_scene_blob = \
            SceneModel.create_secrets_scene_file(
                left_eye_calibration_blob,
                right_eye_calibration_blob,
                device_calibration_blob,
                user_save_path=user_save_path,
                device_save_path=device_save_path,
                scene_save_path=scene_save_path,
                requires_grad=requires_grad
            )

        return cls(parameter_file_name=secrets_scene_blob)  # type: ignore

    def _compute_relief_plane(self):
        """_compute_relief_plane.

        Computes a plane with respect to which we compute the user's eye
        relief. The relief plane depends on the nominal orientation of the
        user's gaze, but it is attached to the coordinate system of the device.
        """
        for eye, subsystem in zip(self.user.eyes, self.device.subsystems):
            gaze_inSubsystem = eye.get_gaze_direction_inOther(subsystem)
            apex_inSubsystem = eye.get_cornea_apex_inOther(subsystem)
            relief_plane = \
                primitives.Plane.create_from_origin_and_normal_inParent(
                    subsystem, apex_inSubsystem, gaze_inSubsystem
                )
            subsystem.set_eye_relief_plane(relief_plane)

    def compute_working_distance(self, subsystem_index=0, camera_index=0):
        """compute_working_distance.

        Compute the working distance between the eye associated with a
        subsystem and a camera in that subsystem. Working distance is defined
        as the distance between the camera optical center and a plane that
        intersects the cornea apex and that is orthogonal to the camera optical
        axis.

        Args:
            subsystem_index (int, optional): Subsystem associated with the eye
            with respect to which we wish to compute a working distance.
            Defaults to 0.

            camera_index (int, optional): Camera within the subsystem with
            respect to which we wish to compute a working distance. Defaults to
            0.
        """
        camera = self.device.subsystems[subsystem_index].cameras[camera_index]
        eye = self.user.eyes[subsystem_index]

        cornea_apex_inCamera = eye.get_cornea_apex_inOther(camera)

        return cornea_apex_inCamera[-1]

    def apply_occluding_contour_inOther(
        self,
        other_node,
        points_inOther,
        subsystem_index=0,
        reference_point_inOther=None
    ):
        """
        Remove points behind the occluding contour generated by the
        reference point and the eye associated with the indicated subsystem. If
        the reference point is not provided it is assumed to be the optical
        center of the first camera of the subsystem.

        Args:
            other_node (Node): node in the pose graph in which the coordinates
            of the points in the input list points_inOther are provided.

            points_inOther (list of torch.Tensors): list of torch Tensors
            holding coordinates of points whose visibility/occlusion is to be
            tested, in the coordinate system of other_node.

            subsystem_index (int, optional): index of subsystem whose
            associated eye generates the occluding contour. Defaults to 0.

            reference_point_inOther (torch.Tensor, optional): (3,) torch Tensor
            holding coordinates of reference point that generates the occluding
            contour, in the coordinate system of other_node. Defaults to None,
            in which case the optical center of the first camera of the
            subsystem is used.

        Returns:
            list of torch.Tensor: a version of points_inOther where occluded
            points are replaced with None or removed altogether, depending on
            the value of set_to_none.
        """
        cornea = self.user.eyes[subsystem_index].cornea
        transform_toCornea_fromOther = \
            other_node.get_transform_toOther_fromSelf(cornea)
        if reference_point_inOther is None:
            subsystem = self.device.subsystems[subsystem_index]
            camera = subsystem.cameras[0]
            reference_point_inCornea = \
                camera.get_optical_center_inOther(cornea)
        else:
            reference_point_inCornea = \
                transform_toCornea_fromOther.transform(reference_point_inOther)

        polar_plane_inCornea = \
            cornea.compute_polar_plane_to_point_inEllipsoid(
                reference_point_inCornea
            )

        # This is a new node in the pose graph. It has a coordinate system
        # attached to it, but it is itself coordinate free.
        polar_plane = \
            primitives.Plane.create_from_homogeneous_coordinates_inParent(
                cornea, polar_plane_inCornea
            )

        # We use the signed distance of the reference point as a reference.
        reference_sign = \
            polar_plane.compute_signed_distance_to_point_inParent(
                reference_point_inCornea
            )

        # We then check the signed distance of each point in the list.
        for i in range(len(points_inOther)):
            if points_inOther[i] is not None:
                other_sign = \
                    polar_plane.compute_signed_distance_to_point_inOther(
                        points_inOther[i], other_node
                    )

                if other_sign * reference_sign <= 0:
                    points_inOther[i] = None

        return points_inOther

    def generate_glints_inCornea(
        self,
        subsystem_index=0,
        camera_index=0,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True
    ):
        """
        Generate 3D glint locations in the coordinate system of the cornea
        as seen from a given camera in a given subsystem of the device.

        Args:
            subsystem_index (int, optional): index of subsystem. Defaults to 0.

            camera_index (int, optional): index of camera. Defaults to 0.

            apply_device_occluder (bool, optional): whether to omit glints
            occluded by the occluder. Defaults to True.

            apply_eyelids_occlusion (bool, optional): if True, apply occlusion
            by eyelids. Defaults to True.
        """
        subsystem = self.device.subsystems[subsystem_index]
        if subsystem.occluder is None:
            apply_device_occluder = False

        # TODO: Figure out how to handle a single camera looking at both eyes.
        eye_index = subsystem.eye_index
        eye = self.user.eyes[eye_index]

        cornea = eye.cornea
        transform_toSubsystem_fromCornea = \
            cornea.get_transform_toOther_fromSelf(subsystem)

        # We seek the point at which each LED reflects towards the camera's
        # optical center.
        camera = subsystem.cameras[camera_index]
        optical_center_inCornea = camera.get_optical_center_inOther(cornea)

        led_set = subsystem.led_set
        transform_toCornea_fromLEDs = \
            led_set.get_transform_toOther_fromSelf(cornea)

        glints_inCornea = []
        glints_inSubsystem = []
        leds_inSubsystem = []
        for led_inLEDs in led_set.coordinates.T:
            led_inCornea = transform_toCornea_fromLEDs.transform(led_inLEDs)
            single_glint_inCornea = \
                eye.generate_glint_inCornea(
                    led_inCornea,
                    optical_center_inCornea,
                    apply_eyelids_occlusion=apply_eyelids_occlusion
                )

            glints_inCornea = glints_inCornea + [single_glint_inCornea, ]

            # These are used only if we apply the device occluder.
            if apply_device_occluder:
                if single_glint_inCornea is not None:
                    single_glint_inSubsystem = \
                        transform_toSubsystem_fromCornea.transform(
                            single_glint_inCornea
                        )

                    single_led_inSubsystem = \
                        transform_toSubsystem_fromCornea.transform(
                            led_inCornea
                        )
                else:
                    single_glint_inSubsystem = None
                    single_led_inSubsystem = None

                glints_inSubsystem = \
                    glints_inSubsystem + [single_glint_inSubsystem, ]
                leds_inSubsystem = \
                    leds_inSubsystem + [single_led_inSubsystem, ]

        # If requested, check for occlusions by the occluder.
        if apply_device_occluder:
            # First from glint to camera.
            optical_center_inSubsystem = \
                camera.get_optical_center_inOther(subsystem)
            visible_glints_inSubsystem = \
                subsystem.apply_occluder_inSubsystem(
                    glints_inSubsystem, optical_center_inSubsystem)

            # Then, from glint to LEDs. This is trickier, as the reference
            # point keeps changing and there's a pairing of glints and LEDs.
            # For the camera, the optical center was the single reference
            # point. Now, we need a loop.
            N = len(visible_glints_inSubsystem)
            for i, single_led_inSubsystem, single_glint_inSubsystem in \
                    zip(
                        range(N),
                        leds_inSubsystem,
                        visible_glints_inSubsystem
                    ):
                # The occluder code expects a list of points.
                list_with_single_glint_inSubsystem = \
                    subsystem.apply_occluder_inSubsystem(
                        [single_glint_inSubsystem, ], single_led_inSubsystem
                    )
                if list_with_single_glint_inSubsystem[0] is None:
                    glints_inCornea[i] = None

        return glints_inCornea

    def generate_glints_inOther(
        self,
        other_node=None,
        subsystem_index=0,
        camera_index=0,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True
    ):
        """
        Generate 3D glint locations in the coordinate system of an arbitrary
        node in the pose graph as seen from a given camera in a given subsystem
        of the device.

        Args:
            other_node (Node, optional): node in the pose graph in which to
            generate the 3D coordinates of the glints. if None, use the root of
            the scene.

            subsystem_index (int, optional): index of subsystem. Defaults to 0.

            camera_index (int, optional): index of camera. Defaults to 0.

            apply_device_occluder (bool, optional): whether to omit glints
            occluded by the occluder. Defaults to True.

            apply_eyelids_occlusion (bool, optional): if True, glints below the
            eyelids are marked as None. Defaults to True.

        Returns:
            list of torch.Tensors: list with either (3,) tensors holding the
            coordinates of the glints generated by each LED of the input
            subsystem in the coordinate system of other_node, or None if the
            corresponding LED does not produce a glint.
        """
        glints_inCornea = \
            self.generate_glints_inCornea(
                subsystem_index=subsystem_index,
                camera_index=camera_index,
                apply_device_occluder=apply_device_occluder,
                apply_eyelids_occlusion=apply_eyelids_occlusion
            )

        subsystem = self.device.subsystems[subsystem_index]

        # TODO: Figure out how to handle a single camera looking at both eyes.
        eye_index = subsystem.eye_index
        eye = self.user.eyes[eye_index]

        if other_node is None:
            other_node = self.get_root()

        cornea = eye.cornea
        transform_toOther_fromCornea = \
            cornea.get_transform_toOther_fromSelf(other_node)

        glints_inOther = []
        for single_glint_inCornea in glints_inCornea:
            if single_glint_inCornea is None:
                glints_inOther = glints_inOther + [None, ]
            else:
                glints_inOther = \
                    glints_inOther + \
                    [
                        transform_toOther_fromCornea.transform(
                            single_glint_inCornea
                        ),
                    ]

        return glints_inOther

    def generate_refracted_pupil_inCornea(
        self,
        subsystem_index=0,
        camera_index=0,
        num_points=30,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True
    ):
        """
        Generate pupil in the coordinate system of the user's cornea as
        observed by a given camera in a given subsystem of the device. The
        output is a set of points with coordinates in the coordinate system of
        the user's cornea.

        Args:
            subsystem_index (int, optional): index of device subsystem from
            which to generate the refracted pupil. Defaults to 0.

            camera_index (int, optional): index of the camera in the subsystem
            from which to generate the refracted pupil. Defaults to 0.

            num_points (int, optional): number of points to sample along the
            pupil. Defaults to 30.

            apply_device_occluder (bool, optional): if True, occlude pupil
            points using the device's occluder, if one is present. Defaults to
            True.

            apply_eyelids_occlusion (bool, optional): if True, occlude pupil
            points hidden by the eyelids. Defaults to True.

        Returns:
            list of Union(None | torch.Tensor): list of refracted pupil points
            in the cornea coordinate system, for visible points, or None for
            occluded or otherwise invisible points

            list of torch.Tensor: list of angles in radians corresponding to
            the parameterizing angle of each point, visible or not.
        """
        subsystem = self.device.subsystems[subsystem_index]

        # TODO: Figure out how to handle a single camera looking at both eyes.
        eye_index = subsystem.eye_index
        eye = self.user.eyes[eye_index]

        cornea = eye.cornea
        transform_toSubsystem_fromCornea = \
            cornea.get_transform_toOther_fromSelf(subsystem)
        camera = subsystem.cameras[camera_index]
        destination_inCornea = camera.get_optical_center_inOther(cornea)
        destination_inSubsystem = camera.get_optical_center_inOther(subsystem)

        refractions_inCornea, angles_rad = \
            eye.generate_refracted_pupil_inCornea(
                destination_inCornea,
                num_points=num_points,
                apply_eyelids_occlusion=apply_eyelids_occlusion
            )

        # Filter out refractions occluded by occluder, if required.
        if subsystem.occluder is not None and apply_device_occluder:
            for i in range(len(refractions_inCornea)):
                single_refraction_inCornea = refractions_inCornea[i]
                if single_refraction_inCornea is not None:
                    refraction_inSubsystem = \
                        transform_toSubsystem_fromCornea.transform(
                            single_refraction_inCornea
                        )

                    if subsystem.occluder.is_ray_occluded_inParent(
                        refraction_inSubsystem,
                        destination_inSubsystem - refraction_inSubsystem
                    ):
                        refractions_inCornea[i] = None

        return refractions_inCornea, angles_rad

    def generate_refracted_pupil_inOther(
        self,
        other_node=None,
        subsystem_index=0,
        camera_index=0,
        num_points=30,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True
    ):
        """
        Generate 3D locations of the refraction points on the cornea in the
        coordinate system of other_node of the pupil points as seen from a
        given camera in a given subsystem of the device.

        Args:
            other_node (Node, optional): node in the pose graph in which to
            generate the 3D coordinates of the pupil refraction points. if
            None, use the root of the scene.

            subsystem_index (int, optional): index of subsystem. Defaults to 0.

            camera_index (int, optional): index of camera. Defaults to 0.

            num_points (int, optional): number of points sampled along the
            pupil's contour (prior to occlusion). Defaults to 30.

            apply_device_occluder (bool, optional): whether to omit glints
            occluded by the occluder. Defaults to True.

            apply_eyelids_occlusion (bool, optional): if True, refraction
            points behind the eyelids are None. Defaults to True.

        Returns:
            list of torch.Tensors: list of (3,) torch Tensors with coordinates
            of the pupil points the surface of the cornea.

            list of torch.Tensors: list of (1,) torch Tensor with angles in
            radian parameterizing the pupil points along the pupil.
        """
        refracted_pupil_inCornea, angles_rad = \
            self.generate_refracted_pupil_inCornea(
                subsystem_index=subsystem_index,
                camera_index=camera_index,
                num_points=num_points,
                apply_device_occluder=apply_device_occluder,
                apply_eyelids_occlusion=apply_eyelids_occlusion
            )

        subsystem = self.device.subsystems[subsystem_index]

        # TODO: Figure out how to handle a single camera looking at both eyes.
        eye_index = subsystem.eye_index
        eye = self.user.eyes[eye_index]

        if other_node is None:
            other_node = self.get_root()

        cornea = eye.cornea
        transform_toOther_fromCornea = \
            cornea.get_transform_toOther_fromSelf(other_node)

        refracted_pupil_inOther = []
        for single_point in refracted_pupil_inCornea:
            if single_point is None:
                pt = None
            else:
                pt = transform_toOther_fromCornea.transform(single_point)

            refracted_pupil_inOther = refracted_pupil_inOther + [pt, ]

        return refracted_pupil_inOther, angles_rad

    def generate_occluding_contour_inCornea(
        self,
        subsystem_index=0,
        camera_index=0,
        num_points=30,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True
    ):
        """
        Generate occluding contour of the user's cornea as observed from a
        given camera in a given subsystem of the device. The output is list of
        torch tensors corresponding to the coordinates of the occluding contour
        in the coordinate system of the cornea.

        Args:
            other_node (Node): node in the pose graph in which to generate the
            occluding contour.

            subsystem_index (int, optional): index of subsystem. Defaults to 0.

            camera_index (int, optional): index of camera. Defaults to 0.

            num_points (int, optional): number of points sampled along the
            pupil's contour (prior to occlusion). Defaults to 30.

            apply_device_occluder (bool, optional): whether to omit points
            occluded by the occluder. Defaults to True.

            apply_eyelids_occlusion (bool, optional): if True, points behind
            the eyelids are set to None. Defaults to True.

        Returns:
            list of torch.Tensors: list of (3,) torch Tensors with coordinates
            of the visible points of the occluding contour on the surface of
            the cornea.
        """
        subsystem = self.device.subsystems[subsystem_index]
        camera = subsystem.cameras[camera_index]

        # TODO: Figure out how to handle a situation in which a subsystem is
        # trained on more than one both eyes.
        eye = self.user.eyes[subsystem.eye_index]

        optical_center_inCornea = camera.get_optical_center_inOther(eye.cornea)

        occluding_contour_inCornea = \
            eye.generate_occluding_contour_inCornea(
                optical_center_inCornea,
                num_points=num_points,
                apply_eyelids_occlusion=apply_eyelids_occlusion
            )

        if apply_device_occluder:
            transform_toSubsystem_fromCornea = \
                eye.cornea.get_transform_toOther_fromSelf(subsystem)
            occluding_contour_inSubsystem = []
            for pt in occluding_contour_inCornea:
                if pt is None:
                    single_occluding_contour_inSubsystem = None
                else:
                    single_occluding_contour_inSubsystem = \
                        transform_toSubsystem_fromCornea.transform(pt)

                occluding_contour_inSubsystem = \
                    occluding_contour_inSubsystem + \
                    [single_occluding_contour_inSubsystem, ]

            optical_center_inSubsystem = \
                camera.get_optical_center_inOther(subsystem)

            visible_contour_inSubsystem = \
                subsystem.apply_occluder_inSubsystem(
                    occluding_contour_inSubsystem, optical_center_inSubsystem
                )

            for i in range(len(visible_contour_inSubsystem)):
                if visible_contour_inSubsystem[i] is None:
                    occluding_contour_inCornea[i] = None

        return occluding_contour_inCornea

    def generate_occluding_contour_inOther(
        self,
        other_node=None,
        subsystem_index=0,
        camera_index=0,
        num_points=30,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True
    ):
        """
        Generate occluding contour in coordinate system of other_node.

        Args:
            other_node (Node, optional): Node in which to represent the
            coordinates of the occluding contour. Defaults to None, in which
            case the root node of the scene's pose graph is used.

            subsystem_index (int, optional): index of the device subsystem in
            which to generate the occluding contour. Defaults to 0.

            camera_index (int, optional): index of the camera in the subsystem
            from which viewpoint to generate the occluding contour. Defaults to
            0.

            num_points (int, optional): number of points to sample along the
            occluding contour. Defaults to 30.

            apply_device_occluder (bool, optional): if True, set points
            occluded by the subsystem's occluder to None. Defaults to True.

            apply_eyelids_occlusion (bool, optional): if True, points behind
            the eyelids are set to None. Defaults to True.

        Returns:
            list of torch.Tensors: list whose entries are either None (for
            points occluded or behind the eyelids) or (3,) torch tensors with
            coordinates of points along the occluding contour represented in
            the coordinate system of other_node.
        """
        occluding_contour_inCornea = \
            self.generate_occluding_contour_inCornea(
                subsystem_index=subsystem_index,
                camera_index=camera_index,
                num_points=num_points,
                apply_device_occluder=apply_device_occluder,
                apply_eyelids_occlusion=apply_eyelids_occlusion
            )

        if other_node is None:
            other_node = self.get_root()

        subsystem = self.device.subsystems[subsystem_index]
        eye = self.user.eyes[subsystem.eye_index]
        transform_toOther_fromCornea = \
            eye.cornea.get_transform_toOther_fromSelf(other_node)

        occluding_contour_inOther = []
        for pt in occluding_contour_inCornea:
            if pt is None:
                single_occluding_contour_inSubsystem = None
            else:
                single_occluding_contour_inSubsystem = \
                    transform_toOther_fromCornea.transform(pt)

            occluding_contour_inOther = \
                occluding_contour_inOther + \
                [single_occluding_contour_inSubsystem, ]

        return occluding_contour_inOther

    def generate_limbus_inOther(
        self,
        other_node=None,
        subsystem_index=0,
        camera_index=0,
        num_points=30,
        apply_device_occluder=True,
        apply_eyelids_occlusion=True,
        apply_corneal_bulge_occlusion=True
    ):
        """
        Generate limbus points in the coordinate system of the user's cornea.

        Args:
            other_node (Node, optional): Node in which to represent the
            coordinates of the occluding contour. Defaults to None, in which
            case the root node of the scene's pose graph is used.

            subsystem_index (int, optional): index of subsystem. Defaults to 0.

            camera_index (int, optional): index of camera. Defaults to 0.

            num_points (int, optional): number of points sampled along the
            limbus' contour (prior to occlusion). Defaults to 30.

            apply_device_occluder (bool, optional): whether to omit limbus
            points occluded by the occluder. Defaults to True.

            apply_eyelids_occlusion (bool, optional): if True, limbus points
            behind the eyelids are None. Defaults to True.

            apply_corneal_bulge_occlusion (bool, optional): If True, limbus
            points behind the corneal bulge are set to None. Defaults to True.

        Returns:
            list of Union(None | torch.Tensor): list with either the
            coordinates of a limbus point represented as a (3,) torch tensor,
            if the limbus point is visible, or None, if the limbus point is not
            visible.

            list of torch.Tensors: list with angles in radians parameterizing
            limbus point along the limbus contour, regardless of whether the
            point is visible.
        """
        if other_node is None:
            other_node = self.get_root()

        eye = self.user.eyes[subsystem_index]
        limbus_inOther, angles_rad = \
            eye.generate_limbus_inOther(
                other_node=other_node,
                num_points=num_points,
                apply_eyelids_occlusion=apply_eyelids_occlusion
            )

        subsystem = self.device.subsystems[subsystem_index]
        camera = subsystem.cameras[camera_index]
        camera_center_inOther = camera.get_optical_center_inOther(other_node)
        if apply_device_occluder:
            limbus_inOther = \
                subsystem.apply_occluder_inOther(
                    other_node,
                    limbus_inOther,
                    reference_point_inOther=camera_center_inOther
                )

        if apply_corneal_bulge_occlusion:
            limbus_inOther = \
                self.apply_occluding_contour_inOther(
                    other_node,
                    limbus_inOther,
                    subsystem_index=subsystem_index,
                    reference_point_inOther=camera_center_inOther
                )

        return limbus_inOther, angles_rad
