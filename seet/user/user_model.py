"""user_model.py

User model class.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import json
import seet.core as core
import seet.primitives as primitives
from seet.user import user_configs
from seet.user import eye_model
import os
from pathlib import PurePath
import torch


class UserModel(core.Node):
    """UserModel.

    User model class. A user has a coordinate system and two rotation
    centers; to each rotation center, and eye is attached. The origin of the
    user's coordinate system is a point between the apexes of the user's pupils
    when the user is at a nominal gaze. "Nominal gaze" is defined as no
    horizontal rotation and some user-dependent vertical rotation. For a
    default user, that rotation is seven degrees downwards, per the outcome of
    comfort studies.
    """

    def __init__(
        self,
        scene,
        transform_toScene_fromUser,
        name="",
        parameter_file_name=os.path.join(
            user_configs.USER_DIR,
            "default_user/default_user.json"
        ),
        requires_grad=False
    ):
        """
        Initialize user.

        Args:
            scene (Node): parent node of the user. Typically, this will be a
            scene object.

            transform_toScene_fromUser (SE3): transformation from coordinate
            system of device to that of scene.

            name (str, optional): name of object.

            parameter_file_name (str, optional): name of user; not as in "John
            Smith". Defaults to os.path.join( user_configs.USER_MODELS_DIR,
            "default_user/default_user.json" ).

            requires_grad (bool): flag indicating if parameters of user model
            should be differentiable.
        """

        self.parameter_file_name = parameter_file_name

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            user_parameters = json.load(parameter_file_stream)

        # Codename of user
        if "name" in user_parameters.keys():
            name = user_parameters["name"]
        else:
            name = ""

        super().__init__(
            scene,
            transform_toScene_fromUser,
            name=name,
            requires_grad=requires_grad
        )

        # As a convenience we give the user access to the eyes. But it is
        # important to remember that the eye's parent is not the user, but the
        # rotation center.
        self.eyes = []

        local_requires_grad = None
        self.rotation_centers = []
        for eye_dict in user_parameters["eyes"]:
            if requires_grad is None:
                if "requires grad" in eye_dict.keys() and \
                        eye_dict["requires grad"]:
                    local_requires_grad = True
                else:
                    local_requires_grad = False
            else:
                local_requires_grad = requires_grad

            val = eye_dict["intrinsics"]
            # Create a rotation center. The final location of the rotation
            # center will depend on the nominal gaze, so we start with a
            # standard location in the center of the user's face, and fix
            # it later.
            rotation_center = \
                primitives.Point(
                    self, core.SE3.create_identity()  # No need for gradient.
                )
            if isinstance(val, int):
                # We mirror a previous eye.

                # Mirror the rotation center. This overrides the rotation
                # center created above.
                rotation_center = \
                    primitives.Point.mirror(self.rotation_centers[val])

                eye = rotation_center.children[0]
            else:
                if isinstance(val, str):
                    USER_DIR = eye_dict.get("path")
                    if USER_DIR is None:
                        USER_DIR = user_configs.USER_DIR
                    eye_file_name = os.path.join(USER_DIR, val)
                else:
                    eye_file_name = val

                # We create the eye in the user's coordinate system, with a
                # frontal gaze (horizontal and vertical rotation angles equal
                # to zero in the user's coordinate system). We then rotate the
                # gaze to the direction indicated in the file, and translate
                # the eye so that the cornea apex matches the desired value,
                # and anchor the eye at its new rotation center.

                eye = \
                    eye_model.EyeModel(
                        rotation_center,
                        core.SE3.create_identity(),  # Initially, forward gaze.
                        parameter_file_name=eye_file_name,
                        requires_grad=local_requires_grad
                    )

                gaze_angles_deg = \
                    torch.tensor(
                        eye_dict["nominal gaze angles"],
                        requires_grad=local_requires_grad
                    )

                # Then rotate to nominal gaze. Parent is the rotation center.
                eye.rotate_from_gaze_angles_inParent(
                    gaze_angles_deg, move_eyelids=False
                )

                # Now, translate the rotation center so that the cornea apex is
                # in the correct position.
                current_cornea_apex_inUser = eye.get_cornea_apex_inOther(self)
                desired_cornea_apex_inUser = \
                    torch.tensor(
                        eye_dict["extrinsics"],
                        requires_grad=local_requires_grad
                    )
                translation_fromParent_toParent = \
                    desired_cornea_apex_inUser - current_cornea_apex_inUser

                # This carries the whole eye with it. The parent of the
                # rotation center is the user, but the eye is attached to the
                # rotation center.
                rotation_center.translate_inParent(
                    translation_fromParent_toParent
                )

            self.rotation_centers = self.rotation_centers + [rotation_center, ]
            self.eyes = self.eyes + [eye, ]

    def get_kwargs(self):
        """Augment base-class method.

        Returns:
            dict: dictionary of keyword arguments.
        """

        base_kwargs = super().get_kwargs()
        this_kwargs = {"parameter_file_name": self.parameter_file_name}

        return {**base_kwargs, **this_kwargs}

    @staticmethod
    def create_secrets_user_file(
        left_eye_calibration_blob,
        right_eye_calibration_blob,
        user_save_path=None,
        requires_grad=False
    ):
        """
        Create a user file from left and right user-calibration blobs. The
        user is floating, as it is not attached to a scene.

        Args:
            left_eye_calibration_blob (str): path and name of user-calibration
            blob for left eye.

            right_eye_calibration_blob (str): path and name of user-calibration
            blob for right eye.

            user_save_path (str, optional): path where converted
            user-calibration blob will be saved. Defaults to None, in which
            case the converted file will not be saved.

            requires_grad (bool, optional): whether parameters of the user are
            differentiable. Defaults to False.

        Returns:
            dict: user-model with parameters defined by input calibration
            blob(s).

            SE3: SE(3) element that transforms from the left eye coordinate
            system to the left camera coordinate system.

            SE3: SE(3) element that transforms from the right eye coordinate
            system to the right camera coordinate system.
        """

        secrets_user_dict = dict()
        secrets_user_dict["name"] = \
            "Created from " + \
            left_eye_calibration_blob + \
            " and " + right_eye_calibration_blob

        left_eye_dict, transform_toLeftCamera_fromLeftEye = \
            eye_model.EyeModel.create_secrets_eye_file(
                left_eye_calibration_blob, requires_grad=requires_grad
            )
        right_eye_dict, transform_toRightCamera_fromRightEye = \
            eye_model.EyeModel.create_secrets_eye_file(
                right_eye_calibration_blob, requires_grad=requires_grad
            )

        # Use default user parameters to fill in gaps.
        parameter_file_name = \
            os.path.join(
                user_configs.USER_DIR, r"default_user/default_user.json"
            )
        with open(parameter_file_name, "r") as parameter_file_stream:
            parameters_dict = json.load(parameter_file_stream)

        gaze_angles = parameters_dict["eyes"][0]["nominal gaze angles"]
        x, y, z = parameters_dict["eyes"][0]["extrinsics"]

        secrets_user_dict["eyes"] = \
            [
                {
                    "path": None,
                    "intrinsics": left_eye_dict,
                    "requires grad": requires_grad,
                    "nominal gaze angles": gaze_angles,
                    "extrinsics": [x, y, z]
                },
                {
                    "path": None,
                    "intrinsics": right_eye_dict,
                    "requires grad": requires_grad,
                    "nominal gaze angles": gaze_angles,
                    "extrinsics": [-x, y, z]
                },
        ]

        # Save SECRETS device configuration file, if required.
        if user_save_path is not None:
            base_name = PurePath(user_save_path).parts[-1]
            device_json_file_name = base_name + "_device_model.json"
            file_name = os.path.join(user_save_path, device_json_file_name)
            os.makedirs(user_save_path, exist_ok=True)
            with open(file_name, "w") as file_stream:
                json.dump(secrets_user_dict, file_stream)

        return \
            secrets_user_dict, \
            transform_toLeftCamera_fromLeftEye, \
            transform_toRightCamera_fromRightEye

    @classmethod
    def create_user_from_real_data(
        cls,
        scene,
        transform_toScene_fromUser,
        left_eye_blob,
        right_eye_blob,
        user_save_path=None,
        requires_grad=False
    ):
        """
        Create a user for at least one of a left or a right user-calibration
        blob.

        Args:
            left_eye_blob (str): path and name of user-calibration blob for
            left eye.

            right_eye_blob (str): path and name of user-calibration blob for
            right eye.

            user_save_path (str, optional): path where converted
            user-calibration blob will be saved. Defaults to None, in which
            case the converted file will not be saved.

            requires_grad (bool, optional): whether parameters of the user are
            differentiable. Defaults to False.

        Returns:
            UserModel: user model with parameters defined by input calibration
            blob(s).
        """

        secrets_user_blob, _, _ = \
            UserModel.create_secrets_user_file(
                left_eye_blob,
                right_eye_blob,
                user_save_path=user_save_path,
                requires_grad=requires_grad
            )

        return \
            cls(
                scene,
                transform_toScene_fromUser,
                parameter_file_name=secrets_user_blob
            )

    def compute_IPD(self):
        """compute_IPD.

        Compute the user's IPD
        """

        pupils_inUser = []
        for i in range(2):
            transform_toUser_fromPupil = \
                self.eyes[i].pupil.get_transform_toOther_fromSelf(self)
            pupils_inUser = \
                pupils_inUser + \
                [transform_toUser_fromPupil.transform(torch.zeros(3)), ]

        return torch.linalg.norm(pupils_inUser[0] - pupils_inUser[1])

    def verge_at_point_inSelf(self, point_inSelf):
        """
        Rotate the user's eyes so that they verge at the given point.

        Rotate the user's eyes so that they verge at the given point.

        Args:
            point_inSelf (torch.tensor): (3,) torch tensor holding coordinates
            of a point in the user's coordinate system.
        """

        # Find angles between current gaze directions and desired gaze
        # directions.
        for eye in self.eyes:
            transform_toSelf_fromEyeParent = \
                eye.parent.get_transform_toOther_fromSelf(self)
            point_inEyeParent = \
                transform_toSelf_fromEyeParent.inverse_transform(point_inSelf)
            eye.direct_at_point_inParent(point_inEyeParent)

    def get_vergence_point_inSelf(self):
        """
        Get the vergence point of the user's eyes in the user coord. system.

        If the rays from the eyes do not intersect, the point closest to both
        rays is provided.

        Returns:
            torch.Tensor: (3,) tensor corresponding to coordinates of vergence
            point in the user's coordinate system.
        """

        left_origin_inUser = self.eyes[0].parent.get_coordinates_inOther(self)
        left_gaze_inUser = self.eyes[0].get_gaze_direction_inOther(self)
        left_ray = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self, left_origin_inUser, left_gaze_inUser
            )

        right_origin_inUser = self.eyes[1].parent.get_coordinates_inOther(self)
        right_gaze_inUser = self.eyes[1].get_gaze_direction_inOther(self)
        right_ray = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self, right_origin_inUser, right_gaze_inUser
            )

        vergence_point_inUser = \
            primitives.Ray.intersect_rays_inOther(self, left_ray, right_ray)

        # The rays get attached to the pose graph, and are not deleted unless
        # we do so explicitly.
        self.remove_child(left_ray)
        self.remove_child(right_ray)

        return vergence_point_inUser

    def get_disparity_mrad(self):
        """Compute the disparity between gaze directions.

        A vergence point is computed as the least-squares approximation to the
        point closest to the gaze rays departing from each eye. If the vergence
        point is exact, i.e., the gaze rays intersect, the disparity is zero.
        Otherwise we compute the disparity as twice the angle between a gaze
        ray and the ray from the rotation center and the vergence point. This
        angle is the same for either gaze ray.

        Returns:
            torch.Tensor: (1,) torch.Tensor corresponding to disparity angle in
            mrad.
        """
        left_origin_inUser = self.eyes[0].parent.get_coordinates_inOther(self)
        left_gaze_inUser = self.eyes[0].get_gaze_direction_inOther(self)
        left_ray = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self, left_origin_inUser, left_gaze_inUser
            )

        right_origin_inUser = self.eyes[1].parent.get_coordinates_inOther(self)
        right_gaze_inUser = self.eyes[1].get_gaze_direction_inOther(self)
        right_ray = \
            primitives.Ray.create_from_origin_and_dir_inParent(
                self, right_origin_inUser, right_gaze_inUser
            )

        disparity_mrad = primitives.Ray.get_disparity_mrad(left_ray, right_ray)

        self.remove_child(left_ray)
        self.remove_child(right_ray)

        return disparity_mrad
