"""scene_sampler.py

Class for sampling eye-tracking scenes.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import json
import seet.core as core
import seet.scene as scene
from seet.sampler import sampler_configs, sampler_utils
from seet.sampler import user_sampler, device_sampler
import numpy
import os
import pandas
import random
import torch

CPU_DEVICE_NAME = "cpu"


class SceneSampler:
    """
    Class for sampling a eye-tracking scene. A scene consists of a user and
    a device, which are sampled independently with respect to their parameters.
    In addition, the scene sampler samples the relative position of the user
    and the device.
    """

    def __init__(
        self,
        et_scene=None,
        num_samples=None,
        parameter_file_name=os.path.join(
            sampler_configs.SAMPLER_DIR,
            r"default_sampler/default_scene_sampler.json"
        ),
        requires_grad=False
    ):
        """
        Create a scene sampler. Sampling is performed by independently
        perturbing the parameters of the scene's user and device, and also
        perturbing the relative placement of both.

        Args:
            et_scene (SceneMode, optional): canonical scene; the mode of the
            scene distribution. It may be overwritten by the contents of the
            parameter file. Defaults to the default scene.

            num_samples (int, optional): number of samples to be generated. It
            may be overwritten by the contents of the parameter file. Defaults
            to 100.

            parameter_file_name (str or dict, optional): If a str, this is the
            name of a json file with parameters for user sampling. If a
            dictionary, the keys and values are the parameters for user
            sampling. Defaults to os.path.join( sampler_configs.SAMPLER_DIR,
            r"default_sampler/default_scene_sampler.json" ).

            requires_grad (bool, optional): whether the parameters of the
            samples scenes are differentiable. Defaults to False.
        """

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            self.sampler_parameters = json.load(parameter_file_stream)

        if et_scene is None:
            scene_parameters = \
                self.sampler_parameters.get("scene parameters", None)
            if scene_parameters is None:
                self.scene = scene.SceneModel(requires_grad=requires_grad)
            else:
                path = scene_parameters.get("path", None)
                if path is None:
                    path = scene.scene_configs.SCENE_DIR
                scene_file_name = \
                    os.path.join(
                        path, scene_parameters["parameter file"]
                    )
                self.scene = \
                    scene.SceneModel(
                        parameter_file_name=scene_file_name,
                        requires_grad=requires_grad
                    )
        else:
            self.scene = et_scene

        self.num_samples = \
            self.sampler_parameters.get("num samples", num_samples) \
            if num_samples is None else num_samples

        self.reset_sampler()

    def reset_sampler(self):
        """reset_sampler.

        Reset the sampler so that it can again generate samples. When the
        methods generate_data_for_visibility_analysis and
        generate_data_for_iris_analysis are called, the samples are exhausted,
        so the reset_sampler method must be called before the data-generation
        methods can be called again.
        """

        seed = self.sampler_parameters.get("seed")
        if seed is not None:
            numpy.random.seed(seed)
            random.seed(seed)
            torch.manual_seed(seed)

        self._set_device_sampler()
        self._set_user_sampler()

    def _set_device_sampler(self):
        """_set_device_sampler.

        Set a sampler for the device and for the fit transformations the
        device will undergo.
        """

        device_sampling_parameters = \
            self.sampler_parameters["device sampling parameters"]

        path = device_sampling_parameters.get("path", None)
        if path is None:
            path = sampler_configs.SAMPLER_DIR
        parameter_file_name = device_sampling_parameters["parameter file"]
        if isinstance(parameter_file_name, str):
            parameter_file_name = os.path.join(path, parameter_file_name)

        if device_sampling_parameters.get("apply", False):
            self.device_sampler = \
                device_sampler.DeviceSampler(
                    self.scene.device,
                    self.num_samples,
                    parameter_file_name=parameter_file_name
                )
        else:
            self.device_sampler = None

        device_fit_sampling_parameters = \
            self.sampler_parameters["device fit perturbation parameters"]

        self.device_rotation_sampler = \
            sampler_utils.Sampler.read_SE3_parameters(
                device_fit_sampling_parameters,
                "device rotational perturbation"
            )
        self.device_translation_sampler = \
            sampler_utils.Sampler.read_SE3_parameters(
                device_fit_sampling_parameters,
                "device translational perturbation"
            )

    def _set_user_sampler(self):
        """_set_user_sampler.

        Set a sampler for users.
        """

        user_sampling_parameters = \
            self.sampler_parameters["user sampling parameters"]

        path = user_sampling_parameters.get("path", None)
        if path is None:
            path = sampler_configs.SAMPLER_DIR
        parameter_file_name = user_sampling_parameters["parameter file"]
        if isinstance(parameter_file_name, str):
            parameter_file_name = os.path.join(path, parameter_file_name)

        if user_sampling_parameters.get("apply", False):
            self.user_sampler = \
                user_sampler.UserSampler(
                    self.scene.user,
                    self.num_samples,
                    parameter_file_name=parameter_file_name
                )
        else:
            self.user_sampler = None

    def generate_samples(self):
        """Generate samples of a scene.

        Yields:
            SceneModel: sampled scene.
        """

        # Sample device.
        if self.device_sampler is None:
            device_generator = \
                (self.scene.device for _ in range(self.num_samples))
        else:
            device_generator = self.device_sampler.generate_samples()

        # Sample user.
        if self.user_sampler is None:
            user_generator = \
                (self.scene.user for _ in range(self.num_samples))
        else:
            user_generator = self.user_sampler.generate_samples()

        # Combine each device and user sample to generate a sample scene.
        for device, user in zip(device_generator, user_generator):
            # TODO (Task 39831677): This is very wasteful, we should create
            # some factory methods to create scenes from user and devices
            # rather than from configuration files. Moreover, the replacement
            # of the device and user in the original scene breaks their
            # connection. For example, subsystems of a device have a
            # eye_relief_plane that depends on the corresponding user's eye.
            # This connection needs to be reestablished by explicitly calling #
            # scene._compute_relief_plane()
            scene = self.scene.branchcopy(self.scene)

            # Replace the user and the device in the scene with new ones.
            transform_toScene_fromDevice = \
                scene.device.transform_toParent_fromSelf
            scene.add_child(
                device, transform_toSelf_fromChild=transform_toScene_fromDevice
            )
            scene.remove_child(scene.device)
            scene.device = device

            transform_toScene_fromUser = scene.user.transform_toParent_fromSelf
            scene.add_child(
                user, transform_toSelf_fromChild=transform_toScene_fromUser
            )
            scene.remove_child(scene.user)
            scene.user = user

            # Device and user have already been sampled, but the device fit has
            # not. The rotation is provided in the scene coordinate system.
            if len(self.device_rotation_sampler) > 0:
                angle_axis_deg = \
                    [
                        self.device_rotation_sampler[i].generate_sample()
                        for i in range(3)
                    ]

                angle_axis_rad = core.deg_to_rad(torch.tensor(angle_axis_deg))
                rotation_matrix_toScene_fromScene = \
                    core.rotation_matrix(angle_axis_rad)
                transform_toScene_fromScene = \
                    core.groups.SO3(rotation_matrix_toScene_fromScene)

                apply = True
            else:
                transform_toScene_fromScene = core.groups.SE3.create_identity()

                apply = False

            if len(self.device_translation_sampler) > 0:
                # The device translation is provided in in the coordinate
                # system of the user's left eye at nominal gaze direction.
                translation = \
                    [
                        self.device_translation_sampler[i].generate_sample()
                        for i in range(3)
                    ]

                # We need to convert it to the scene coordinate system.
                translation_toEye_fromEye = \
                    core.groups.SE3(torch.tensor(translation))
                translation_toScene_fromEye = \
                    user.eyes[0].get_transform_toOther_fromSelf(self.scene)
                translation_toScene_fromScene = \
                    core.groups.SE3.compose_transforms(
                        translation_toScene_fromEye,
                        core.groups.SE3.compose_transforms(
                            translation_toEye_fromEye,
                            translation_toScene_fromEye.create_inverse()
                        )
                    )

                transform_toScene_fromScene = \
                    core.groups.SE3.compose_transforms(
                        translation_toScene_fromScene,
                        transform_toScene_fromScene
                    )

                apply = True

            if apply:
                # Perturb the device fit. We want to update the pose of the
                # device by applying update_transform_toParent_fromSelf where
                # parent is the scene and self is the device. We have:
                #
                # update_transform_toScene_fromDevice = \
                #   transform_toScene_fromScene *
                #   transform_toScene_fromDevice
                scene.device.update_transform_toParent_fromSelf(
                    transform_toScene_fromScene
                )

            # This is tricky. Since the device and user were added to the scene
            # in a clunky way, we need to explicitly call the computation of
            # the eye relief plane.
            scene._compute_relief_plane()

            yield scene

            root = scene.get_root()
            del root

    def generate_data_for_sensitivity_analysis(
        self,
        calib_grid=[3, 3],
        pose_grid=[5, 4],
        device=None
    ):
        """Generate data for sensitivity analysis.

        Data is generated by sampling an ET scene, i.e., sampling device fit
        and user parameters. For each simulated user, the derivative of the
        eye-pose parameters with respect to leds, camera extrinsics, camera
        intrinsics, glints, pupil points, and limbus points is computed.
        Derivatives with respect to glint points are computed N x M times,
        where N x M is the size of the grid of points towards which the
        simulated users direct their gaze.

        Args:
            calib_grid (list, optional): number of horizontal and vertical grid
            points towards which simulated users will direct their gaze during
            eye-shape calibration. Defaults to [3, 3].

            pose_grid (list, optional): number of horizontal and vertical grid
            points towards which simulated users will direct their gaze during
            eye-pose estimation. Defaults to [5, 4].

            device (torch.device, optional): device on which to perform the
            computations. If None, use CPU. Defaults to None.

        Returns:
            list of list of torch.Tensor: (N, 2) list of torch tensors, where
            each tensor corresponds to the Jacobian of eye-shape parameters of
            one out of N scenes with respect to the parameters of one of the
            two scene's subsystems.

            (N, 2, M) list of int: (N, 2) list of M indices, where each final
            (M,) list is associated to an ouput Jacobian tensor, specifying the
            columns of the tensor corresponding to derivatives with respect to
            LEDs, camera extrinsic and intrinsic parameters, and glints, pupil
            and limbus points.
        """
        
        # Import here to avoid circular imports
        from sensitivity_analysis import (
            DataWrapper, 
            EyePoseCovariance, 
            EyePoseDerivatives, 
            EyeShapeCovariance, 
            EyeShapeDerivatives
        )

        num_shape_params = EyeShapeDerivatives.get_num_parameters()
        num_pose_params = EyePoseDerivatives.get_num_parameters()

        # TODO: Replace with device.sample_fov method.
        fov_deg = self.scene.device.display_fov / 2
        calib_list_deg = \
            [
                torch.tensor((h.item(), v.item()))
                for h in torch.linspace(-fov_deg[0], fov_deg[0], calib_grid[0])
                for v in torch.linspace(-fov_deg[1], fov_deg[1], calib_grid[1])
            ]

        pose_list_deg = \
            [
                torch.tensor((h.item(), v.item()))
                for h in torch.linspace(-fov_deg[0], fov_deg[0], pose_grid[0])
                for v in torch.linspace(-fov_deg[1], fov_deg[1], pose_grid[1])
            ]

        d_shape_d_data = []
        d_pose_d_data = []
        # This is somewhat special, specific to the computation of the
        # covariance of the pupil center.
        d_pupil_center_d_pose_and_dist = []

        # We need to know the breakdown in the components of the derivative,
        # i.e., which indices refer to leds, camera extrinsics, camera
        # intrinsics, glints, pupil points, and limbus points. For shape
        # estimation, the derivatives have the following format:
        #
        #                                  ... frame i ...
        #                     leds | extr. | intr. | glints | pupil | limbus
        #     shape        ->
        # D = pose         ->
        #     pupil lift.  ->
        #     limbus lift. ->
        #
        # For pose estimation, we have:
        #
        #                     shape | leds | extr. | intr.| glin.| pup.| limbus
        #     shape        ->
        # D = pupil lift.  ->
        #     limbus lift. ->
        #
        # The number of leds and of extrinsic and intrinsic camera parameters
        # is fixed, but the number of glints, pupil, and limbus points is
        # variable. We need to know them in order to keep track of which
        # indices in the derivative represent what.

        d_shape_d_data_indices = []
        d_pose_d_data_indices = []
        counter = 0
        device = device if device is not None else torch.device(CPU_DEVICE_NAME)
        for scene_idx, et_scene in enumerate(self.generate_samples()):
            derivative_data = DataWrapper(et_scene)

            d_shape_d_data_scene = []
            d_shape_d_data_scene_indices = []
            d_pose_d_data_scene = []
            d_pose_d_data_scene_indices = []
            d_pupil_center_d_pose_and_dist_scene = []

            # Number of subsystems doesn't change, but OK, this is clear.
            num_subsystems = len(et_scene.device.subsystems)
            for subsystem_idx in range(num_subsystems):
                derivative_data.set_eye_and_subsystem(index=subsystem_idx)
                shape_cov_calc = EyeShapeCovariance(derivative_data)
                shape_cov_calc.set_list_of_gaze_angles(calib_list_deg)

                # The optimization data consists of eye shape, eye pose, and
                # lifting parameters. We care only about the shape components.
                # But we care about all the data: leds, camera extrinsics,
                # camera intrinsics, glints, pupil points, and limbus points.
                d_optim_d_data = shape_cov_calc.compute_d_optim_d_data()
                d_optim_d_data_indices = \
                    shape_cov_calc.compute_d_optim_d_data_indices()

                d_shape_d_data_scene = \
                    d_shape_d_data_scene + \
                    [d_optim_d_data[:num_shape_params, :], ]

                d_shape_d_data_scene_indices = \
                    d_shape_d_data_scene_indices + [d_optim_d_data_indices, ]

                d_pose_d_data_subsystem = []
                d_pose_d_data_subsystem_indices = []
                d_pupil_center_d_pose_and_dist_subsystem = []
                for pose_deg in pose_list_deg:
                    derivative_data.rotate_eye(pose_deg)

                    pose_cov_calc = EyePoseCovariance(derivative_data)

                    d_optim_d_data = pose_cov_calc.compute_d_optim_d_data()
                    d_optim_d_data_indices = \
                        pose_cov_calc.compute_d_optim_d_data_indices()

                    d_pose_d_data_subsystem = \
                        d_pose_d_data_subsystem + \
                        [d_optim_d_data[:num_pose_params, :], ]

                    d_pose_d_data_subsystem_indices = \
                        d_pose_d_data_subsystem_indices + \
                        [d_optim_d_data_indices, ]

                    # Pupil center x is given by
                    #
                    # x = n * dist + c,
                    #
                    # where n is the unit gaze direction, dist is the distance
                    # between the pupil center and the rotation center, and c
                    # is the rotation center. Moreover,
                    #
                    # n = R_x(vert. angle) * R_y(horiz. angle).
                    #
                    # Therefore,
                    #
                    # d_pupil_center_d_pose_d_(angles, c, dist) =
                    #   (d_n_d_angles * dist, eye(3), n)

                    # TODO: n and derivative_data.eye_pose_parameters.angles_deg should be on GPU if device is GPU.
                    # However, because they are both derived from the scene, which is on CPU, we likely need to move the entire scene to GPU
                    # to make sure we don't break the computation graph
                    eye = derivative_data.eye
                    n = eye.get_gaze_direction_inParent()
                    d_n_d_angles = \
                        core.compute_auto_jacobian_from_tensors(
                            n, derivative_data.eye_pose_parameters.angles_deg
                        )


                    dist = eye.distance_from_rotation_center_to_pupil_plane
                    d_pupil_d_pose_and_dist_ = \
                        torch.hstack(
                            (d_n_d_angles * dist, torch.eye(3), n.view(3, 1))
                        )

                    d_pupil_center_d_pose_and_dist_subsystem = \
                        d_pupil_center_d_pose_and_dist_subsystem + \
                        [d_pupil_d_pose_and_dist_, ]

                    derivative_data.unrotate_eye(pose_deg)

                    yield counter
                    counter += 1

                d_pose_d_data_scene = \
                    d_pose_d_data_scene + [d_pose_d_data_subsystem, ]
                d_pose_d_data_scene_indices = \
                    d_pose_d_data_scene_indices + \
                    [d_pose_d_data_subsystem_indices, ]
                d_pupil_center_d_pose_and_dist_scene = \
                    d_pupil_center_d_pose_and_dist_scene + \
                    [d_pupil_center_d_pose_and_dist_subsystem, ]

            d_shape_d_data = d_shape_d_data + [d_shape_d_data_scene, ]
            d_shape_d_data_indices = \
                d_shape_d_data_indices + [d_shape_d_data_scene_indices, ]

            d_pose_d_data = d_pose_d_data + [d_pose_d_data_scene, ]
            d_pose_d_data_indices = \
                d_pose_d_data_indices + [d_pose_d_data_scene_indices, ]

            d_pupil_center_d_pose_and_dist = \
                d_pupil_center_d_pose_and_dist + \
                [d_pupil_center_d_pose_and_dist_scene, ]

        return \
            d_shape_d_data, \
            d_shape_d_data_indices, \
            d_pose_d_data, \
            d_pose_d_data_indices, \
            d_pupil_center_d_pose_and_dist
