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

    def generate_data_for_visibility_analysis(self, gaze_grid=[5, 4]):
        """generate_data_for_visibility_analysis.

        Generate data for visibility analysis.

        Args:
            gaze_grid (list, optional): size of gaze direction grid on which to
            sample the gaze directions. First element is the number of
            horizontal samples, second element is the number of vertical
            samples. Defaults to [5, 4].
        """

        # We want to collect data about LED visibility, working distance, eye
        # clipping, gaze direction, eye relief.

        #######################################################################
        # Device-only data
        header_subsystem = ["Subsystem", ]
        header_grid_angle = ["Horiz. angle", "Vert. angle"]
        # This assumes that the subsystems have the same number of LEDs.
        num_leds = self.scene.device.subsystems[0].led_set.num
        header_LEDs = ["LED {:02d}".format(i) for i in range(1, num_leds + 1)]

        #######################################################################
        # Device + user data.
        header_camera_center_in_pupil = \
            [
                "Camera center in pupil {:s}".format(ax)
                for ax in ["x", "y", "z"]
            ]
        header_delta_eye_relief = ["Delta eye relief", ]
        header_scene_index = ["Scene index", ]

        #######################################################################
        # User-only data.
        header_gaze_direction = \
            ["Gaze {:s}".format(ax) for ax in ["x", "y", "z"]]
        header_IPD = ["IPD", ]

        #######################################################################
        # Putting it all together
        header = \
            header_subsystem + \
            header_grid_angle + \
            header_LEDs + \
            header_camera_center_in_pupil + \
            header_delta_eye_relief + \
            header_scene_index + \
            header_gaze_direction + \
            header_IPD

        num_subsystems = len(self.scene.device.subsystems)
        data = []
        for scene_index, et_scene in enumerate(self.generate_samples()):
            for subsystem_index in range(num_subsystems):
                ###############################################################
                # Device-only data.
                # Subsystem data.
                row_subsystem = [subsystem_index, ]

                # Grid angle data.
                if gaze_grid[0] != 1 or gaze_grid[1] != 1:
                    fov_range_deg = self.scene.device.display_fov / 2
                    h_fov_range_deg = \
                        torch.linspace(
                            -fov_range_deg[0], fov_range_deg[0], gaze_grid[0]
                        )
                    v_fov_range_deg = \
                        torch.linspace(
                            -fov_range_deg[1], fov_range_deg[1], gaze_grid[1]
                        )
                    rotate = True
                else:
                    h_fov_range_deg = torch.zeros(1)
                    v_fov_range_deg = torch.zeros(1)
                    rotate = False

                subsystem = et_scene.device.subsystems[subsystem_index]

                camera_index = 0  # In the future, we may have stereo.
                camera = subsystem.cameras[camera_index]

                eye = et_scene.user.eyes[subsystem_index]

                for hi in range(gaze_grid[0]):
                    for vi in range(gaze_grid[1]):
                        # Rotate the eye if required.
                        if rotate:
                            angles_deg = \
                                torch.stack(
                                    (h_fov_range_deg[hi], v_fov_range_deg[vi])
                                )
                            eye.rotate_from_gaze_angles_inParent(angles_deg)
                        else:
                            angles_deg = torch.zeros(2)

                        row_grid_angle = \
                            [*angles_deg.clone().detach().numpy()]

                        #######################################################
                        # Device plus user data.
                        # Scene (device + user) index.

                        # LED-visibility data.
                        glints_inCamera = \
                            et_scene.generate_glints_inOther(
                                other_node=camera,
                                subsystem_index=subsystem_index,
                                camera_index=camera_index
                            )

                        row_LEDs = \
                            [int(g is not None) for g in glints_inCamera]

                        # Camera center in pupil.
                        transform_toPupil_fromCamera = \
                            camera.get_transform_toOther_fromSelf(eye.pupil)
                        optical_center_in_pupil = \
                            transform_toPupil_fromCamera.transform(
                                torch.zeros(3)
                            )
                        row_camera_center_in_pupil = \
                            [*optical_center_in_pupil.clone().detach().numpy()]

                        # Eye-relief data.
                        eye_relief_plane = subsystem.eye_relief_plane
                        cornea_apex_inPlane = \
                            eye.get_cornea_apex_inOther(eye_relief_plane)

                        delta_eye_relief = \
                            -1 * eye_relief_plane.\
                            compute_signed_distance_to_point_inPlane(
                                cornea_apex_inPlane
                            ).clone().detach().numpy()

                        row_delta_eye_relief = [delta_eye_relief, ]

                        # Keeping track of the samples.
                        row_scene_index = [scene_index, ]

                        #######################################################
                        # User-only data.
                        # Gaze-direction data.
                        gaze_direction_inScene = \
                            eye.get_gaze_direction_inOther(et_scene)
                        row_gaze_direction = \
                            [*gaze_direction_inScene.clone().detach().numpy()]

                        # IPD data.
                        IPD_data = et_scene.user.compute_IPD()
                        row_IPD = [IPD_data.clone().detach().numpy(), ]

                        #######################################################
                        # Putting it all together.
                        row = \
                            row_subsystem + \
                            row_grid_angle + \
                            row_LEDs + \
                            row_camera_center_in_pupil + \
                            row_delta_eye_relief + \
                            row_scene_index + \
                            row_gaze_direction + \
                            row_IPD

                        data = data + [row, ]

                        if rotate:
                            eye.unrotate_from_gaze_angles_inParent(angles_deg)

        return pandas.DataFrame(data, columns=header)

    def generate_data_for_iris_analysis(self, num_angles=10, num_radii=10):
        """generate_data_for_iris_analysis.

        Generate data with area of visible iris in mm, and area of projected
        iris in camera in pixels, as well as percentage of iris that is
        visible. The data is for nominal gaze direction only, as this is a very
        expensive computation.

        Args:
            num_angles (int, optional): number of angles to be sampled around
            the iris. Defaults to 10.

            num_radii (int, optional): number of radii to be sample for each
            angle. Defaults to 10.

        Returns:
            pandas.DataFrame: pandas data frame with columns corresponding to
            scene subsystem, percentage of iris visible, area of visible iris
            in mm^2, and area of the projection of the visible iris in the
            camera, in pixels^2.
        """

        header = [
            "Subsystem",
            "Percentage visible",
            "Area in iris [mm^2]",
            "Area in image [pix^2]"
        ]

        # Parameters to compute area element
        d_theta = torch.tensor(2 * torch.pi / num_angles, requires_grad=True)

        num_subsystems = len(self.scene.device.subsystems)
        data = []
        for et_scene in self.generate_samples():
            for subsystem_index in range(num_subsystems):
                camera = et_scene.device.subsystems[subsystem_index].cameras[0]
                eye = et_scene.user.eyes[subsystem_index]
                cornea = eye.cornea
                limbus = eye.limbus

                ###############################################################
                # Sample iris in polar coordinates.
                optical_center_inCornea = \
                    camera.get_optical_center_inOther(cornea)
                transform_toCornea_fromLimbus = \
                    limbus.get_transform_toOther_fromSelf(cornea)

                d_r = limbus.radius / num_radii
                # TODO: This is ugly! We need to create a function T0() that
                # generates the detached value of the zero tensor.
                area_total_mm = core.T0.clone().detach()
                area_mm = core.T0.clone().detach()
                area_pix = core.T0.clone().detach()
                for i in range(num_angles):
                    theta = i * d_theta
                    for j in range(num_radii):
                        r = j * d_r
                        x = r * torch.cos(theta)
                        y = r * torch.sin(theta)
                        point_in2DIris = torch.hstack((x, y))

                        point_in3DIris = \
                            torch.hstack((point_in2DIris, core.T0))
                        iris_point_inCornea = \
                            transform_toCornea_fromLimbus.transform(
                                point_in3DIris
                            )

                        # Make sure point is inside cornea.
                        if cornea.compute_algebraic_distance_inEllipsoid(
                            iris_point_inCornea
                        ) >= 0:
                            continue

                        d_x_d_y = r * d_r * d_theta
                        area_total_mm += d_x_d_y

                        refraction_point_inCornea = \
                            cornea.compute_refraction_point_inEllipsoid(
                                optical_center_inCornea,
                                iris_point_inCornea,
                                eta_at_destination=cornea.refractive_index
                            )
                        if refraction_point_inCornea is None:
                            continue

                        # Check occlusion by device occluder, if one is
                        # present.
                        unit_list_refraction_point_inCornea = \
                            et_scene.device.subsystems[
                                subsystem_index
                            ].apply_occluder_inOther(
                                cornea,
                                [refraction_point_inCornea, ],  # list
                                reference_point_inOther=optical_center_inCornea
                            )
                        # Input is list of points, and so is output.
                        refraction_point_inCornea = \
                            unit_list_refraction_point_inCornea[0]

                        if refraction_point_inCornea is not None:
                            area_mm += d_x_d_y

                            refraction_point_inPixels = \
                                camera.project_toPixels_fromOther(
                                    refraction_point_inCornea, cornea
                                )

                            d_inPixels_d_in2DIris = \
                                core.compute_auto_jacobian_from_tensors(
                                    refraction_point_inPixels, point_in2DIris
                                )

                            area_pix += \
                                d_x_d_y * \
                                torch.abs(
                                    torch.linalg.det(d_inPixels_d_in2DIris)
                                )

                percentage = (100 * area_mm / area_total_mm).item()
                data = \
                    data + \
                    [
                        [
                            subsystem_index,
                            percentage,
                            area_mm.item(),
                            area_pix.item()
                        ]
                    ]

        return pandas.DataFrame(data, columns=header)
