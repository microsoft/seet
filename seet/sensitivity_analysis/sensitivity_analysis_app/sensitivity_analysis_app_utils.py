"""Utility methods and classes for covariance analysis.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import datetime
import json
from seet.core import numeric
from seet.sampler import SceneSampler
from seet.sensitivity_analysis import \
    CameraCovarianceCalculator, \
    EyeShapeCovariance, \
    EyePoseCovariance, \
    FeaturesCovarianceCalculator, \
    LEDsCovarianceCalculator
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import pickle
import torch


class SensitivityAnalysisAppUtils():
    def __init__(self):
        self.data_dictionary = None
        self.leds_covariance_calculator = None
        self.camera_covariance_calculator = None
        self.features_covariance_calculator = None

    def load_derivatives(self, derivatives_file_name):
        self.derivatives_file_name = derivatives_file_name
        with open(derivatives_file_name, 'rb') as file_stream:
            self.data_dictionary = pickle.load(file_stream)

    def load_data(self, data_dictionary):
        self.data_dictionary = data_dictionary

    def generate_data(self, input_file_name, num_samples=None):
        scene_sampler = \
            SceneSampler(
                num_samples=num_samples,
                parameter_file_name=input_file_name,
                requires_grad=True
            )

        # The generation of data for sensitivity analysis yields the sample
        # number that has been processed, and after all samples are processed
        # it emits a "StopIteration" signal containing the result. Because of
        # the yielding, the output of the method is a generator, that is best
        # used inside a loop.
        generator = scene_sampler.generate_data_for_sensitivity_analysis()
        while True:
            try:
                # This can be used by a higher-level function, e.g., a GUI, to
                # update a progress bar.
                yield next(generator)

            except StopIteration as data_container:
                # Once the generator stops iterating, the stop iterator
                # container will hold the derivatives (the return value of the
                # method) in its value.
                shape_derivatives, \
                    shape_indices, \
                    pose_derivatives, \
                    pose_indices, \
                    pupil_center_derivatives = data_container.value
                break

        self.data_dictionary = \
            {
                "shape derivatives": shape_derivatives,
                "shape indices": shape_indices,
                "pose derivatives": pose_derivatives,
                "pose indices": pose_indices,
                "pupil center derivatives": pupil_center_derivatives
            }

    def save_data(self, output_dir_name):
        results_path = os.path.join(output_dir_name, "derivatives")
        os.makedirs(results_path, exist_ok=True)

        now = datetime.datetime.now()
        prefix = now.strftime("%Y-%m-%d @ %H-%M-%S.%f")
        file_prefix = os.path.join(results_path, prefix)
        output_file_name = file_prefix + " dict.pkl"

        with open(output_file_name, "wb") as output_file_stream:
            pickle.dump(self.data_dictionary, output_file_stream)

        return output_file_name

    def load_configuration(self, input_dir_name):
        self.configuration_dir_name = input_dir_name

        #######################################################################
        # LEDS
        leds_file_name = os.path.join(input_dir_name, "led_covariances.json")
        self.leds_covariance_calculator = \
            LEDsCovarianceCalculator(leds_file_name)

        #######################################################################
        # Camera
        camera_file_name = \
            os.path.join(input_dir_name, "camera_covariances.json")
        self.camera_covariance_calculator = \
            CameraCovarianceCalculator(camera_file_name)

        #######################################################################
        # Image features
        features_file_name = \
            os.path.join(input_dir_name, "feature_covariances.json")
        self.features_covariance_calculator = \
            FeaturesCovarianceCalculator(features_file_name)

    def load_stds(self, std_names, std_dict):
        self.std_names = std_names
        self.std_dict = std_dict
        stds = [self.std_dict[name] for name in self.std_names]
        self.stds = torch.tensor(stds, requires_grad=True)

        # LED standard deviations are the first three components.
        self.led_stds = self.stds[:3]
        self.leds_covariance_calculator.set_stds(self.led_stds)

        # Camera standard deviations are components 3:18. That's 6 extrinsic
        # parameters (3:10) and 9 intrinsic parameters (4 (10:14) for pinhole
        # model plus 5 (14:18) for distortion in polynomial3K model.)
        self.camera_stds = self.stds[3:18]
        self.camera_covariance_calculator.set_stds(self.camera_stds)

        # Features are two dimensional
        self.features_stds = \
            torch.hstack(
                (self.stds[18:].view(-1, 1), self.stds[18:].view(-1, 1))
            ).flatten()
        # Features are two dimensional.
        self.features_covariance_calculator.set_stds(self.features_stds)

    def compute_covariances(self):
        """Compute pose covariances.
        """

        # The experiment computes the derivatives for N sampled scenes, two
        # eyes per scene, and for pose parameters, 5 x 4 times to cover the
        # field of view.
        d_shape_d_param = self.data_dictionary["shape derivatives"]
        d_shape_d_param_indices = self.data_dictionary["shape indices"]
        d_pose_d_param = self.data_dictionary["pose derivatives"]
        d_pose_d_param_indices = self.data_dictionary["pose indices"]
        d_pupil_center_d_pose_and_dist = \
            self.data_dictionary["pupil center derivatives"]

        # Number of scenes must match.
        assert (len(d_pose_d_param) == len(d_pose_d_param_indices))
        assert (len(d_pose_d_param) == len(d_shape_d_param))
        assert (len(d_pose_d_param) == len(d_shape_d_param_indices))
        assert (len(d_pose_d_param) == len(d_pupil_center_d_pose_and_dist))

        # These will be triply nested lists. The first level corresponds to a
        # sampled scene. The second level corresponds to a subsystem in the
        # scene's device. The third level corresponds to a gaze direction.
        self.eye_pose_covariance = []
        self.pupil_center_covariance = []

        # Iterate over all eye-tracking scenes (sampled combination of device
        # and user) used to generate the data.
        for scene_idx in range(len(d_shape_d_param)):
            d_shape_d_param_scene = d_shape_d_param[scene_idx]
            d_shape_d_param_scene_indices = d_shape_d_param_indices[scene_idx]
            d_pose_d_param_scene = d_pose_d_param[scene_idx]
            d_pose_d_param_scene_indices = d_pose_d_param_indices[scene_idx]
            d_pupil_center_d_pose_and_dist_scene = \
                d_pupil_center_d_pose_and_dist[scene_idx]

            # Number of subsystems for the device in each scene must match.
            assert (
                len(d_pose_d_param_scene) ==
                len(d_pose_d_param_scene_indices)
            )
            assert (
                len(d_pose_d_param_scene) ==
                len(d_shape_d_param_scene)
            )
            assert (
                len(d_pose_d_param_scene) ==
                len(d_shape_d_param_scene_indices)
            )
            assert (
                len(d_pose_d_param_scene) ==
                len(d_pupil_center_d_pose_and_dist_scene)
            )

            eye_pose_covariance_scene = []
            pupil_center_covariance_scene = []
            # Each scene has multiple (typically two) subsystems; iterate over
            # them.
            for subsystem_idx in range(len(d_shape_d_param_scene)):
                d_shape_d_param_subsystem = \
                    d_shape_d_param_scene[subsystem_idx]
                d_shape_d_param_subsystem_indices = \
                    d_shape_d_param_scene_indices[subsystem_idx]

                d_pose_d_param_subsystem = d_pose_d_param_scene[subsystem_idx]
                d_pose_d_param_subsystem_indices = \
                    d_pose_d_param_scene_indices[subsystem_idx]

                d_pupil_center_d_pose_and_dist_subsystem = \
                    d_pupil_center_d_pose_and_dist_scene[subsystem_idx]

                # Each subsystem of each scene will yield an eye-shape
                # covariance matrix.
                eye_shape_covariance = \
                    EyeShapeCovariance.compute_covariance_standalone(
                        d_shape_d_param_subsystem,
                        d_shape_d_param_subsystem_indices,
                        True,  # Should not be hardcoded. with_limbus is True.
                        self.leds_covariance_calculator,
                        self.camera_covariance_calculator,
                        self.features_covariance_calculator
                    )

                # Number of gaze directions for pose in each subsystem must
                # match.
                assert (
                    len(d_pose_d_param_subsystem) ==
                    len(d_pose_d_param_subsystem_indices)
                )
                assert (
                    len(d_pose_d_param_subsystem) ==
                    len(d_pupil_center_d_pose_and_dist_subsystem)
                )

                # For each subsystem, there are multiple gaze directions;
                # iterate over them.
                eye_pose_covariance_subsystem = []
                pupil_center_covariance_subsystem = []
                for gaze_idx in range(len(d_pose_d_param_subsystem)):
                    # Computation of eye-pose covariance.
                    gaze_d_pose_d_param = d_pose_d_param_subsystem[gaze_idx]
                    gaze_d_pose_d_param_indices = \
                        d_pose_d_param_subsystem_indices[gaze_idx]

                    eye_pose_covariance = \
                        EyePoseCovariance.compute_covariance_standalone(
                            gaze_d_pose_d_param,
                            gaze_d_pose_d_param_indices,
                            False,  # Shouldn't hardcode with_limbus is False.
                            eye_shape_covariance,
                            self.leds_covariance_calculator,
                            self.camera_covariance_calculator,
                            self.features_covariance_calculator
                        )

                    # Computation of pupil-center covariance.
                    gaze_d_pupil_center_d_pose_and_dist = \
                        d_pupil_center_d_pose_and_dist_subsystem[gaze_idx]

                    pupil_center_covariance = \
                        EyePoseCovariance.\
                        compute_covariance_pupil_center_standalone(
                            gaze_d_pupil_center_d_pose_and_dist,
                            eye_pose_covariance,
                            eye_shape_covariance
                        )

                    eye_pose_covariance_subsystem = \
                        eye_pose_covariance_subsystem + [eye_pose_covariance, ]

                    pupil_center_covariance_subsystem = \
                        pupil_center_covariance_subsystem + \
                        [pupil_center_covariance, ]

                eye_pose_covariance_scene = \
                    eye_pose_covariance_scene + \
                    [eye_pose_covariance_subsystem, ]

                pupil_center_covariance_scene = \
                    pupil_center_covariance_scene + \
                    [pupil_center_covariance_subsystem, ]

            self.eye_pose_covariance = \
                self.eye_pose_covariance + [eye_pose_covariance_scene, ]
            self.pupil_center_covariance = \
                self.pupil_center_covariance + \
                [pupil_center_covariance_scene, ]

    def compute_covariances_for_KPIs(self):
        """Compute the covariance for the final pupil position and gaze
        direction KPIs.
        """

        # Average results across scenes.
        psi_variance_quantile = 0.0
        position_variance_quantile = 0.0
        for cov_scene, pupil_cov_scene in zip(
            self.eye_pose_covariance, self.pupil_center_covariance
        ):
            # All eye-pose covariances for a given user. For a given scene,
            # this includes all subsystems of the device, i.e., left and right
            # eyes, and all gaze directions.
            eye_pose_covariance_user = \
                [
                    cov_subsystem_and_gaze
                    for cov_subsystem in cov_scene
                    for cov_subsystem_and_gaze in cov_subsystem
                ]
            pupil_covariance_user = \
                [
                    pupil_cov_subsystem_and_gaze
                    for pupil_cov_subsystem in pupil_cov_scene
                    for pupil_cov_subsystem_and_gaze in pupil_cov_subsystem
                ]

            # Compute quantiles across subsystems and gaze directions. For
            # gaze, we've hardcoded the 80th percentile.
            psi_variance_list = [
                torch.diag(cov[:2, :2]).sum()
                for cov in eye_pose_covariance_user
            ]

            psi_variance = torch.stack(psi_variance_list)
            gaze_quantile = 0.8  # Should not hardcode this.
            psi_variance_quantile = \
                psi_variance_quantile + \
                torch.quantile(psi_variance, gaze_quantile)

            # For pupil position, we've hardcoded the 50th percentile.
            pupil_position_covariance = \
                torch.hstack(
                    [
                        torch.diag(cov).view((3, 1))
                        for cov in pupil_covariance_user
                    ]
                )
            position_quantile = 0.5  # Should not hardcode this.
            position_variance_quantile = \
                position_variance_quantile + \
                torch.quantile(
                    pupil_position_covariance, position_quantile, dim=1
                )

        # To average, we divide by number of scenes.
        num_scenes = len(self.eye_pose_covariance)
        self.psi_variance = psi_variance_quantile / num_scenes
        self.position_variance = position_variance_quantile / num_scenes

    def compute_contributions_to_KPIs(self):
        """Compute the contribution of each input parameter to final errors.
        """

        self.d_psi_std_d_stds = \
            numeric.compute_auto_jacobian_from_tensors(
                torch.sqrt(self.psi_variance), self.stds
            )
        self.d_position_std_d_stds = \
            numeric.compute_auto_jacobian_from_tensors(
                torch.sqrt(self.position_variance), self.stds
            )

    def plot_results(
        self, gaze_kpi_deg, pose_kpi_mm, all_plots=False
    ):
        """Visualize results of experiment
        """

        gaze_error_std = self.d_psi_std_d_stds * self.stds
        normalizer = torch.sqrt(self.psi_variance) / gaze_error_std.sum()
        # Gaze is 1 by N, where N is the number of std components
        gaze_error_std = (gaze_error_std * normalizer).view(1, -1)

        position_error_std = self.d_position_std_d_stds * self.stds.view(1, -1)
        normalizer = \
            torch.sqrt(self.position_variance) / \
            (position_error_std).sum(dim=1)
        # Position is 3 (x,y,z) by N, where N is the number of std components.
        position_error_std = position_error_std * normalizer.view(-1, 1)

        # Plot standard deviations for gaze and position. Group results into
        # contributions from LEDs, camera extrinsics, camera intrinsics, and
        # features.
        fig, axs = \
            plt.subplots(1, 3, figsize=(8, 3), width_ratios=(1, 3, 1))
        suptitle = "Standard Deviation of Errors"

        # Create space for the legend.
        axs[2].axis("off")

        width = 0.8
        start = (0, 3, 9, 18)
        stop = (3, 9, 18, 21)
        label = \
            (
                "LEDs",
                "Camera Extrinsics",
                "Camera Intrinsics",
                "Image Features",
                "KPI"
            )
        x = (("gaze angle",), ("$x$ coord.", "$y$ coord.", "$z$ coord."))
        c = ("r", "g", "b", "magenta")
        error_std = (gaze_error_std, position_error_std)
        titles = ("Gaze", "Position")
        y_labels = (r"$\sigma$ [deg]", r"$\sigma$ [mm]")
        kpi_line = (gaze_kpi_deg, pose_kpi_mm)
        kpi_line_min_max = [[[0, 1]], [[0, 1 / 3], [1 / 3, 2 / 3], [2 / 3, 1]]]
        kpi_legend = False

        h = []
        for i in range(len(error_std)):
            std_i = error_std[i].clone().detach()

            x_i = x[i]
            y_i0 = torch.zeros(len(x_i))

            # Iterate over LED, cam extr., cam intr., and image feature
            # contribs. stds coefficients are:
            #
            # 0:3 for LEDs 3:6, 6:9, for camera location and orientation 9:13,
            # and 13:18 for camera pinhole and distortion 18:21 for image
            # features.
            ax = axs[i]
            hi = []
            for j in range(len(start)):
                base_rgb = torch.tensor(mpl.colors.to_rgb(c[j]))
                dark_factor = 0.3
                darker_rgb = dark_factor * base_rgb
                lighter_rgb = (1 - dark_factor) * torch.ones(3) + darker_rgb
                middle_rgb = 0.5 * (darker_rgb + lighter_rgb)
                # Stretch for contrast.
                max_rgb = middle_rgb.max()
                min_rgb = middle_rgb.min()
                stretch_rgb = (middle_rgb - min_rgb) / (max_rgb - min_rgb)
                final_rgb = \
                    (1 - dark_factor) * middle_rgb + dark_factor * stretch_rgb

                a_j, b_j = start[j], stop[j]
                y_ij = std_i[:, a_j:b_j].sum(dim=1)
                hi.append(
                    ax.bar(
                        x_i,
                        y_ij,
                        width,
                        bottom=y_i0,
                        color=final_rgb.numpy(),
                        label=label[j]
                    )[0]
                )
                y_i0 = y_i0 + y_ij

            # Draw KPI line.
            assert len(kpi_line[i]) == len(kpi_line_min_max[i])
            for k in range(len(kpi_line_min_max[i])):
                min_max = kpi_line_min_max[i][k]
                if kpi_line[i][k] is not None:
                    h_kpi = \
                        ax.axhline(
                            kpi_line[i][k],
                            xmin=min_max[0],
                            xmax=min_max[1],
                            c="red",
                            ls="--",
                            lw=3
                        )
                    if kpi_legend is False:
                        hi.append(h_kpi)
                        kpi_legend = True

            h = h + [hi, ]
            ax.grid(visible=True, axis="y")
            ax.set_title(titles[i])
            ax.set_ylabel(y_labels[i])

        # Create legend
        i_max = 0
        for i in range(len(h)):
            if len(h[i]) >= len(h[i_max]):
                i_max = i

        # We reverse the order of the legend so that the labels pile from top
        # to bottom a
        fig.legend(h[i_max][::-1], label[::-1], loc="center right", ncol=1)

        plt.suptitle(suptitle)
        plt.tight_layout()

        # Save plots. We save in the same directory as the configuration files,
        # in a subdirectory with the same prefix as the input derivatives, and
        # a new subdirectory therein.
        derivatives_base_name = os.path.basename(self.derivatives_file_name)
        derivatives_dir_name = \
            "results with derivatives " + \
            os.path.splitext(derivatives_base_name)[0]  # Strip extension pkl.
        derivatives_path_name = \
            os.path.join(self.configuration_dir_name, derivatives_dir_name)

        self.plot_output_dir_name = \
            os.path.join(
                derivatives_path_name,
                "plot " +
                datetime.datetime.now().strftime("%Y-%m-%d @ %H-%M-%S.%f")
            )
        os.makedirs(self.plot_output_dir_name, exist_ok=True)

        # Save extra parameters used to produce plots.
        data_file_name = os.path.join(self.plot_output_dir_name, "data.json")
        with open(data_file_name, "w") as data_file_stream:
            json.dump(self.std_dict, data_file_stream)

        plt_file_name = os.path.join(self.plot_output_dir_name, "main.png")
        plt.savefig(plt_file_name, bbox_inches="tight")

        plt_file_name = os.path.join(self.plot_output_dir_name, "main.pdf")
        plt.savefig(plt_file_name, bbox_inches="tight")
        plt.show()

        if all_plots:
            self.plot_detailed_results(gaze_error_std, position_error_std)

    def plot_detailed_results(self, gaze_error_std, position_error_std):
        suptitle = "Standard Deviation of Errors"

        width = 0.8
        start = (0, 3, 9, 18)
        stop = (3, 9, 18, 21)
        label = \
            (
                "LEDs",
                "Camera Extrinsics",
                "Camera Intrinsics",
                "Image Features",
                "KPI"
            )
        c = ("r", "g", "b", "magenta")

        error_std = (gaze_error_std, position_error_std)

        # Isolate impact of components. LEDs, cam extr., cam intr., image
        # features
        width = 0.8
        x = ((r"$\theta$",), ("$x$", "$y$", "$z$"))
        titles = ("Gaze", "Position")
        y_labels = ("%", "%")
        contribution_label = (
            ("x LED coords.", "y LED coords.", "z LED coords."),
            (
                "pitch", "yaw", "roll",
                "opt. center x ", "opt. center y", "opt. center z"
            ),
            (
                "principal pt. x", "principal pt. y",
                "focal length x", "focal length y",
                "dist. center x", "dist. center y",
                "dist. coeff. k0", "dist. coeff. k1", "dist. coeff. k2"
            ),
            ("glint", "pupil", "limbus")
        )
        # We don't switch the color pallete for LEDs, we switch for the camera
        # extrinsics at component 3 (so we have rotation and translation as
        # separate blocks), we switch for the camera intrinsics at component 4
        # (so we have pinhole and distortion as separate blocks.)
        color_switch = (3, 3, 4, 3)
        for k in range(len(contribution_label)):
            label_k = contribution_label[k]

            # Plot standard deviations for gaze and position.
            fig, axs = \
                plt.subplots(1, 3, figsize=(8, 3), width_ratios=(1, 3, 1))

            # Create space for the legend.
            axs[2].axis("off")

            # Create colors.
            base_rgb = torch.tensor(mpl.colors.to_rgb(c[k]))
            dark_factor = 0.3
            darker_rgb = dark_factor * base_rgb
            lighter_rgb = (1 - dark_factor) * torch.ones(3) + darker_rgb

            # "i" iterates over gaze and position.
            for i in range(len(error_std)):
                std_i = error_std[i].clone().detach()
                normalizer_ki = \
                    std_i[:, start[k]:stop[k]].sum(dim=1, keepdim=True)
                x_i = x[i]
                y_i0 = torch.zeros(len(x_i))

                # Iterate over LED, cam extr., cam intr., and image feature
                # contribs. stds coefficients are:
                #
                # 0:3 for LEDs 3:6, 6:9, for camera location and orientation
                # 9:13, and 13:18 for camera pinhole and distortion 18:21 for
                # image features.
                ax = axs[i]
                h = []

                # "j" iterates over the component of the contribution for each
                # element k (LEDs, camera extr., camera intr., image features).
                # For example, LEDs have components "x", "y", and "y"; camera
                # intr. have components "px", "py", "fx", "fy", "cx", "cy",
                # "k0", "k1", and "k2".
                num_components = stop[k] - start[k]
                jj = torch.linspace(0, 1, num_components)

                do_switch = False
                if num_components > color_switch[k]:
                    do_switch = True
                    other_jj = \
                        torch.hstack(
                            (
                                torch.linspace(0, 1, color_switch[k]),
                                torch.linspace(
                                    0, 1, num_components - color_switch[k])
                            )
                        )

                for j in range(start[k], stop[k]):
                    if do_switch:
                        color_inversion = True
                        alpha = other_jj[j - start[k]]
                        if j - start[k] < color_switch[k]:
                            current_rgb = \
                                (1 - alpha) * lighter_rgb + alpha * darker_rgb
                        else:
                            if color_inversion:
                                new_darker_rgb = lighter_rgb
                                new_lighter_rgb = darker_rgb
                                color_inversion = False
                            current_rgb = \
                                (1 - alpha) * new_lighter_rgb + \
                                alpha * new_darker_rgb
                            current_rgb = \
                                torch.ones_like(current_rgb) - current_rgb
                    else:
                        alpha = jj[j - start[k]]
                        current_rgb = \
                            (1 - alpha) * lighter_rgb + alpha * darker_rgb

                    y_ij = \
                        (
                            100.0 * std_i[:, j].view(-1, 1) / normalizer_ki
                        ).flatten()
                    h.append(
                        ax.bar(
                            x_i,
                            y_ij,
                            width,
                            bottom=y_i0,
                            color=current_rgb.numpy(),
                            label=label_k[j - start[k]]
                        )[0]
                    )
                    y_i0 = y_i0 + y_ij

                ax.set_ylim([0, 100])
                ax.grid(visible=True, axis="y")
                ax.set_title(titles[i])
                ax.set_ylabel(y_labels[i])

            # Create legend
            fig.legend(h[::-1], label_k[::-1], loc="center right", ncol=1)

            plt.suptitle(suptitle + ". Contribution from " + label[k])
            plt.tight_layout()

            plt_file_name = \
                os.path.join(self.plot_output_dir_name, label[k] + ".png")
            plt.savefig(plt_file_name, bbox_inches="tight")

            plt_file_name = \
                os.path.join(self.plot_output_dir_name, label[k] + ".pdf")
            plt.savefig(plt_file_name, bbox_inches="tight")

            plt.show()
            plt.draw()

    def close(self):
        # Close all figures.
        plt.close()
