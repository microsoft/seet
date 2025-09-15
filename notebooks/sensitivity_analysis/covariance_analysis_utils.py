"""Utility methods and classes for covariance analysis.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import ipywidgets
from IPython.display import display
from seet.core import numeric
from seet.sensitivity_analysis import \
    CameraCovarianceCalculator, \
    EyeShapeCovariance, \
    EyePoseCovariance, \
    FeaturesCovarianceCalculator, \
    LEDsCovarianceCalculator, \
    SENSITIVITY_ANALYSIS_DIR
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import torch


def std_slider(name, max, min=0.0):
    """Basic slider for setting the standard deviation for a single variable.

    Args:
        name (str): variable name.

        max (float): maximum value of standard deviation.

        min(float, optional): minimum value of standard deviation. Defaults to
        0.0.
    """

    return \
        ipywidgets.FloatSlider(
            value=max/2,
            min=min,
            max=max,
            step=max/10,
            description=name,
            orientation='vertical'
        )


def multi_slider(name, max, min=[0.0, 0.0, 0.0], each_name=["x", "y", "z"]):
    """Multi slider.

    Useful to group sliders for x, y, z or pitch, yaw, and roll.

    Args:
        name (str): slider-group name.

        max (list of float): (3,) list with maximum values for each slider.

        min (list, optional): (3,) list with minimum values for each slider.
        Defaults to [0.0, 0.0, 0.0].

        name of each slider (list, optional): _description_.
        Defaults to ["x", "y", "z"].
    """

    sliders = \
        ipywidgets.HBox(
            [
                std_slider(each_name[i], max[i], min=min[i])
                for i in range(len(each_name))
            ],
        )

    return \
        ipywidgets.VBox(
            [ipywidgets.Label(value=name), sliders],
            layout=ipywidgets.Layout(border="1px solid")
        )


def leds_ui():
    """UI for varying standard deviation of LED coordinates.
    """

    return ipywidgets.HBox(
        [
            multi_slider(
                name="Standard dev. for coordinates [mm]:", max=[3.0] * 3
            )
        ]
    )


def camera_extrinsics_ui():
    """UI for varying standard deviation of camera-pose parameters.
    """

    rotation_widget = \
        multi_slider(
            name="Standard dev. for rotation [mrad]:",
            max=[5.0] * 3,
            each_name=["pitch", "yaw", "roll"]
        )
    translation_widget = \
        multi_slider(
            name="Standard dev. for translation [mm]:", max=[1.0, 1.0, 1.0]
        )

    return \
        ipywidgets.HBox(
            [rotation_widget, translation_widget]
        )


def camera_intrinsics_ui():
    pinhole_widget = \
        multi_slider(
            name="Standard dev. for pinhole parameters [pix]:",
            max=[0.5] * 4,
            min=[0.0] * 4,
            each_name=["px", "py", "fx", "fy"]
        )
    distortion_widget = \
        multi_slider(
            name="Standard dev. for distortion parameters [adimensional]:",
            max=[0.2, 0.2, 0.2, 0.2, 0.2],
            min=[0.0] * 5,
            each_name=["cx", "cy", "k0", "k1", "k2"]
        )

    return ipywidgets.HBox([pinhole_widget, distortion_widget])


def features_ui():
    """UI for varying standard deviation of glints, pupil, and limbus points.
    """

    features_widget = \
        [
            multi_slider(
                name="Standard dev. for features [pix]:",
                max=[2.0, 2.0, 2.0],
                each_name=["glint", "pupil", "limbus"]
            )
        ]

    return ipywidgets.HBox(features_widget)


def ui():
    """UI for varying standard deviations of all parameters.
    """

    tab_widget = ipywidgets.Tab()
    names_and_widgets = \
        {
            "LEDs": leds_ui(),
            "Camera extrinsics": camera_extrinsics_ui(),
            "Camera intrinsics": camera_intrinsics_ui(),
            "Image features": features_ui()
        }
    tab_widget.children = [names_and_widgets[key] for key in names_and_widgets]
    for i, key in enumerate(names_and_widgets):
        tab_widget.set_title(i, key)

    default_path = \
        os.path.join(SENSITIVITY_ANALYSIS_DIR, "atlas_1.2_covariances")
    text_widget = \
        ipywidgets.Text(value=default_path, description="Config. files:")

    return ipywidgets.VBox([text_widget, tab_widget])


class CreatePlots():
    """Class for creating plots.
    """

    def __init__(self, data_dictionary):
        """Initialize getter from a ipywidget.Tab object.

        Args:
            ui_widget (ipywidget.VBox): ui object.

            data_dictionary (dict): dictionary containing data required to
            computed covariances.
        """

        self.ui_widget = ui()
        display(self.ui_widget)
        self.text_widget, \
            self.tab_widget = \
            self.ui_widget.children

        self.data_dictionary = data_dictionary

    @staticmethod
    def get_values(ui_object, values=[]):
        """Get values from ui.

        Args:
            ui_object (ipywidgets.Tab): ui object.

            values (list, optional): Should not be touched. Defaults to [].

        Returns:
            list float: list with values of standard deviations for (in order)
            x, y, z camera translation in mm; pitch, yaw, roll camera rotation
            in mrad; x, y coordinates of camera principal point in pix; x, y
            components of camera focal length in pix; x, y adimensional
            coordinates of camera center of distortion; k0, k1, k2 camera
            distortion parameters;  x, y, z coordinates of LEDs in mm; x, y
            coordinates of glints, pupil, and limbus in pix.
        """

        if hasattr(ui_object, "value"):
            val = ui_object.value
            if isinstance(val, float):
                values = values + [val]
        else:
            if hasattr(ui_object, "children"):
                for child in ui_object.children:
                    values = CreatePlots.get_values(child, values)

        return values

    def load_parameters(self):
        """Load covariance-calculator parameters.
        """

        dir_name = self.text_widget.value
        base_name = os.path.split(dir_name)[-1]
        base_name = base_name.replace("_covariances", "")
        full_name = os.path.join(dir_name, base_name)
        self.stds = \
            torch.tensor(self.get_values(self.ui_widget), requires_grad=True)

        leds_file_name = full_name + "_led_covariances.json"
        self.leds_covariance_calculator = \
            LEDsCovarianceCalculator(leds_file_name)
        # LED standard deviations are the first three components.
        self.leds_covariance_calculator.set_stds(self.stds[0:3])

        camera_file_name = full_name + "_camera_covariances.json"
        self.camera_covariance_calculator = \
            CameraCovarianceCalculator(camera_file_name)
        # Camera standard deviations are components 3:18. That's 6 extrinsic
        # parameters (3:10) and 9 intrinsic parameters (4 (10:14) for pinhole
        # model plus 5 (14:18) for distortion in polynomial3K model.)
        self.camera_covariance_calculator.set_stds(self.stds[3:18])

        features_file_name = full_name + "_feature_covariances.json"
        self.features_covariance_calculator = \
            FeaturesCovarianceCalculator(features_file_name)
        # Features are two dimensional.
        self.features_covariance_calculator.set_stds(
            torch.hstack((self.stds[18:21], self.stds[18:21]))
        )

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

        # Number of scenes must match.
        assert (len(d_pose_d_param) == len(d_pose_d_param_indices))
        assert (len(d_pose_d_param) == len(d_shape_d_param))
        assert (len(d_pose_d_param) == len(d_shape_d_param_indices))

        self.load_parameters()

        # This will be a triply nested list. The first level corresponds to a
        # sampled scene. The second level corresponds to a subsystem in the
        # scene's device. The third level corresponds to a gaze direction.
        self.eye_pose_covariance = []

        # Iterate over all eye-tracking scenes (sampled combination of device
        # and user) used to generate the data.
        for scene_idx in range(len(d_shape_d_param)):
            d_shape_d_param_scene = d_shape_d_param[scene_idx]
            d_shape_d_param_scene_indices = d_shape_d_param_indices[scene_idx]
            d_pose_d_param_scene = d_pose_d_param[scene_idx]
            d_pose_d_param_scene_indices = d_pose_d_param_indices[scene_idx]

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

            eye_pose_covariance_scene = []
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

                # For each subsystem, there are multiple gaze directions;
                # iterate over them.
                eye_pose_covariance_subsystem = []
                for gaze_idx in range(len(d_pose_d_param_subsystem)):
                    gaze_d_pose_d_param = d_pose_d_param_subsystem[gaze_idx]
                    gaze_d_pose_d_param_indices = \
                        d_pose_d_param_subsystem_indices[gaze_idx]

                    eye_pose_covariance = \
                        EyePoseCovariance.compute_covariance_standalone(
                            gaze_d_pose_d_param,
                            gaze_d_pose_d_param_indices,
                            False,  # Shouldn't hardcode with_limbs is False.
                            eye_shape_covariance,
                            self.leds_covariance_calculator,
                            self.camera_covariance_calculator,
                            self.features_covariance_calculator
                        )

                    eye_pose_covariance_subsystem = \
                        eye_pose_covariance_subsystem + [eye_pose_covariance, ]

                eye_pose_covariance_scene = \
                    eye_pose_covariance_scene + \
                    [eye_pose_covariance_subsystem, ]

            self.eye_pose_covariance = \
                self.eye_pose_covariance + [eye_pose_covariance_scene, ]

    def compute_covariances_for_KPIs(self):
        """Compute the covariance for the final pupil position and gaze
        direction KPIs.
        """

        # PyTorch derivatives fail! A bug? We need the derivative of the angle
        # to nominal (a single angle in 3D) with respect to the horizontal and
        # vertical components of gaze direction. This exists only as a limit.
        # Intermediate calculations using chain rule yield 0/0, so we do it by
        # hand.
        d_psi_d_gaze = torch.ones((1, 2)) * torch.sqrt(torch.tensor(0.5))

        # Average results across scenes.
        psi_variance_quantile = 0.0
        position_variance_quantile = 0.0
        for cov_scene in self.eye_pose_covariance:
            eye_pose_covariance_user = \
                [
                    cov
                    for cov_gaze in cov_scene
                    for cov in cov_gaze
                ]

            # Compute quantiles across subsystems and gaze directions. For
            # gaze, we've hardcoded the 80th percentile.
            psi_variance_list = [
                d_psi_d_gaze @ cov[:2, :2] @ d_psi_d_gaze.T
                for cov in eye_pose_covariance_user
            ]

            psi_variance = torch.stack(psi_variance_list)
            fov_gaze_quantile = 0.8  # Should not hardcode this.
            psi_variance_quantile = \
                psi_variance_quantile + \
                torch.quantile(psi_variance, fov_gaze_quantile)

            # For position, we've hardcoded the 50th percentile.
            position_covariance = \
                torch.hstack(
                    [
                        torch.diag(cov[2:, 2:]).view((3, 1))
                        for cov in eye_pose_covariance_user
                    ]
                )
            position_quantile = 0.5  # Should not hardcode this.
            position_variance_quantile = \
                position_variance_quantile + \
                torch.quantile(position_covariance, position_quantile, dim=1)

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

    def plot_results(self, gaze_kpi_deg=0.75, pose_kpi_mm=0.75):
        """Visualize results of experiment
        """

        plt.close("all")

        self.compute_covariances()
        self.compute_covariances_for_KPIs()
        self.compute_contributions_to_KPIs()

        # These are the values that must be stacked in the bar plot. The rows
        # correspond to the KPI parameters (gaze, and x, y, and z eye
        # position), and the columns correspond to the contributions of LEDs
        # (0:3), camera intrinsics (3:6 for location, 6:9 for orientation),
        # camera extrinsics (9:13 for pinhole, 13:18 for distortion), and image
        # features (18:21 for glints, pupil, and limbus).
        gaze_error_std = self.d_psi_std_d_stds
        normalizer = torch.sqrt(self.psi_variance) / gaze_error_std.sum()
        # Gaze is 1 by N, where N is the number of std components
        gaze_error_std = (gaze_error_std * normalizer).view(1, -1)

        position_error_std = self.d_position_std_d_stds
        normalizer = \
            torch.sqrt(self.position_variance) / \
            (position_error_std).sum(dim=1)
        # Position is 3 (x,y,z) by N, where N is the number of std components.
        position_error_std = position_error_std * normalizer.view(-1, 1)

        # Plot standard deviations for gaze and position. Group results into
        # contributions from LEDs, camera extrinsics, camera intrinsics, and
        # features.
        fig, axs = plt.subplots(1, 3, figsize=(8, 3), width_ratios=(1, 3, 1))

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
        y_labels = ("$\sigma$ [deg]", "$\sigma$ [mm]")
        kpi_line = (gaze_kpi_deg, pose_kpi_mm)
        suptitle = "Standard Deviation of Errors"

        for i in range(len(error_std)):
            std_i = error_std[i].clone().detach()

            x_i = x[i]
            y_i0 = torch.zeros(len(x_i))

            # Iterate over LED, cam extr., cam intr., and image feature
            # contribs. stds coefficients are:
            #
            # 0:3 for LEDs
            # 3:6, 6:9, for camera location and orientation
            # 9:13, and 13:18 for camera pinhole and distortion
            # 18:21 for image features.
            ax = axs[i]
            h = []
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
                h.append(
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
            h.append(ax.axhline(kpi_line[i], c="red", ls="--", lw=3))

            ax.set_title(titles[i])
            ax.set_ylabel(y_labels[i])

        # Create legend
        fig.legend(h, label, loc="center right", ncol=1)

        plt.suptitle(suptitle)
        plt.tight_layout()

        plt.show()

        # Isolate impact of components.
        # LEDs, cam extr., cam intr., image features
        width = 0.8
        x = (("$\\theta$",), ("$x$", "$y$", "$z$"))
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
                # 0:3 for LEDs
                # 3:6, 6:9, for camera location and orientation
                # 9:13, and 13:18 for camera pinhole and distortion
                # 18:21 for image features.
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
                ax.set_title(titles[i])
                ax.set_ylabel(y_labels[i])

            # Create legend
            fig.legend(h, label_k, loc="center right", ncol=1)

            plt.suptitle(suptitle + ". Contribution from " + label[k])
            plt.tight_layout()

            plt.show()
