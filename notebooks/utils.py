"""utils.py

Utilities for generating data.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import IPython.display as display
import ipywidgets
import seet
import os
import pandas
import torch

def get_configuration_files(scene_name):
    """Get files for configuring scene and sampler for different SEET scenes.

    Args:
        scene_name (str): scene name.
    """

    scene_file_name = \
        os.path.join(
            seet.scene.SCENE_DIR,
            scene_name + "_scene/" + scene_name + "_scene.json"
        )
    sampler_file_name = \
        os.path.join(
            seet.sampler.SAMPLER_DIR,
            r"default_sampler/default_scene_sampler.json"
        )

    return scene_file_name, sampler_file_name


def get_device(device_list=["default"]):
    """get_device.

    Select the SEET scene from a dropdown list

    Args:
        device_list (list, optional): List of valid devices. Defaults to
        ["default"].
    """

    dropdown_widget = \
        ipywidgets.RadioButtons(
            options=["default"],
            value="default",
            description="Device:",
            disabled=False
        )
    return dropdown_widget


def get_path(scene_name=".", show=False, description="Results:"):
    """get_path.

    Get the output directory for results
    """
    text_widget = \
        ipywidgets.Text(
            value=".\\results\\" + scene_name,
            placeholder="Default is '.'",
            description=description,
            disabled=False
        )

    if show:
        display.display(text_widget)

    return text_widget


def get_experiment_info(save_results=True):
    dropdown_widget = get_device()
    display.display(dropdown_widget)

    if save_results:
        text_widget = \
            get_path(
                scene_name=dropdown_widget.value, show=False  # type: ignore
            )

        def on_value_change(change):
            old_value = text_widget.value
            paths = str(old_value).split(change["old"])
            text_widget.value = paths[0] + change["new"] + paths[1]

        dropdown_widget.observe(on_value_change, names="value")  # type: ignore
        display.display(text_widget)

        return dropdown_widget, text_widget
    else:
        return dropdown_widget


def generate_data_for_visibility_analysis(scene_sampler, gaze_grid=[5, 4]):
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
    header_subsystem = [
        "Subsystem",
    ]
    header_grid_angle = ["Horiz. angle", "Vert. angle"]
    # This assumes that the subsystems have the same number of LEDs.
    num_leds = scene_sampler.scene.device.subsystems[0].led_set.num
    header_LEDs = ["LED {:02d}".format(i) for i in range(1, num_leds + 1)]

    #######################################################################
    # Device + user data.
    header_camera_center_in_pupil = [
        "Camera center in pupil {:s}".format(ax) for ax in ["x", "y", "z"]
    ]
    header_delta_eye_relief = [
        "Delta eye relief",
    ]
    header_scene_index = [
        "Scene index",
    ]

    #######################################################################
    # User-only data.
    header_gaze_direction = ["Gaze {:s}".format(ax) for ax in ["x", "y", "z"]]
    header_IPD = [
        "IPD",
    ]

    #######################################################################
    # Putting it all together
    header = (
        header_subsystem
        + header_grid_angle
        + header_LEDs
        + header_camera_center_in_pupil
        + header_delta_eye_relief
        + header_scene_index
        + header_gaze_direction
        + header_IPD
    )

    num_subsystems = len(scene_sampler.scene.device.subsystems)
    data = []
    for scene_index, et_scene in enumerate(scene_sampler.generate_samples()):
        for subsystem_index in range(num_subsystems):
            ###############################################################
            # Device-only data.
            # Subsystem data.
            row_subsystem = [
                subsystem_index,
            ]

            # Grid angle data.
            if gaze_grid[0] != 1 or gaze_grid[1] != 1:
                fov_range_deg = scene_sampler.scene.device.display_fov / 2
                h_fov_range_deg = torch.linspace(
                    -fov_range_deg[0], fov_range_deg[0], gaze_grid[0]
                )
                v_fov_range_deg = torch.linspace(
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
                        angles_deg = torch.stack(
                            (h_fov_range_deg[hi], v_fov_range_deg[vi])
                        )
                        eye.rotate_from_gaze_angles_inParent(angles_deg)
                    else:
                        angles_deg = torch.zeros(2)

                    row_grid_angle = [*angles_deg.clone().detach().numpy()]

                    #######################################################
                    # Device plus user data.
                    # Scene (device + user) index.

                    # LED-visibility data.
                    glints_inCamera = et_scene.generate_glints_inOther(
                        other_node=camera,
                        subsystem_index=subsystem_index,
                        camera_index=camera_index,
                    )

                    row_LEDs = [int(g is not None) for g in glints_inCamera]

                    # Camera center in pupil.
                    transform_toPupil_fromCamera = (
                        camera.get_transform_toOther_fromSelf(eye.pupil)
                    )
                    optical_center_in_pupil = transform_toPupil_fromCamera.transform(
                        torch.zeros(3)
                    )
                    row_camera_center_in_pupil = [
                        *optical_center_in_pupil.clone().detach().numpy()
                    ]

                    # Eye-relief data.
                    eye_relief_plane = subsystem.eye_relief_plane
                    cornea_apex_inPlane = eye.get_cornea_apex_inOther(eye_relief_plane)

                    delta_eye_relief = (
                        -1
                        * eye_relief_plane.compute_signed_distance_to_point_inPlane(
                            cornea_apex_inPlane
                        )
                        .clone()
                        .detach()
                        .numpy()
                    )

                    row_delta_eye_relief = [
                        delta_eye_relief,
                    ]

                    # Keeping track of the samples.
                    row_scene_index = [
                        scene_index,
                    ]

                    #######################################################
                    # User-only data.
                    # Gaze-direction data.
                    gaze_direction_inScene = eye.get_gaze_direction_inOther(et_scene)
                    row_gaze_direction = [
                        *gaze_direction_inScene.clone().detach().numpy()
                    ]

                    # IPD data.
                    IPD_data = et_scene.user.compute_IPD()
                    row_IPD = [
                        IPD_data.clone().detach().numpy(),
                    ]

                    #######################################################
                    # Putting it all together.
                    row = (
                        row_subsystem
                        + row_grid_angle
                        + row_LEDs
                        + row_camera_center_in_pupil
                        + row_delta_eye_relief
                        + row_scene_index
                        + row_gaze_direction
                        + row_IPD
                    )

                    data = data + [
                        row,
                    ]

                    if rotate:
                        eye.unrotate_from_gaze_angles_inParent(angles_deg)

    return pandas.DataFrame(data, columns=header)


def generate_data_for_iris_analysis(scene_sampler, num_angles=10, num_radii=10):
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
        "Area in image [pix^2]",
    ]

    # Parameters to compute area element
    d_theta = torch.tensor(2 * torch.pi / num_angles, requires_grad=True)

    num_subsystems = len(scene_sampler.scene.device.subsystems)
    data = []
    for et_scene in scene_sampler.generate_samples():
        for subsystem_index in range(num_subsystems):
            camera = et_scene.device.subsystems[subsystem_index].cameras[0]
            eye = et_scene.user.eyes[subsystem_index]
            cornea = eye.cornea
            limbus = eye.limbus

            ###############################################################
            # Sample iris in polar coordinates.
            optical_center_inCornea = camera.get_optical_center_inOther(cornea)
            transform_toCornea_fromLimbus = limbus.get_transform_toOther_fromSelf(
                cornea
            )

            d_r = limbus.radius / num_radii
            # TODO: This is ugly! We need to create a function T0() that
            # generates the detached value of the zero tensor.
            area_total_mm = seet.core.T0.clone().detach()
            area_mm = seet.core.T0.clone().detach()
            area_pix = seet.core.T0.clone().detach()
            for i in range(num_angles):
                theta = i * d_theta
                for j in range(num_radii):
                    r = j * d_r
                    x = r * torch.cos(theta)
                    y = r * torch.sin(theta)
                    point_in2DIris = torch.hstack((x, y))

                    point_in3DIris = torch.hstack((point_in2DIris, seet.core.T0))
                    iris_point_inCornea = transform_toCornea_fromLimbus.transform(
                        point_in3DIris
                    )

                    # Make sure point is inside cornea.
                    if (
                        cornea.compute_algebraic_distance_inEllipsoid(
                            iris_point_inCornea
                        )
                        >= 0
                    ):
                        continue

                    d_x_d_y = r * d_r * d_theta
                    area_total_mm += d_x_d_y

                    refraction_point_inCornea = (
                        cornea.compute_refraction_point_inEllipsoid(
                            optical_center_inCornea,
                            iris_point_inCornea,
                            eta_at_destination=cornea.refractive_index,
                        )
                    )
                    if refraction_point_inCornea is None:
                        continue

                    # Check occlusion by device occluder, if one is
                    # present.
                    unit_list_refraction_point_inCornea = et_scene.device.subsystems[
                        subsystem_index
                    ].apply_occluder_inOther(
                        cornea,
                        [
                            refraction_point_inCornea,
                        ],  # list
                        reference_point_inOther=optical_center_inCornea,
                    )
                    # Input is list of points, and so is output.
                    refraction_point_inCornea = unit_list_refraction_point_inCornea[0]

                    if refraction_point_inCornea is not None:
                        area_mm += d_x_d_y

                        refraction_point_inPixels = camera.project_toPixels_fromOther(
                            refraction_point_inCornea, cornea
                        )

                        d_inPixels_d_in2DIris = seet.core.compute_auto_jacobian_from_tensors(
                            refraction_point_inPixels, point_in2DIris
                        )

                        area_pix += d_x_d_y * torch.abs(
                            torch.linalg.det(d_inPixels_d_in2DIris)
                        )

            percentage = (100 * area_mm / area_total_mm).item()
            data = data + [
                [subsystem_index, percentage, area_mm.item(), area_pix.item()]
            ]

    return pandas.DataFrame(data, columns=header)
