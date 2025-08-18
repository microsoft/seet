"""utils.py

Utilities for generating data.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import IPython.display as display
import ipywidgets
import kiruna
import os


def get_configuration_files(scene_name):
    """Get files for configuring scene and sampler for different Kiruna scenes.

    Args:
        scene_name (str): scene name. Should be one of "palau", "atlas_1.2",
        "sydney", or "tasman".
    """

    scene_file_name = \
        os.path.join(
            kiruna.scene.SCENE_DIR,
            scene_name + "_scene/" + scene_name + "_scene.json"
        )
    sampler_file_name = \
        os.path.join(
            kiruna.sampler.SAMPLER_DIR,
            r"default_sampler/default_scene_sampler.json"
        )

    # Sydney needs to construct scene with parameters
    if scene_name == "sydney":
        device_blob = \
            os.path.join(
                kiruna.device.DEVICE_DIR,
                r"sydney_device/sydney_device_calibration.json"
            )
        eye_blobs = []

        for eye in ("left", "right"):
            eye_file = r"sydney_user/{:s}_eye_calibration.json".format(eye)
            eye_blob = os.path.join(kiruna.user.USER_DIR, eye_file)
            eye_blobs = eye_blobs + [eye_blob, ]

        scene_file_name = \
            kiruna.scene.SceneModel.create_kiruna_scene_file(
                eye_blobs[0], eye_blobs[1], device_blob
            )

    # Palau has a special sampler.
    elif scene_name == "palau":
        sampler_file_name = \
            os.path.join(
                kiruna.sampler.SAMPLER_DIR,
                r"palau_sampler/palau_scene_sampler.json"
            )

    # Atlas 1.2 has a special sampler
    elif scene_name == "atlas_1.2":
        sampler_file_name = \
            os.path.join(
                kiruna.sampler.SAMPLER_DIR,
                r"atlas_1.2_sampler/atlas_1.2_scene_sampler.json"
            )

    return scene_file_name, sampler_file_name


def get_device(device_list=["atlas_1.2", "palau", "sydney", "tasman", "p47", "p47_POC", "test", "p53_wearable"]):
    """get_device.

    Select the Kiruna scene from a dropdown list

    Args:
        device_list (list, optional): List of valid devices. Defaults to
        ["atlas_1.2", "palau", "sydney", "tasman"].
    """

    dropdown_widget = \
        ipywidgets.RadioButtons(
            options=["atlas_1.2", "palau", "sydney", "tasman", "p47", "p47_POC", "test", "p53_wearable"],
            value="p47",
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
