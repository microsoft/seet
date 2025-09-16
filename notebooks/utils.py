"""utils.py

Utilities for generating data.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import IPython.display as display
import ipywidgets
import seet
import os


def get_configuration_files(scene_name):
    """Get files for configuring scene and sampler for different Kiruna scenes.

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

    Select the Kiruna scene from a dropdown list

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
