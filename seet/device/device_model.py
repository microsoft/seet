"""device_model.py.

Device class. A device is a set of SubsystemModel objects, typically two.
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
from seet.device import device_configs
from seet.device import subsystem_model
import json
import os
from pathlib import PurePath
import torch


class DeviceModel(core.Node):
    """DeviceModel.

    Class for an eye-tracking device, consisting of a number of eye-tracking
    subsystem.
    """

    def __init__(
        self,
        scene,
        transform_toScene_fromDevice,
        name="",
        parameter_file_name=os.path.join(
            device_configs.DEVICE_DIR,
            r"default_device/default_device_model.json"
        ),
        requires_grad=False
    ):
        """
        Initialize ET device.

        Args:
            scene (Node): parent node of the device. Typically, this will be a
            Scene object.

            transform_toScene_fromDevice (SE3): transformation from coordinate
            system of device to that of scene.

            name (str, optional): name of object.

            parameter_file_name (str, optional): name of parameter file with
            configuration of device. Defaults to
            os.path.join(device_configs.DEVICE_MODELS_DIR,
            "default_device/default_device_model.json").

            requires_grad (bool, optional): if not None, overwrites value in
            parameter file.

        Returns:
            DeviceModel: DeviceModel object.
        """
        self.parameter_file_name = parameter_file_name

        with core.Node.open(self.parameter_file_name) as parameter_file_stream:
            device_parameters = json.load(parameter_file_stream)

        # Name of the ET device.
        if "name" in device_parameters.keys():
            name = device_parameters["name"]
        else:
            name = name

        super().__init__(
            scene,
            transform_toScene_fromDevice,
            name=name,
            requires_grad=requires_grad
        )

        self.display_fov = \
            torch.tensor(device_parameters["display angular fov"])

        self.subsystems = []
        for subsystem_model_dict in device_parameters["et subsystems"]:
            val = subsystem_model_dict["subsystem"]
            if isinstance(val, int):
                # The ET subsystem is just a mirror of another ET subsystem in
                # the coordinate system of the device.
                subsystem = \
                    subsystem_model.SubsystemModel.mirror(self.subsystems[val])
            else:
                if isinstance(val, str):
                    DEVICE_DIR = subsystem_model_dict.get("path")
                    if DEVICE_DIR is None:
                        DEVICE_DIR = device_configs.DEVICE_DIR
                    subsystem_model_file_name = \
                        os.path.join(DEVICE_DIR, val)
                else:
                    subsystem_model_file_name = val

                if requires_grad is None or requires_grad is False:
                    if "requires grad" in subsystem_model_dict.keys() and \
                            subsystem_model_dict["requires grad"]:
                        local_requires_grad = True
                    else:
                        local_requires_grad = False
                else:
                    local_requires_grad = requires_grad

                transform_matrix = \
                    torch.tensor(
                        subsystem_model_dict["extrinsics"],
                        requires_grad=local_requires_grad
                    )

                transform_toDevice_fromSubsystemModel = \
                    core.SE3(transform_matrix)

                subsystem = \
                    subsystem_model.SubsystemModel(
                        self,
                        transform_toDevice_fromSubsystemModel,
                        parameter_file_name=subsystem_model_file_name,
                        requires_grad=local_requires_grad
                    )

            self.subsystems = self.subsystems + [subsystem, ]

    def get_kwargs(self):
        """Augment base-class method.

        Returns:
            dict: dictionary with keyword values.
        """
        base_kwargs = super().get_kwargs()
        this_kwargs = {"parameter_file_name": self.parameter_file_name}

        return {**base_kwargs, **this_kwargs}

    @staticmethod
    def create_secrets_device_file(
        calibration_blob,
        device_save_path=None,
        requires_grad=False
    ):
        """
        Create a device file from a calibration blob.

        Args:
            calibration_blob (str or dict): path and name of json calibration
            blob. It may also be a dictionary with the calibration blob.

            device_save_path (bool, optional): path where converted
            device-calibration blob will be saved. Defaults to None, in which
            case the converted file is not saved.

            requires_grad (bool, optional): whether parameters of the scene are
            differentiable. Defaults to False.

        Returns:
            DeviceModel: device model with parameters defined by input
            calibration blob.
        """
        # Read in the device data.
        with core.Node.open(calibration_blob) as calibration_blob_stream:
            calibration_blob_data = json.load(calibration_blob_stream)

        calibration_information = \
            calibration_blob_data["CalibrationInformation"]

        # We are assuming that there are two cameras, one per subsystem.
        left_subsystem_model_dict = dict()
        right_subsystem_model_dict = dict()

        # Device calibration has units of meters, but secrets uses mm.
        scale_toMM_fromM = 1000.0

        #######################################################################
        # Eyebox field of view. It is different for each eye, but we use a
        # single one by averaging. To compute field of view, we use the
        # intrinsic parameters A of the eyeboxes, as shown below:
        #
        #     [fx,  0, px]
        # A = [ 0, fy, py]
        #     [ 0,  0,  1]
        #
        # +0.5
        #   *\___
        #   |    \___
        #   |        \___
        #   | theta/2  / \___
        #   |         |      \___
        #   +---------+-----------*
        #   |      f
        #   |
        #   |
        #   |
        #   *
        # -0.5
        #
        # From the above, we see that
        #
        # theta = 2 * arctan(0.5 / f).
        #
        # This applies to both x and y, by replacing f with fx and fy.

        displays = calibration_information["Displays"]
        affine_left = torch.tensor(displays[0]["Affine"])
        affine_right = torch.tensor(displays[1]["Affine"])
        theta_x_rad = \
            (
                torch.atan(0.5 / affine_left[0]) +
                torch.atan(0.5 / affine_right[0])
            )
        theta_y_rad = \
            (
                torch.atan(0.5 / affine_left[4]) +
                torch.atan(0.5 / affine_right[4])
            )
        display_fov_deg = \
            core.rad_to_deg(
                torch.stack([theta_x_rad, theta_y_rad])
            ).tolist()

        #######################################################################
        # Cameras section of subsystem.
        cameras = calibration_information["Cameras"]
        for camera_dict in cameras:
            purpose = camera_dict["Purpose"]
            if purpose != "CALIBRATION_CameraPurposeEyeTracking":
                # Ignore non-eye-tracking cameras.
                continue

            # Get the intrinsics parameters of the camera.
            resolution = \
                torch.tensor(
                    [
                        1.0 * camera_dict["SensorWidth"],  # Float, not int
                        1.0 * camera_dict["SensorHeight"]  # Float, not int
                    ],
                    requires_grad=requires_grad
                )
            # Model parameters are, in this order.
            #
            # 0: px, normalized x coordinate of principal point.
            # 1: py, normalized y coordinate of principal point.
            # 2: fx, normalized horizontal focal length.
            # 3: fy, normalized vertical focal length.
            # 4: cx, normalized x coordinate of distortion center.
            # 5: cy, normalized y coordinate of distortion center.
            # 6, 7, 9: k0, k1, k2 components of polynomial 3K distortion model.
            model_parameters = \
                torch.tensor(
                    camera_dict["Intrinsics"]["ModelParameters"],
                    requires_grad=requires_grad
                )

            # De-normalized principal point.
            principal_point = resolution * model_parameters[:2]

            # De-normalized focal lengths.
            focal_lengths = resolution * model_parameters[2:4]

            # Distortion parameters.
            distortion_center = model_parameters[4:6]
            distortion_coefficients = model_parameters[6:9]

            # Create output in SECRETS format
            intrinsics_dict = dict()
            intrinsics_dict["pinhole parameters"] = \
                {
                    "requires grad": requires_grad,
                    "focal lengths": focal_lengths.clone().detach().tolist(),
                    "principal point":
                        principal_point.clone().detach().tolist(),
                    "resolution": resolution.clone().detach().tolist()
            }
            intrinsics_dict["distortion parameters"] = \
                {
                    "requires grad": requires_grad,
                    "distortion center":
                        distortion_center.clone().detach().tolist(),
                    "distortion coefficients":
                        distortion_coefficients.clone().detach().tolist()
            }

            # Get the extrinsic parameters of the camera. These are in the
            # coordinate system of the 0 head-tracking camera, and should be
            # converted to a different coordinate system afterwards.
            R_toCamera_fromDevice = \
                torch.tensor(
                    camera_dict["Rt"]["Rotation"], requires_grad=requires_grad
                ).view((3, 3))
            # The rotation matrices in the calibration blob have a weird 90 deg
            # rotation around z that needs to be removed
            R_toCamera_fromDevice = \
                R_toCamera_fromDevice @ core.rotation_around_z(90.0)

            t_toCamera_fromDevice = \
                scale_toMM_fromM * \
                torch.tensor(
                    camera_dict["Rt"]["Translation"],
                    requires_grad=requires_grad
                ).view((3, 1))

            transform_matrix_toCamera_fromDevice = \
                torch.vstack(
                    (
                        torch.hstack(
                            (R_toCamera_fromDevice, t_toCamera_fromDevice)
                        ),
                        torch.tensor([0.0, 0.0, 0.0, 1.0])
                    )
                )

            extrinsics = \
                transform_matrix_toCamera_fromDevice.clone().detach().tolist()

            secrets_camera = \
                {
                    "requires grad": requires_grad,
                    "extrinsics": extrinsics,
                    "extrinsics type": "direct",
                    "path": None,
                    "intrinsics": intrinsics_dict
                }
            if camera_dict["Location"] == "CALIBRATION_CameraLocationET0":
                left_subsystem_model_dict["cameras"] = [secrets_camera, ]
            else:
                right_subsystem_model_dict["cameras"] = [secrets_camera, ]

        #######################################################################
        # LED section of subsystem.
        identity = \
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0]
            ]
        secrets_leds = dict()
        secrets_leds["requires grad"] = requires_grad
        left_coordinates = []
        right_coordinates = []
        for led in calibration_information["EyeTrackingLeds"]:
            # Note the rotation around z by 90 degrees.
            coordinates = [[led["Y"], -led["X"], led["Z"]], ]
            coordinates = scale_toMM_fromM * torch.tensor(coordinates)
            coordinates = coordinates.tolist()

            if led["AssignedEye"] == "CALIBRATION_DisplayEyeLeft":
                left_coordinates = left_coordinates + coordinates
            else:
                right_coordinates = right_coordinates + coordinates

        for idx, pair in enumerate(
            (
                (left_subsystem_model_dict, left_coordinates),
                (right_subsystem_model_dict, right_coordinates)
            )
        ):
            subsystem, coordinates = pair
            subsystem["eye indices"] = [idx]  # One eye per subsystem.
            subsystem["LEDs"] = dict()
            subsystem["LEDs"]["requires grad"] = requires_grad
            subsystem["LEDs"]["extrinsics"] = identity
            subsystem["LEDs"]["path"] = None
            subsystem["LEDs"]["coordinates"] = \
                {
                    "name": "LEDs created from " + calibration_blob,
                    "requires grad": requires_grad,
                    "coordinates": coordinates,
                }

        secrets_device_dict = \
            {
                "name": "Device created from " + calibration_blob,
                "display angular fov": display_fov_deg,
                "et subsystems": [
                    {
                        "path": None,
                        "subsystem": left_subsystem_model_dict,
                        "requires grad": requires_grad,
                        "extrinsics": identity
                    },
                    {
                        "path": None,
                        "subsystem": right_subsystem_model_dict,
                        "requires grad": requires_grad,
                        "extrinsics": identity
                    },
                ]
            }

        # Save SECRETS device configuration file, if required.
        if device_save_path is not None:
            base_name = PurePath(device_save_path).parts[-1]
            device_json_file_name = base_name + "_device_model.json"
            file_name = os.path.join(device_save_path, device_json_file_name)
            os.makedirs(device_save_path, exist_ok=True)
            with open(file_name, "w") as file_stream:
                json.dump(secrets_device_dict, file_stream)

        return secrets_device_dict

    @classmethod
    def create_device_from_real_data(
        cls,
        scene,
        transform_toScene_fromDevice,
        calibration_blob,
        device_save_path=None,
        requires_grad=False
    ):
        """
        Create a device from a calibration blob, and either a left-eye or
        right-eye user-calibration blob (or both). At least one of
        left_calib_blob and right_calib_blob must be provided.

        The device is "floating," in that it is not attached to any scene.

        Args:
            scene (Node): parent node of the device. Typically, this will be a
            Scene object.

            transform_toScene_fromDevice (SE3): transformation from coordinate
            system of device to that of scene.

            calibration_blob (str or dict): path and name of json calibration
            blob. It may also be a dictionary with the calibration blob.

            device_save_path (bool, optional): path where converted
            device-calibration blob will be saved. Defaults to None, in which
            case the converted file is not saved.

            requires_grad (bool, optional): whether parameters of the scene are
            differentiable. Defaults to False.

        Returns:
            DeviceModel: device model with parameters defined by input
            calibration blob.
        """
        secrets_device_blob = \
            DeviceModel.create_secrets_device_file(
                calibration_blob,
                device_save_path=device_save_path,
                requires_grad=requires_grad
            )

        return \
            cls(
                scene,
                transform_toScene_fromDevice,
                parameter_file_name=secrets_device_blob  # type: ignore
            )

    def sample_fov(self, grid):
        """Sample the device's display field of view uniformly.

        Args:
            grid (list of int): list [M, N] for M and N integers corresponding
            to the number of horizontal and vertical samples in the devices's
            field of view.

        Returns:
            list of torch.Tensor: list with size M * N where each entry is a
            (2,) torch.Tensor corresponding to a horizontal and vertical gaze
            direction in degrees.
        """

        display_fov = self.display_fov
        h_angles = \
            torch.linspace(
                -display_fov[0] / 2, display_fov[0] / 2, grid[0]
            )
        v_angles = \
            torch.linspace(
                -display_fov[1] / 2, display_fov[1] / 2, grid[0]
            )

        return [torch.hstack((h, v)) for h in h_angles for v in v_angles]
