"""subsystem_model.py.

Class representing an eye-tracking (ET) subsystem. An ET subsystem is a set
of LEDs a and cameras trained at a user's left or right eye (but not both).
"""


__author__ = "Paulo R. S. Mendonca (padossa@microsoft.com)"


import seet.core as core
import seet.device as device
import json
import os
import torch


class SubsystemModel(core.Node):
    """SubsystemModel.

    Class for an eye-tracking subsystem, consisting of a number of cameras
    and leds, and associated to one or more user's eyes.
    """

    def __init__(
        self,
        et_device,
        transform_toDevice_fromSubsystemModel,
        name="",
        parameter_file_name=os.path.join(
            device.DEVICE_DIR,
            r"default_device/default_left_subsystem_model.json"
        ),
        requires_grad=False
    ):
        """
        Initialize ET subsystem.

        Args:
            et_device (Node): parent node of ET subsystem. Typically, this will
            correspond to the device the pose graph.

            transform_toDevice_fromSubsystemModel (SE3): SE3 object describing
            the transformation from the coordinate system of the ET subsystem
            to that of its parent. Typically, this will be the identity element
            of SE3.

            name (str, optional): name of object.

            parameter_file_name (str, optional): parameter file with
            configuration of ET subsystem. Defaults to
            os.path.join(device.DEVICE_MODELS_DIR,
            "default_device/default_left_subsystem_model.json").

            requires_grad (bool): if True, camera, LED, and (if present)
            occluder extrinsic parameters of ET subsystem are differentiable.
            Defaults to False.

        Returns:
            SubsystemModel: SubsystemModel object.
        """
        self.parameter_file_name = parameter_file_name

        with core.Node.open(parameter_file_name) as parameter_file_stream:
            subsystem_model_parameters = json.load(parameter_file_stream)

        # Name the SubsystemModel.
        if "name" in subsystem_model_parameters.keys():
            name = subsystem_model_parameters["name"]
        else:
            name = name

        super().__init__(
            et_device,
            transform_toDevice_fromSubsystemModel,
            name=name,
            requires_grad=requires_grad
        )

        # These are the indices of user's eyes (typically either 0, 1; highly
        # unlikely to be both), with which the ET subsystem is associated.
        self.eye_indices = subsystem_model_parameters["eye indices"]

        # Most ET subsystems will be associated with a single eye.
        self.eye_index = self.eye_indices[0]

        #######################################################################
        # Cameras
        self.cameras = []
        for camera_dict in subsystem_model_parameters["cameras"]:
            if requires_grad is None or requires_grad is False:
                if "requires grad" in camera_dict.keys():
                    camera_requires_grad = camera_dict["requires grad"]
                else:
                    camera_requires_grad = False
            else:
                camera_requires_grad = requires_grad

            transform_matrix = \
                torch.tensor(
                    camera_dict["extrinsics"],
                    requires_grad=camera_requires_grad
                )

            # Default extrinsics from CAD models are TO camera FROM device.
            if "extrinsics type" not in camera_dict.keys() or \
                    camera_dict["extrinsics type"] != "inverse":
                # This is a bit awkward, as if the type is *not* inverse, we
                # invert it here.
                transform_matrix = torch.linalg.pinv(transform_matrix)

            transform_toSubsystemModel_fromCamera = core.SE3(transform_matrix)
            if isinstance(camera_dict["intrinsics"], str):
                camera_file_name = \
                    os.path.join(
                        device.DEVICE_DIR, camera_dict["intrinsics"]
                    )
            else:
                camera_file_name = camera_dict["intrinsics"]

            self.cameras = \
                self.cameras + \
                [
                    device.Polynomial3KCamera(
                        self,
                        transform_toSubsystemModel_fromCamera,
                        parameter_file_name=camera_file_name,
                        requires_grad=camera_requires_grad
                    )
                ]

        # Many systems will have a single/default camera per eye.
        self.camera = self.cameras[0]

        #######################################################################
        # LEDs
        leds_dict = subsystem_model_parameters["LEDs"]
        path = leds_dict.get("path", device.DEVICE_DIR)
        if path is None:
            path = device.DEVICE_DIR

        if requires_grad is None:
            if "requires grad" in leds_dict.keys():
                leds_requires_grad = leds_dict["requires grad"]
            else:
                leds_requires_grad = False
        else:
            leds_requires_grad = requires_grad

        transform_matrix = \
            torch.tensor(
                leds_dict["extrinsics"],
                requires_grad=leds_requires_grad
            )
        transform_toSubsystemModel_fromLEDs = core.SE3(transform_matrix)

        coordinates = leds_dict["coordinates"]
        if isinstance(coordinates, str):
            leds_file_name = os.path.join(path, coordinates)
        else:
            leds_file_name = coordinates

        self.led_set = \
            device.LEDs(
                self,
                transform_toSubsystemModel_fromLEDs,
                parameter_file_name=leds_file_name,
                requires_grad=leds_requires_grad
            )

        #######################################################################
        # Occluders, if any.
        occluder_dict = subsystem_model_parameters.get("occluder", None)
        apply = False
        if occluder_dict is not None:
            apply = occluder_dict.get("apply", True)

        if apply:
            if requires_grad is None:
                if "requires grad" in occluder_dict.keys():
                    occluder_requires_grad = occluder_dict["requires grad"]
                else:
                    occluder_requires_grad = False
            else:
                occluder_requires_grad = requires_grad

            transform_matrix = \
                torch.tensor(
                    occluder_dict["extrinsics"],
                    requires_grad=occluder_requires_grad
                )
            transform_toSubsystemModel_fromLEDs = core.SE3(transform_matrix)

            coordinates = occluder_dict["coordinates"]
            if isinstance(coordinates, str):
                occluder_file_name = \
                    os.path.join(
                        device.DEVICE_DIR, occluder_dict["coordinates"]
                    )
            else:
                occluder_file_name = coordinates

            self.occluder = \
                device.Occluder(
                    self,
                    transform_toSubsystemModel_fromLEDs,
                    parameter_file_name=occluder_file_name
                )
        else:
            self.occluder = None

        #######################################################################
        # We start without knowledge of the user's eye relief.
        self.eye_relief_plane = None

    def get_kwargs(self):
        """Augment base-class method.

        Returns:
            dict: dictionary with keyword values.
        """
        base_kwargs = super().get_kwargs()
        this_kwargs = {"parameter_file_name": self.parameter_file_name}

        return {**base_kwargs, **this_kwargs}

    @classmethod
    def mirror(cls, this_node):
        """mirror.

        We must overload the mirroring of a subsystem because the subsystem
        contains information about the eye to which the subsystem is
        associated.

        Args:
            this_node (SubsystemModel): subsystem to be mirrored with respect
            to x = 0 plane.
        """
        new_node = \
            super(SubsystemModel, cls).mirror(this_node)

        # Fix the eyes associated with the subsystem.
        N = len(new_node.eye_indices)
        if N == 1:
            new_node.eye_indices[0] = 1 - new_node.eye_indices[0]
        else:
            for i in range(N):
                new_node.eye_indices[i] = N - new_node.eye_indices[i] - 1

        new_node.eye_index = new_node.eye_indices[0]

        return new_node

    def set_eye_relief_plane(self, plane):
        """set_eye_relief_plane.

        Add an eye relief plane to the device. This can be done only when we
        have a user available.

        Args:
            plane (primitives.Plane): a node plane, which is added to the
            coordinate system of the subsystem.
        """
        self.eye_relief_plane = plane
        self.add_child(plane)

    def apply_occluder_inSubsystem(
        self,
        points_inSubsystem,
        reference_point_inSubsystem,
        set_to_none=True
    ):
        """
        Filter out from input list the points the points whose rays pointing
        towards the reference point are occluded by the occluder of the
        subsystem.

        Args:
            points_inSubsystem (list of torch.Tensors): list of (3,) torch
            Tensors whose visibility is to be evaluated. May be None, otherwise
            the points are in the coordinate system of the subsystem.

            reference_point_inSubsystem (torch.Tensor): (3,) torch tensor
            endpoint of ray whose visibility with respect to occluder is being
            tested.

            set_to_none (bool, optional): if True, occluded points will be set
            to None. Otherwise, they will be removed from the list. Defaults to
            True.
        """
        # If subsystem does not have an occluder, return the input list.
        if self.occluder is None:
            return points_inSubsystem

        visible_points_inSubsystem = []
        for single_point_inSubsystem in points_inSubsystem:
            if single_point_inSubsystem is None:
                single_visible_point_inSubsystem = None
            elif self.occluder.is_ray_occluded_inParent(
                reference_point_inSubsystem,
                single_point_inSubsystem - reference_point_inSubsystem
            ):
                single_visible_point_inSubsystem = None
            else:
                single_visible_point_inSubsystem = single_point_inSubsystem

            if single_visible_point_inSubsystem is None and \
                    set_to_none is False:
                continue

            visible_points_inSubsystem = \
                visible_points_inSubsystem + \
                [single_visible_point_inSubsystem, ]

        return visible_points_inSubsystem

    def apply_occluder_inOther(
        self,
        other_node,
        points_inOther,
        reference_point_inOther=None,
        set_to_none=True
    ):
        """
        Filter out from input list of points in the coordinate system of
        other_node those points whose rays pointing towards the reference point
        are occluded by the occluder.

        Args:
            other_node (Node): node in whose coordinate system the input and
            reference points are represented.

            points_inOther (list of torch.Tensor): list of points, each
            represented in the coordinate system of other_node as a (3,)
            torch.Tensor, whose visibility/occlusion is to be tested.

            reference_point_inOther (torch.Tensor, optional): point in the
            coordinate system of other_node which defines the direction for
            which the visibility/occlusion of the points in the input list is
            to be considered. Defaults to None, in which case the optical
            center of the first camera of the subsystem is used.

            set_to_none (bool, optional): If True, points_inOther which are not
            visible are set to None. Otherwise, the points are replaced.
            Defaults to True.

        Returns:
            list of torch.Tensors: if set_to_none is True, returns a list of
            points in the coordinate system of other which is identical to the
            input list but for the occluded points, which are replaced with
            None. Otherwise, returns a shortened version of the input list,
            from which the occluded points were removed.
        """
        if self.occluder is None:
            return points_inOther

        # Transform the points to the coordinate system of the subsystem. Note
        # that from the perspective of other_node, other_node is self, and
        # other is the argument to the method get_transform_toOther_fromSelf.
        transform_toSubsystem_fromOther = \
            other_node.get_transform_toOther_fromSelf(self)
        points_inSubsystem = []
        for single_point_inOther in points_inOther:
            if single_point_inOther is None:
                single_point_inSubsystem = None
            else:
                single_point_inSubsystem = \
                    transform_toSubsystem_fromOther.transform(
                        single_point_inOther
                    )

            points_inSubsystem = \
                points_inSubsystem + [single_point_inSubsystem, ]

        # Transform the reference point as well.
        if reference_point_inOther is None:
            reference_point_inSubsystem = \
                self.cameras[0].get_optical_optical_center_inOther(self)
        else:
            reference_point_inSubsystem = \
                transform_toSubsystem_fromOther.transform(
                    reference_point_inOther
                )

        visible_points_inSubsystem = \
            self.apply_occluder_inSubsystem(
                points_inSubsystem,
                reference_point_inSubsystem,
                set_to_none=True  # If required we shorten the list later.
            )

        new_points_inOther = []
        for single_point_inSubsystem, single_point_inOther \
                in zip(visible_points_inSubsystem, points_inOther):
            if single_point_inSubsystem is None:
                # We shorten the list or set the occluded points to None.
                if set_to_none:
                    new_points_inOther = new_points_inOther + [None, ]
                else:
                    continue
            else:
                new_points_inOther = \
                    new_points_inOther + [single_point_inOther, ]

        return new_points_inOther
